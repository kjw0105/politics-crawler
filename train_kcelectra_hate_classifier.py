import argparse
import inspect
import random
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
except ModuleNotFoundError as e:
    raise SystemExit(
        f"Missing dependency: {e.name}. Install first with:\n"
        "pip install -r requirements_kcelectra.txt"
    ) from e


class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "weighted_precision": p_weighted,
        "weighted_recall": r_weighted,
        "weighted_f1": f1_weighted,
    }

    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=1, zero_division=0
        )
        metrics.update(
            {
                "binary_precision": p_bin,
                "binary_recall": r_bin,
                "binary_f1": f1_bin,
            }
        )

    return metrics


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def build_model_text(row, text_column, keyword_column, title_column, body_column):
    comment = _clean_text(row.get(text_column))
    keyword = _clean_text(row.get(keyword_column))
    title = _clean_text(row.get(title_column))
    body = _clean_text(row.get(body_column))

    if comment:
        parts = []
        if keyword:
            parts.append(f"[KW] {keyword}")
        parts.append(f"[COMMENT] {comment}")
        return " ".join(parts), "comment"

    parts = []
    if keyword:
        parts.append(f"[KW] {keyword}")
    if title:
        parts.append(f"[TITLE] {title}")
    if body:
        parts.append(f"[POST] {body}")
    return " ".join(parts).strip(), "post_fallback"


def parse_label_list(raw: str):
    if raw is None:
        return []
    labels = []
    seen = set()
    for token in str(raw).split(","):
        val = token.strip().lower()
        if not val:
            continue
        if val in seen:
            continue
        labels.append(val)
        seen.add(val)
    return labels


def normalize_label_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train KcELECTRA text classifier (binary or multiclass)."
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default=r"C:\Users\woozz\Desktop\data.xlsx",
        help="Path to Excel file.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="comment_text",
        help="Column name that contains text to classify.",
    )
    parser.add_argument(
        "--keyword_column",
        type=str,
        default="matched_keyword",
        help="Keyword column used as context.",
    )
    parser.add_argument(
        "--title_column",
        type=str,
        default="TITLE",
        help="Post title column used as fallback context.",
    )
    parser.add_argument(
        "--body_column",
        type=str,
        default="TEXT",
        help="Post body column used as fallback context.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="beomi/KcELECTRA-base-v2022",
        help="Hugging Face model name or local model path for initialization.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/kcelectra_hate",
        help="Directory to save checkpoints and the final model.",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Legacy row-order binary mode (kept for backward compatibility)
    parser.add_argument(
        "--non_hate_count",
        type=int,
        default=25,
        help="Legacy mode: number of non-hate rows from top.",
    )
    parser.add_argument(
        "--hate_count",
        type=int,
        default=25,
        help="Legacy mode: number of hate rows after non_hate_count.",
    )

    # Label-column mode (recommended)
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help=(
            "If provided, labels are read from this column. "
            "This mode supports multiclass."
        ),
    )
    parser.add_argument(
        "--class_labels",
        type=str,
        default="no,derivation,misogyny",
        help=(
            "Comma-separated label list and class order for label_column mode. "
            "Example: no,derivation,misogyny"
        ),
    )
    parser.add_argument(
        "--exclude_labels",
        type=str,
        default="cannot judgment",
        help="Comma-separated labels to exclude in label_column mode.",
    )
    return parser


def build_dataset_from_label_column(df: pd.DataFrame, args):
    if args.label_column not in df.columns:
        raise ValueError(
            f"label_column '{args.label_column}' not found. Available: {list(df.columns)}"
        )

    class_names = parse_label_list(args.class_labels)
    if len(class_names) < 2:
        raise ValueError("--class_labels must contain at least 2 labels.")

    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    exclude_set = set(parse_label_list(args.exclude_labels))

    labels_raw = df[args.label_column].map(normalize_label_value)
    not_blank = labels_raw != ""
    not_excluded = ~labels_raw.isin(exclude_set)
    in_defined_classes = labels_raw.isin(class_to_id.keys())
    keep_mask = not_blank & not_excluded & in_defined_classes

    unknown_labels = sorted(
        set(labels_raw[not_blank & not_excluded].unique()) - set(class_to_id.keys())
    )
    if unknown_labels:
        print(f"Warning: unknown labels dropped: {unknown_labels}")

    selected_df = df.loc[keep_mask].copy().reset_index(drop=True)
    selected_labels_raw = labels_raw[keep_mask].reset_index(drop=True)
    label_ids = selected_labels_raw.map(class_to_id).astype(int).tolist()

    model_texts = []
    text_sources = []
    for _, row in selected_df.iterrows():
        model_text, source = build_model_text(
            row=row,
            text_column=args.text_column,
            keyword_column=args.keyword_column,
            title_column=args.title_column,
            body_column=args.body_column,
        )
        model_texts.append(model_text)
        text_sources.append(source)

    data = pd.DataFrame(
        {
            "text": model_texts,
            "label": label_ids,
            "label_name": selected_labels_raw.tolist(),
            "text_source": text_sources,
        }
    )

    empty_count = int((data["text"] == "").sum())
    if empty_count > 0:
        print(
            "Warning: found "
            f"{empty_count} rows that are empty even after fallback. They will be excluded."
        )
        data = data[data["text"] != ""].reset_index(drop=True)

    return data, class_names


def build_dataset_legacy_row_order(df: pd.DataFrame, args):
    total_rows = len(df)
    if total_rows <= args.non_hate_count:
        raise ValueError(
            f"Need more than {args.non_hate_count} rows so both classes exist, "
            f"but found {total_rows}."
        )

    requested_total = args.non_hate_count + args.hate_count
    usable_total = min(total_rows, requested_total)
    actual_hate_count = usable_total - args.non_hate_count

    if usable_total < requested_total:
        print(
            f"Warning: requested {requested_total} rows "
            f"({args.non_hate_count} non-hate + {args.hate_count} hate), "
            f"but only {usable_total} rows available. "
            f"Using {args.non_hate_count} non-hate + {actual_hate_count} hate."
        )

    selected_df = df.iloc[:usable_total].copy()
    model_texts = []
    text_sources = []
    for _, row in selected_df.iterrows():
        model_text, source = build_model_text(
            row=row,
            text_column=args.text_column,
            keyword_column=args.keyword_column,
            title_column=args.title_column,
            body_column=args.body_column,
        )
        model_texts.append(model_text)
        text_sources.append(source)

    labels = [0] * args.non_hate_count + [1] * actual_hate_count
    label_names = ["non_hate"] * args.non_hate_count + ["hate"] * actual_hate_count

    data = pd.DataFrame(
        {
            "text": model_texts,
            "label": labels,
            "label_name": label_names,
            "text_source": text_sources,
        }
    )

    empty_count = int((data["text"] == "").sum())
    if empty_count > 0:
        print(
            "Warning: found "
            f"{empty_count} rows that are empty even after fallback. They will be excluded."
        )
        data = data[data["text"] != ""].reset_index(drop=True)

    return data, ["non_hate", "hate"]


def main():
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    excel_path = Path(args.excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    if args.text_column not in df.columns:
        raise ValueError(
            f"Column '{args.text_column}' not found. Available columns: {list(df.columns)}"
        )

    if args.label_column:
        data, class_names = build_dataset_from_label_column(df, args)
        print(f"Mode: label_column ({args.label_column})")
    else:
        data, class_names = build_dataset_legacy_row_order(df, args)
        print("Mode: legacy_row_order_binary")

    source_counts = data["text_source"].value_counts().to_dict()
    class_counts = data["label_name"].value_counts().to_dict()
    print(f"Input source counts: {source_counts}")
    print(f"Usable samples after cleanup: {len(data)}")
    print(f"Class counts: {class_counts}")
    print(f"Class mapping: { {i: n for i, n in enumerate(class_names)} }")

    label_id_counts = data["label"].value_counts()
    if len(label_id_counts) < 2:
        raise ValueError("After filtering, only one class remains.")
    if (label_id_counts < 2).any():
        raise ValueError(
            "Each class needs at least 2 samples after cleanup for stratified split. "
            f"Current counts: {label_id_counts.to_dict()}"
        )

    texts = data["text"].tolist()
    labels = data["label"].tolist()

    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=args.eval_size,
            random_state=args.seed,
            stratify=labels,
        )
    except ValueError as e:
        raise ValueError(
            f"Train/validation split failed ({e}). "
            "Try lowering --eval_size or adding more labeled rows."
        ) from e

    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=False, max_length=args.max_length
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=False, max_length=args.max_length
    )

    train_dataset = TextClassificationDataset(train_encodings, train_labels)
    val_dataset = TextClassificationDataset(val_encodings, val_labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs_base = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": args.seed,
    }

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    ta_param_names = set(ta_params.keys())
    training_kwargs = {
        k: v for k, v in training_kwargs_base.items() if k in ta_param_names
    }
    ignored = [k for k in training_kwargs_base.keys() if k not in ta_param_names]
    if ignored:
        print(f"Note: ignoring unsupported TrainingArguments keys: {ignored}")

    if "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Train size: {len(train_dataset)}, Eval size: {len(val_dataset)}")

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("Final eval metrics:", eval_metrics)

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved model/tokenizer to: {final_dir.resolve()}")


if __name__ == "__main__":
    main()
