import argparse
import re
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ModuleNotFoundError as e:
    raise SystemExit(
        f"Missing dependency: {e.name}. Install first with:\n"
        "pip install -r requirements_kcelectra.txt"
    ) from e


def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict labels using a trained KcELECTRA model (binary or multiclass)."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./outputs/kcelectra_hate/final_model",
        help="Path to trained model directory.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text input for quick test.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Batch input file path (.xlsx or .csv).",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="comment_text",
        help="Comment column in batch file.",
    )
    parser.add_argument(
        "--keyword_column",
        type=str,
        default="matched_keyword",
        help="Keyword column used as context for fallback.",
    )
    parser.add_argument(
        "--title_column",
        type=str,
        default="TITLE",
        help="Post title column used when comment is empty.",
    )
    parser.add_argument(
        "--body_column",
        type=str,
        default="TEXT",
        help="Post body column used when comment is empty.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="Optional label column for metrics.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save prediction result (.xlsx or .csv).",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary mode threshold on positive class probability.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="For --text mode (binary only), print token-level explanation.",
    )
    parser.add_argument(
        "--top_k_tokens",
        type=int,
        default=5,
        help="Number of influential tokens to print in explanation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def predict_texts(texts, tokenizer, model, device, max_length=128, batch_size=32):
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
    return all_probs


def get_label_names(model):
    id2label_cfg = model.config.id2label or {}
    label_names = []
    for i in range(int(model.config.num_labels)):
        if i in id2label_cfg:
            label_names.append(str(id2label_cfg[i]))
        elif str(i) in id2label_cfg:
            label_names.append(str(id2label_cfg[str(i)]))
        else:
            label_names.append(str(i))
    return label_names


def safe_label_for_column(label: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(label).strip().lower()).strip("_")


def is_binary(label_names):
    return len(label_names) == 2


def find_positive_label_index(label_names):
    lowered = [x.lower() for x in label_names]
    if "hate" in lowered:
        return lowered.index("hate")
    return 1


def find_negative_label_index(label_names, positive_index):
    for idx in range(len(label_names)):
        if idx != positive_index:
            return idx
    return 0


def summarize_binary_decision(
    probs, label_names, threshold: float, positive_idx: int, negative_idx: int
):
    positive_prob = float(probs[positive_idx])
    pred_idx = positive_idx if positive_prob >= threshold else negative_idx
    pred_name = label_names[pred_idx]
    margin = abs(positive_prob - threshold)

    if margin < 0.05:
        confidence = "low"
    elif margin < 0.15:
        confidence = "medium"
    else:
        confidence = "high"

    reason = (
        f"{label_names[positive_idx]}_prob={positive_prob:.4f} "
        f"{'>=' if pred_idx == positive_idx else '<'} threshold={threshold:.2f}, "
        f"margin={margin:.4f}, confidence={confidence}"
    )
    return pred_idx, pred_name, positive_prob, margin, confidence, reason


def summarize_decision(hate_prob: float, threshold: float):
    pred = 1 if hate_prob >= threshold else 0
    pred_name = "hate" if pred == 1 else "non_hate"
    margin = abs(hate_prob - threshold)
    if margin < 0.05:
        confidence = "low"
    elif margin < 0.15:
        confidence = "medium"
    else:
        confidence = "high"
    reason = (
        f"hate_prob={hate_prob:.4f} "
        f"{'>=' if pred == 1 else '<'} threshold={threshold:.2f}, "
        f"margin={margin:.4f}, confidence={confidence}"
    )
    return pred, pred_name, margin, confidence, reason


def summarize_multiclass_decision(probs, label_names):
    probs_np = np.array(probs, dtype=float)
    pred_idx = int(np.argmax(probs_np))
    pred_name = label_names[pred_idx]
    pred_prob = float(probs_np[pred_idx])

    sorted_probs = np.sort(probs_np)
    second_prob = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
    gap = pred_prob - second_prob

    if gap < 0.05:
        confidence = "low"
    elif gap < 0.15:
        confidence = "medium"
    else:
        confidence = "high"

    reason = (
        f"top={pred_name}({pred_prob:.4f}), "
        f"second={second_prob:.4f}, gap={gap:.4f}, confidence={confidence}"
    )
    return pred_idx, pred_name, pred_prob, gap, confidence, reason


def explain_single_text_binary(
    text,
    tokenizer,
    model,
    device,
    target_idx,
    max_length=128,
    top_k_tokens=5,
):
    model.eval()
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        base_probs = torch.softmax(base_logits, dim=-1)[0]

    base_target_prob = float(base_probs[target_idx].item())
    base_pred = int(torch.argmax(base_probs).item())

    special_ids = set(tokenizer.all_special_ids)
    mask_id = tokenizer.mask_token_id
    fallback_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    token_ids = input_ids[0].detach().cpu().tolist()
    token_impacts = []

    for pos, tok_id in enumerate(token_ids):
        if tok_id in special_ids:
            continue

        ablated_ids = input_ids.clone()
        ablated_ids[0, pos] = mask_id if mask_id is not None else fallback_id

        with torch.no_grad():
            ablated_logits = model(
                input_ids=ablated_ids, attention_mask=attention_mask
            ).logits
            ablated_probs = torch.softmax(ablated_logits, dim=-1)[0]

        ablated_target_prob = float(ablated_probs[target_idx].item())
        delta = base_target_prob - ablated_target_prob

        token_text = tokenizer.convert_ids_to_tokens(tok_id)
        token_text = token_text.replace("##", "")
        token_text = token_text.replace("\u2581", "")
        token_text = token_text.strip()
        if not token_text:
            continue

        token_impacts.append(
            {
                "token": token_text,
                "delta_target_prob": delta,
                "ablated_target_prob": ablated_target_prob,
            }
        )

    positive_support = sorted(
        [x for x in token_impacts if x["delta_target_prob"] > 0],
        key=lambda x: x["delta_target_prob"],
        reverse=True,
    )[:top_k_tokens]
    negative_support = sorted(
        [x for x in token_impacts if x["delta_target_prob"] < 0],
        key=lambda x: x["delta_target_prob"],
    )[:top_k_tokens]

    return {
        "base_pred": base_pred,
        "base_target_prob": base_target_prob,
        "positive_support": positive_support,
        "negative_support": negative_support,
    }


def build_label_alias_map(label_names):
    alias = {}
    for idx, name in enumerate(label_names):
        alias[str(name).strip().lower()] = idx

    if "non_hate" in alias:
        alias["0"] = alias["non_hate"]
        alias["비혐오"] = alias["non_hate"]
        alias["아니다"] = alias["non_hate"]
    if "hate" in alias:
        alias["1"] = alias["hate"]
        alias["혐오"] = alias["hate"]
        alias["맞다"] = alias["hate"]
    if "no" in alias:
        alias["0"] = alias["no"]
    if "misogyny" in alias:
        alias["여성혐오"] = alias["misogyny"]
    return alias


def parse_label_value(value, label_alias_map, label_count):
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in label_alias_map:
        return int(label_alias_map[s])
    if s.isdigit():
        n = int(s)
        if 0 <= n < label_count:
            return n
    return None


def infer_output_path(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".csv":
        return input_path.with_name(input_path.stem + "_predicted.csv")
    return input_path.with_name(input_path.stem + "_predicted.xlsx")


def read_table(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(input_path)
    raise ValueError("input_path must be .xlsx/.xls/.xlsm or .csv")


def save_table(df: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return
    if suffix in [".xlsx", ".xlsm", ".xls"]:
        df.to_excel(output_path, index=False)
        return
    raise ValueError("output_path must be .xlsx/.xls/.xlsm or .csv")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.text is None and args.input_path is None:
        raise ValueError("Provide either --text or --input_path.")
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1.")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    label_names = get_label_names(model)
    binary_mode = is_binary(label_names)
    print(f"Model labels: {label_names}")

    positive_idx = None
    negative_idx = None
    if binary_mode:
        positive_idx = find_positive_label_index(label_names)
        negative_idx = find_negative_label_index(label_names, positive_idx)

    if args.text is not None:
        probs = predict_texts(
            [args.text],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            batch_size=1,
        )[0]

        if binary_mode:
            pred, pred_name, positive_prob, margin, confidence, reason = (
                summarize_binary_decision(
                    probs=probs,
                    label_names=label_names,
                    threshold=args.threshold,
                    positive_idx=positive_idx,
                    negative_idx=negative_idx,
                )
            )
            print("Input:", args.text)
            print(f"Prediction: {pred_name} ({pred})")
            print(
                f"{label_names[positive_idx]} probability: "
                f"{positive_prob:.4f}"
            )
            print("Reason summary:", reason)
        else:
            pred, pred_name, pred_prob, gap, confidence, reason = (
                summarize_multiclass_decision(
                    probs=probs,
                    label_names=label_names,
                )
            )
            print("Input:", args.text)
            print(f"Prediction: {pred_name} ({pred})")
            print(f"Top probability: {pred_prob:.4f}")
            print("Reason summary:", reason)

        print("Class probabilities:")
        for idx, name in enumerate(label_names):
            print(f"- {name}: {float(probs[idx]):.4f}")

        if args.explain:
            if not binary_mode:
                print("Token explanation is supported only for binary models.")
            else:
                exp = explain_single_text_binary(
                    text=args.text,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    target_idx=positive_idx,
                    max_length=args.max_length,
                    top_k_tokens=args.top_k_tokens,
                )
                print("Explanation (token impact by ablation):")
                print(
                    "Token impact = base positive_prob - positive_prob after masking that token."
                )

                if pred == positive_idx:
                    support = exp["positive_support"]
                    oppose = exp["negative_support"]
                    print(
                        f"Top tokens supporting {label_names[positive_idx]} prediction:"
                    )
                else:
                    support = exp["negative_support"]
                    oppose = exp["positive_support"]
                    print(
                        f"Top tokens supporting {label_names[negative_idx]} prediction:"
                    )

                if support:
                    for item in support:
                        print(
                            f"- {item['token']}: {item['delta_target_prob']:+.4f} "
                            f"(masked positive_prob={item['ablated_target_prob']:.4f})"
                        )
                else:
                    print("- No strong supporting token found.")

                if oppose:
                    print("Top counter-evidence tokens:")
                    for item in oppose[: args.top_k_tokens]:
                        print(
                            f"- {item['token']}: {item['delta_target_prob']:+.4f} "
                            f"(masked positive_prob={item['ablated_target_prob']:.4f})"
                        )

                print(f"Confidence level: {confidence}")

    if args.input_path is not None:
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = read_table(input_path)
        if args.text_column not in df.columns:
            raise ValueError(
                f"Column '{args.text_column}' not found. Available: {list(df.columns)}"
            )

        model_texts = []
        text_sources = []
        for _, row in df.iterrows():
            model_text, source = build_model_text(
                row=row,
                text_column=args.text_column,
                keyword_column=args.keyword_column,
                title_column=args.title_column,
                body_column=args.body_column,
            )
            model_texts.append(model_text)
            text_sources.append(source)

        valid_positions = [i for i, t in enumerate(model_texts) if t != ""]
        valid_texts = [model_texts[i] for i in valid_positions]

        pred_col = [None] * len(df)
        pred_name_col = [None] * len(df)
        pred_prob_col = [None] * len(df)
        margin_col = [None] * len(df)
        confidence_col = [None] * len(df)
        reason_col = [None] * len(df)
        positive_prob_col = [None] * len(df)  # binary-only helper

        prob_columns = {}
        for label in label_names:
            col = f"prob_{safe_label_for_column(label)}"
            prob_columns[col] = [None] * len(df)

        if valid_texts:
            probs_list = predict_texts(
                valid_texts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )

            for pos, probs in zip(valid_positions, probs_list):
                if binary_mode:
                    pred, pred_name, positive_prob, margin, confidence, reason = (
                        summarize_binary_decision(
                            probs=probs,
                            label_names=label_names,
                            threshold=args.threshold,
                            positive_idx=positive_idx,
                            negative_idx=negative_idx,
                        )
                    )
                    pred_prob = float(max(probs))
                    pred_col[pos] = int(pred)
                    pred_name_col[pos] = pred_name
                    pred_prob_col[pos] = pred_prob
                    positive_prob_col[pos] = float(positive_prob)
                    margin_col[pos] = float(margin)
                    confidence_col[pos] = confidence
                    reason_col[pos] = reason
                else:
                    pred, pred_name, pred_prob, margin, confidence, reason = (
                        summarize_multiclass_decision(
                            probs=probs,
                            label_names=label_names,
                        )
                    )
                    pred_col[pos] = int(pred)
                    pred_name_col[pos] = pred_name
                    pred_prob_col[pos] = float(pred_prob)
                    margin_col[pos] = float(margin)
                    confidence_col[pos] = confidence
                    reason_col[pos] = reason

                for class_idx, label in enumerate(label_names):
                    col = f"prob_{safe_label_for_column(label)}"
                    prob_columns[col][pos] = float(probs[class_idx])

        out_df = df.copy()
        out_df["model_input_text"] = model_texts
        out_df["input_source"] = text_sources
        out_df["pred_label"] = pred_col
        out_df["pred_name"] = pred_name_col
        out_df["pred_prob"] = pred_prob_col
        out_df["decision_margin"] = margin_col
        out_df["confidence"] = confidence_col
        out_df["reason_summary"] = reason_col

        if binary_mode:
            out_df[f"{safe_label_for_column(label_names[positive_idx])}_prob"] = (
                positive_prob_col
            )
            if label_names[positive_idx].lower() == "hate":
                out_df["hate_prob"] = positive_prob_col
            else:
                out_df["hate_prob"] = [None] * len(df)
        else:
            out_df["hate_prob"] = [None] * len(df)

        for col_name, values in prob_columns.items():
            out_df[col_name] = values

        print(f"Batch rows: {len(df)}, non-empty rows predicted: {len(valid_texts)}")
        print(
            f"Input source counts: {pd.Series(text_sources).value_counts().to_dict()}"
        )
        pred_dist = pd.Series([x for x in pred_name_col if x is not None]).value_counts()
        print(f"Prediction distribution: {pred_dist.to_dict()}")

        if args.label_column is not None:
            if args.label_column not in out_df.columns:
                raise ValueError(
                    f"label_column '{args.label_column}' not found. "
                    f"Available: {list(out_df.columns)}"
                )
            alias_map = build_label_alias_map(label_names)
            true_labels = out_df[args.label_column].apply(
                lambda x: parse_label_value(
                    x,
                    label_alias_map=alias_map,
                    label_count=len(label_names),
                )
            )
            metric_mask = true_labels.notna() & out_df["pred_label"].notna()
            if int(metric_mask.sum()) > 0:
                y_true = true_labels[metric_mask].astype(int).tolist()
                y_pred = out_df.loc[metric_mask, "pred_label"].astype(int).tolist()
                acc = accuracy_score(y_true, y_pred)
                p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="macro", zero_division=0
                )
                print(
                    f"Metrics on {len(y_true)} rows -> "
                    f"accuracy: {acc:.4f}, "
                    f"macro_precision: {p_macro:.4f}, "
                    f"macro_recall: {r_macro:.4f}, "
                    f"macro_f1: {f1_macro:.4f}"
                )

                if binary_mode:
                    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
                        y_true,
                        y_pred,
                        average="binary",
                        pos_label=positive_idx,
                        zero_division=0,
                    )
                    print(
                        f"Binary metrics (positive={label_names[positive_idx]}) -> "
                        f"precision: {p_bin:.4f}, recall: {r_bin:.4f}, f1: {f1_bin:.4f}"
                    )
            else:
                print("No valid labels found for metrics for current model classes.")

        output_path = (
            Path(args.output_path) if args.output_path else infer_output_path(input_path)
        )
        save_table(out_df, output_path)
        print(f"Saved prediction file: {output_path.resolve()}")


if __name__ == "__main__":
    main()
