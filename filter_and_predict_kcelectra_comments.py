import argparse
from pathlib import Path

try:
    import pandas as pd
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ModuleNotFoundError as e:
    raise SystemExit(
        f"Missing dependency: {e.name}. Install first with:\n"
        "pip install -r requirements_kcelectra.txt"
    ) from e

from predict_kcelectra_hate_classifier import (
    build_model_text,
    predict_texts,
    resolve_device,
    summarize_decision,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Filter comments by keywords from a CSV, then classify with trained KcELECTRA."
        )
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to crawled comments CSV file.",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Keywords to search. Supports comma-separated values as well.",
    )
    parser.add_argument(
        "--keywords_file",
        type=str,
        default=None,
        help="Optional text file path (one keyword per line).",
    )
    parser.add_argument(
        "--comment_column",
        type=str,
        default="comment_text",
        help="Column used for keyword filtering.",
    )
    parser.add_argument(
        "--keyword_match_mode",
        type=str,
        choices=["any", "all"],
        default="any",
        help="any: keep rows containing at least one keyword, all: require all keywords.",
    )
    parser.add_argument(
        "--case_sensitive",
        action="store_true",
        help="Use case-sensitive keyword matching.",
    )
    parser.add_argument(
        "--output_filtered_csv",
        type=str,
        default=None,
        help="Path to save keyword-filtered CSV.",
    )
    parser.add_argument(
        "--output_predicted_csv",
        type=str,
        default=None,
        help="Path to save predicted CSV.",
    )

    # Prediction args (same defaults as existing predictor)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./outputs/kcelectra_hate/final_model",
        help="Path to trained model directory.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help=(
            "Comment column used for model input. "
            "If omitted, comment_column is used."
        ),
    )
    parser.add_argument(
        "--keyword_column",
        type=str,
        default="matched_keyword",
        help="Keyword context column used for model input fallback.",
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
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on hate probability.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    return parser


def infer_filtered_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_filtered.csv")


def infer_predicted_path(filtered_csv: Path) -> Path:
    return filtered_csv.with_name(f"{filtered_csv.stem}_predicted.csv")


def read_csv_with_fallback(input_csv: Path):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(input_csv, encoding=enc), enc
        except UnicodeDecodeError as e:
            last_error = e
    raise ValueError(
        f"Failed to read CSV with tried encodings: {encodings}. Last error: {last_error}"
    )


def normalize_keywords(raw_keywords, keywords_file):
    collected = []

    if raw_keywords:
        for item in raw_keywords:
            if item is None:
                continue
            parts = [p.strip() for p in str(item).split(",")]
            collected.extend([p for p in parts if p])

    if keywords_file:
        fp = Path(keywords_file)
        if not fp.exists():
            raise FileNotFoundError(f"keywords_file not found: {fp}")
        for line in fp.read_text(encoding="utf-8").splitlines():
            kw = line.strip()
            if kw:
                collected.append(kw)

    deduped = []
    seen = set()
    for kw in collected:
        if kw not in seen:
            deduped.append(kw)
            seen.add(kw)
    return deduped


def find_matched_keywords(text: str, keywords, case_sensitive: bool):
    if case_sensitive:
        return [kw for kw in keywords if kw in text]

    lowered_text = text.lower()
    matches = []
    for kw in keywords:
        if kw.lower() in lowered_text:
            matches.append(kw)
    return matches


def filter_rows_by_keywords(
    df: pd.DataFrame,
    comment_column: str,
    keywords,
    keyword_match_mode: str,
    case_sensitive: bool,
) -> pd.DataFrame:
    if comment_column not in df.columns:
        raise ValueError(
            f"comment_column '{comment_column}' not found. Available: {list(df.columns)}"
        )

    comment_series = df[comment_column].fillna("").astype(str)
    matched_lists = comment_series.apply(
        lambda text: find_matched_keywords(
            text=text, keywords=keywords, case_sensitive=case_sensitive
        )
    )

    if keyword_match_mode == "all":
        mask = matched_lists.apply(lambda items: len(items) == len(keywords))
    else:
        mask = matched_lists.apply(lambda items: len(items) > 0)

    filtered = df.loc[mask].copy()
    filtered.insert(0, "source_row_index", filtered.index.astype(int))
    filtered = filtered.reset_index(drop=True)

    selected_matches = matched_lists[mask].tolist()
    filtered["matched_search_keywords"] = ["|".join(items) for items in selected_matches]
    return filtered


def attach_prediction_columns(
    filtered_df: pd.DataFrame,
    tokenizer,
    model,
    args,
    device,
) -> pd.DataFrame:
    model_texts = []
    text_sources = []

    for _, row in filtered_df.iterrows():
        model_text, source = build_model_text(
            row=row,
            text_column=args.text_column,
            keyword_column=args.keyword_column,
            title_column=args.title_column,
            body_column=args.body_column,
        )
        model_texts.append(model_text)
        text_sources.append(source)

    pred_col = [None] * len(filtered_df)
    pred_name_col = [None] * len(filtered_df)
    hate_prob_col = [None] * len(filtered_df)
    margin_col = [None] * len(filtered_df)
    confidence_col = [None] * len(filtered_df)
    reason_col = [None] * len(filtered_df)

    valid_indices = [i for i, text in enumerate(model_texts) if text != ""]
    valid_texts = [model_texts[i] for i in valid_indices]

    if valid_texts:
        probs = predict_texts(
            valid_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        for idx, prob_item in zip(valid_indices, probs):
            if isinstance(prob_item, (list, tuple)):
                if len(prob_item) < 2:
                    raise ValueError(
                        "Model output has fewer than 2 classes; expected binary model."
                    )
                hate_prob = float(prob_item[1])
            else:
                hate_prob = float(prob_item)
            pred, pred_name, margin, confidence, reason = summarize_decision(
                hate_prob=hate_prob, threshold=args.threshold
            )
            pred_col[idx] = int(pred)
            pred_name_col[idx] = pred_name
            hate_prob_col[idx] = hate_prob
            margin_col[idx] = float(margin)
            confidence_col[idx] = confidence
            reason_col[idx] = reason

    out_df = filtered_df.copy()
    out_df["model_input_text"] = model_texts
    out_df["input_source"] = text_sources
    out_df["pred_label"] = pred_col
    out_df["pred_name"] = pred_name_col
    out_df["hate_prob"] = hate_prob_col
    out_df["decision_margin"] = margin_col
    out_df["confidence"] = confidence_col
    out_df["reason_summary"] = reason_col
    return out_df


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.text_column is None:
        args.text_column = args.comment_column

    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1.")

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    keywords = normalize_keywords(args.keywords, args.keywords_file)
    if not keywords:
        raise ValueError("No keywords found. Use --keywords and/or --keywords_file.")

    df, detected_encoding = read_csv_with_fallback(input_csv)
    print(
        f"Loaded CSV: {input_csv.resolve()} (encoding={detected_encoding}), rows={len(df)}"
    )
    print(f"Filtering by {len(keywords)} keywords, mode={args.keyword_match_mode}")

    filtered_df = filter_rows_by_keywords(
        df=df,
        comment_column=args.comment_column,
        keywords=keywords,
        keyword_match_mode=args.keyword_match_mode,
        case_sensitive=args.case_sensitive,
    )

    filtered_path = (
        Path(args.output_filtered_csv)
        if args.output_filtered_csv
        else infer_filtered_path(input_csv)
    )
    save_csv(filtered_df, filtered_path)
    print(f"Filtered rows: {len(filtered_df)} / {len(df)}")
    print(f"Saved filtered CSV: {filtered_path.resolve()}")

    predicted_path = (
        Path(args.output_predicted_csv)
        if args.output_predicted_csv
        else infer_predicted_path(filtered_path)
    )

    if filtered_df.empty:
        empty_out = attach_prediction_columns(
            filtered_df=filtered_df,
            tokenizer=None,
            model=None,
            args=args,
            device=None,
        )
        save_csv(empty_out, predicted_path)
        print("No rows matched keywords. Saved empty prediction CSV.")
        print(f"Saved predicted CSV: {predicted_path.resolve()}")
        return

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    out_df = attach_prediction_columns(
        filtered_df=filtered_df,
        tokenizer=tokenizer,
        model=model,
        args=args,
        device=device,
    )
    save_csv(out_df, predicted_path)
    print(f"Saved predicted CSV: {predicted_path.resolve()}")


if __name__ == "__main__":
    main()
