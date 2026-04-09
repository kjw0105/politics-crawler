import argparse
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError as e:
    raise SystemExit(
        f"Missing dependency: {e.name}. Install first with:\n"
        "python -m pip install -r requirements_kcelectra.txt"
    ) from e


def build_parser():
    parser = argparse.ArgumentParser(
        description="Filter rows from a CSV where comment text contains target keywords."
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV path.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path. Default: <input_stem>_filtered.csv",
    )
    parser.add_argument(
        "--comment_column",
        type=str,
        default="text",
        help="Comment text column name in input CSV.",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Keywords to filter by. Supports comma-separated values.",
    )
    parser.add_argument(
        "--keywords_file",
        type=str,
        default=None,
        help="Text file with one keyword per line.",
    )
    parser.add_argument(
        "--keyword_match_mode",
        choices=["any", "all"],
        default="any",
        help="any: at least one keyword, all: must contain all keywords.",
    )
    parser.add_argument(
        "--case_sensitive",
        action="store_true",
        help="Enable case-sensitive matching.",
    )
    return parser


def read_csv_with_fallback(path: Path):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError as e:
            last_error = e
    raise ValueError(
        f"Failed to read CSV with tried encodings {encodings}. Last error: {last_error}"
    )


def normalize_keywords(raw_keywords, keywords_file):
    merged = []

    if raw_keywords:
        for item in raw_keywords:
            if item is None:
                continue
            parts = [p.strip() for p in str(item).split(",")]
            merged.extend([p for p in parts if p])

    if keywords_file:
        fp = Path(keywords_file)
        if not fp.exists():
            raise FileNotFoundError(f"keywords_file not found: {fp}")
        for line in fp.read_text(encoding="utf-8").splitlines():
            kw = line.strip()
            if kw:
                merged.append(kw)

    deduped = []
    seen = set()
    for kw in merged:
        if kw not in seen:
            deduped.append(kw)
            seen.add(kw)
    return deduped


def find_matched_keywords(text: str, keywords, case_sensitive: bool):
    if case_sensitive:
        return [kw for kw in keywords if kw in text]

    low = text.lower()
    return [kw for kw in keywords if kw.lower() in low]


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    keywords = normalize_keywords(args.keywords, args.keywords_file)
    if not keywords:
        raise ValueError("No keywords found. Use --keywords and/or --keywords_file.")

    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else input_csv.with_name(f"{input_csv.stem}_filtered.csv")
    )

    df, enc = read_csv_with_fallback(input_csv)
    if args.comment_column not in df.columns:
        raise ValueError(
            f"comment_column '{args.comment_column}' not found. Available: {list(df.columns)}"
        )

    comment_series = df[args.comment_column].fillna("").astype(str)
    matched_lists = comment_series.apply(
        lambda t: find_matched_keywords(
            text=t, keywords=keywords, case_sensitive=args.case_sensitive
        )
    )

    if args.keyword_match_mode == "all":
        mask = matched_lists.apply(lambda items: len(items) == len(keywords))
    else:
        mask = matched_lists.apply(lambda items: len(items) > 0)

    filtered = df.loc[mask].copy()
    filtered.insert(0, "source_row_index", filtered.index.astype(int))
    filtered = filtered.reset_index(drop=True)
    filtered["matched_search_keywords"] = ["|".join(x) for x in matched_lists[mask]]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Loaded CSV: {input_csv.resolve()} (encoding={enc}), rows={len(df)}")
    print(f"Filtering by {len(keywords)} keywords, mode={args.keyword_match_mode}")
    print(f"Filtered rows: {len(filtered)} / {len(df)}")
    print(f"Saved filtered CSV: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
