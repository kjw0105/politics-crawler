import argparse
import glob
import os
import pandas as pd


REQUIRED_COLS = ["date", "platform", "source", "keyword", "text"]


def load_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 컬럼 호환 처리(혹시 없으면 만들어두기)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = ""

    # 날짜 파싱(UTC/로컬 혼합 대비: 일단 pandas가 파싱하게 두고 NaT 제거)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # 공백 정리
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df["platform"] = df["platform"].astype(str).str.strip()

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_glob", default="data/raw/youtube*/**/*.csv",
                   help='예: data/raw/youtube*/**/*.csv')
    p.add_argument("--out", default="data/processed/merged_youtube_comments.csv")
    args = p.parse_args()

    paths = sorted(glob.glob(args.in_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched: {args.in_glob}")

    dfs = []
    for path in paths:
        try:
            df = load_one_csv(path)
            df["__srcfile"] = os.path.basename(path)
            dfs.append(df)
        except Exception as e:
            print(f"[SKIP] {path}: {e}")

    merged = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=REQUIRED_COLS)
    if merged.empty:
        raise RuntimeError("Merged dataframe is empty. Check your input CSVs.")

    # ✅ 핵심: 중복 제거(같은 날짜/영상/댓글 텍스트/키워드가 여러번 저장되는 문제 방지)
    # video_id 컬럼이 있다면 포함, 없으면 안전한 키로만 제거
    key_cols = ["date", "keyword", "text"]
    if "video_id" in merged.columns:
        key_cols.insert(1, "video_id")

    merged = merged.sort_values("date")
    merged = merged.drop_duplicates(subset=key_cols, keep="first")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Saved merged -> {args.out} (rows={len(merged)}, files={len(paths)})")


if __name__ == "__main__":
    main()
