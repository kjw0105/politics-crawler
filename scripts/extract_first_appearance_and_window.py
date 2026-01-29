import argparse
import os
import pandas as pd


DEFAULT_ID_COLS = ["url", "video_id", "article_url", "comment_id"]


def pick_id_col(df: pd.DataFrame):
    for c in DEFAULT_ID_COLS:
        if c in df.columns:
            return c
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="통합 댓글 CSV (여러 플랫폼 합친 파일이면 더 좋음)")
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--window_days", type=int, default=7, help="최초 등장 이후 N일 동안의 댓글 덤프")
    p.add_argument("--dedup", action="store_true", help="중복 제거 수행")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # 기본 정리
    for col in ["platform", "source", "keyword", "text"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()

    df = df[df["keyword"] != ""]
    df = df[df["text"] != ""]

    id_col = pick_id_col(df)

    # 중복 제거(선택)
    if args.dedup:
        key_cols = ["date", "platform", "keyword", "text"]
        if id_col:
            key_cols.insert(2, id_col)
        df = df.sort_values("date").drop_duplicates(subset=key_cols, keep="first")

    # ✅ 1) 플랫폼×키워드별 최초 등장(최초 관측)
    group_cols = ["platform", "keyword"]
    df_sorted = df.sort_values("date")

    first_rows = (
        df_sorted.groupby(group_cols, as_index=False)
        .first()  # date 기준으로 정렬되어 있으니 first()가 최초
    )

    # first_hits.csv 만들기
    keep_cols = ["platform", "keyword", "date", "source", "text"]
    if id_col:
        keep_cols.insert(4, id_col)
    first_hits = first_rows[keep_cols].rename(columns={"date": "first_date", "source": "first_source", "text": "first_text"})
    first_hits_path = os.path.join(args.out_dir, "first_hits.csv")
    first_hits.to_csv(first_hits_path, index=False, encoding="utf-8-sig")

    # ✅ 2) 최초 등장 시점부터 N일(window) 댓글 덤프
    window_rows = []
    for _, r in first_hits.iterrows():
        platform = r["platform"]
        keyword = r["keyword"]
        start = pd.to_datetime(r["first_date"])
        end = start + pd.Timedelta(days=args.window_days)

        sub = df[
            (df["platform"] == platform) &
            (df["keyword"] == keyword) &
            (df["date"] >= start) &
            (df["date"] < end)
        ].copy()

        sub["window_start"] = start
        sub["window_end"] = end
        window_rows.append(sub)

        # ✅ 3) 최초 등장일(하루) 덤프도 별도 저장
        day = start.date()
        sub_day = df[
            (df["platform"] == platform) &
            (df["keyword"] == keyword) &
            (df["date"].dt.date == day)
        ].copy()

        out_day_dir = os.path.join(args.out_dir, "first_day_dump")
        os.makedirs(out_day_dir, exist_ok=True)
        out_day_path = os.path.join(out_day_dir, f"{platform}__{keyword}__{day}.csv")
        sub_day.sort_values("date").to_csv(out_day_path, index=False, encoding="utf-8-sig")

    first_window = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    first_window_path = os.path.join(args.out_dir, f"first_window_comments_{args.window_days}d.csv")
    first_window.sort_values(["platform", "keyword", "date"]).to_csv(first_window_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {first_hits_path}")
    print(f"Saved: {first_window_path}")
    print(f"Saved: {os.path.join(args.out_dir, 'first_day_dump')}/*")


if __name__ == "__main__":
    main()
