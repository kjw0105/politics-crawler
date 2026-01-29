import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="merged CSV path")
    p.add_argument("--weekly_threshold", type=int, default=5, help="주간 N회 이상이면 의미있는 등장")
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--report_dir", default="reports")
    args = p.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")

    # 날짜 파싱
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # keyword 정리
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df = df[df["keyword"] != ""]

    # ✅ '주'를 문자열이 아니라 Period로 유지 -> 정렬/비교 안전
    df["week"] = df["date"].dt.to_period("W")  # Period[W-SUN] 등

    # 1) 최초 관측일 (first observed)
    first_any = df.groupby("keyword")["date"].min().sort_values()

    # 2) 주간 빈도
    weekly = (
        df.groupby(["keyword", "week"])
          .size()
          .reset_index(name="n")
          .sort_values(["keyword", "week"])
    )

    # 3) 의미 있는 첫 주(주간 threshold 이상 최초)
    def first_week_over(g: pd.DataFrame):
        g2 = g[g["n"] >= args.weekly_threshold]
        return g2.iloc[0]["week"] if len(g2) else pd.NaT

    first_meaningful = weekly.groupby("keyword").apply(first_week_over)

    # 4) 주간 최대치(확산 강도 참고지표)
    peak_week = weekly.loc[weekly.groupby("keyword")["n"].idxmax()].set_index("keyword")
    peak_week = peak_week.rename(columns={"week": "peak_week", "n": "peak_n"})

    # ----- 저장 -----
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    summary = pd.DataFrame({
        "keyword": first_any.index,
        "first_observed_date": first_any.values,
        f"first_week_ge_{args.weekly_threshold}": first_meaningful.reindex(first_any.index).values,
    }).set_index("keyword")

    summary = summary.join(peak_week[["peak_week", "peak_n"]], how="left").reset_index()

    out_summary = os.path.join(args.out_dir, f"keyword_appearance_summary_week{args.weekly_threshold}.csv")
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

    out_weekly = os.path.join(args.out_dir, "weekly_counts.csv")
    weekly.assign(week=weekly["week"].astype(str)).to_csv(out_weekly, index=False, encoding="utf-8-sig")

    print(f"Saved summary -> {out_summary}")
    print(f"Saved weekly counts -> {out_weekly}")

    # ----- 그래프(키워드별 주간 추세) -----
    # wide 형태로 피벗: index=week, columns=keyword, values=n
    pivot = weekly.pivot(index="week", columns="keyword", values="n").fillna(0).sort_index()
    pivot.index = pivot.index.to_timestamp()  # matplotlib용

    plt.figure(figsize=(12, 6))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], label=col)

    plt.xlabel("Week")
    plt.ylabel("Count (comments with keyword)")
    plt.title("Weekly keyword counts (YouTube comments)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()

    out_png = os.path.join(args.report_dir, "weekly_trends.png")
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"Saved plot -> {out_png}")


if __name__ == "__main__":
    main()
