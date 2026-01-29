import argparse
import pandas as pd

def is_naver_news(url: str) -> bool:
    if not isinstance(url, str):
        return False
    return ("news.naver.com" in url) or ("n.news.naver.com" in url)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", required=True)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)

    mask = df["link"].apply(is_naver_news) | df["originallink"].apply(is_naver_news)
    out = df.loc[mask, ["title", "link", "originallink", "pubDate"]].copy()

    # 네이버 뉴스 링크 우선 채우기 (link가 네이버뉴스면 link, 아니면 originallink)
    out["naver_url"] = out["link"]
    out.loc[~out["naver_url"].apply(is_naver_news), "naver_url"] = out["originallink"]

    out = out.drop_duplicates(subset=["naver_url"]).reset_index(drop=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.out_csv} (rows={len(out)})")

if __name__ == "__main__":
    main()
