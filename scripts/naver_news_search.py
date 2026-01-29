import os, argparse, time
from datetime import datetime
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

API_URL = "https://openapi.naver.com/v1/search/news.json"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help='예: 설거지론')
    p.add_argument("--display", type=int, default=100)   # 최대 100
    p.add_argument("--pages", type=int, default=10)      # pages*display 만큼 수집
    p.add_argument("--sort", default="date", choices=["date", "sim"])
    p.add_argument("--out_dir", default="data/raw/naver_news")
    p.add_argument("--sleep", type=float, default=0.2)
    args = p.parse_args()

    load_dotenv()
    cid = os.getenv("NAVER_CLIENT_ID")
    csec = os.getenv("NAVER_CLIENT_SECRET")
    if not cid or not csec:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 가 .env에 없습니다.")

    headers = {
        "X-Naver-Client-Id": cid,
        "X-Naver-Client-Secret": csec
    }

    rows = []
    start = 1

    for _ in tqdm(range(args.pages), desc="Fetching pages"):
        params = {
            "query": args.query,
            "display": args.display,
            "start": start,
            "sort": args.sort
        }
        r = requests.get(API_URL, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        items = data.get("items", [])
        if not items:
            break

        for it in items:
            rows.append({
                "query": args.query,
                "title": it.get("title", ""),
                "description": it.get("description", ""),
                "originallink": it.get("originallink", ""),
                "link": it.get("link", ""),        # 네이버 뉴스 링크일 때가 많음
                "pubDate": it.get("pubDate", ""),
            })

        start += args.display
        time.sleep(args.sleep)

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(
        args.out_dir,
        f"naver_news_{args.query}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv} (rows={len(rows)})")

if __name__ == "__main__":
    main()
