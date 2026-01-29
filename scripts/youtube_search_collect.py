import os
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import yaml
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def load_keywords(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj.get("keywords", [])


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def first_match(text: str, keywords: List[str]) -> Optional[str]:
    if not text:
        return None
    for k in keywords:
        if not k:
            continue
        # 완성된 키워드만 매칭
        if k in text:
            return k
    return None



def search_video_ids(youtube, query: str, max_videos: int, region_code: str = "KR") -> List[str]:
    ids = []
    req = youtube.search().list(
        part="id",
        q=query,
        type="video",
        maxResults=50,
        order="date",
        regionCode=region_code,
    )
    while req and len(ids) < max_videos:
        res = req.execute()
        for it in res.get("items", []):
            vid = it["id"].get("videoId")
            if vid:
                ids.append(vid)
                if len(ids) >= max_videos:
                    break
        req = youtube.search().list_next(req, res)
    return ids


def get_video_meta(youtube, video_id: str) -> Dict[str, str]:
    res = youtube.videos().list(part="snippet", id=video_id, maxResults=1).execute()
    if not res.get("items"):
        return {"video_title": "", "channel_title": "", "publishedAt": ""}
    s = res["items"][0]["snippet"]
    return {
        "video_title": s.get("title", ""),
        "channel_title": s.get("channelTitle", ""),
        "publishedAt": (s.get("publishedAt", "") or "").replace("Z", "")
    }


def fetch_keyword_comments(youtube, video_id: str, keywords: List[str], max_comments: int, sleep_sec: float) -> List[Dict]:
    rows = []
    req = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText",
        order="time",
    )
    while req and len(rows) < max_comments:
        res = req.execute()
        for item in res.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            text = s.get("textDisplay", "")
            kw = first_match(text, keywords)
            if kw:
                rows.append({
                    "date": (s.get("publishedAt", "") or "").replace("Z", ""),
                    "platform": "youtube",
                    "source": "",  # 채널명
                    "keyword": kw,
                    "video_id": video_id,
                    "video_title": "",
                    "query": "",   # 어떤 키워드로 찾았는지
                    "text": text,
                })
            if len(rows) >= max_comments:
                break
        req = youtube.commentThreads().list_next(req, res)
        time.sleep(sleep_sec)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", default="config/keywords.yaml")
    parser.add_argument("--max_videos_per_keyword", type=int, default=30)
    parser.add_argument("--max_comments_per_video", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--region", default="KR")
    parser.add_argument("--out_dir", default="data/raw/youtube_search")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY가 없습니다. .env에 넣어주세요.")

    keywords = load_keywords(args.keywords)
    if not keywords:
        raise RuntimeError("keywords.yaml이 비어 있습니다.")

    youtube = build("youtube", "v3", developerKey=api_key)

    run_day = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(args.out_dir, run_day)
    ensure_dir(out_dir)

    all_rows = []

    for kw in keywords:
        try:
            video_ids = search_video_ids(youtube, query=kw, max_videos=args.max_videos_per_keyword, region_code=args.region)
            for vid in tqdm(video_ids, desc=f"Search:{kw}", leave=False):
                meta = get_video_meta(youtube, vid)
                rows = fetch_keyword_comments(youtube, vid, keywords, args.max_comments_per_video, args.sleep)
                for r in rows:
                    r["source"] = meta["channel_title"]
                    r["video_title"] = meta["video_title"]
                    r["query"] = kw
                all_rows.extend(rows)
        except HttpError as e:
            print(f"[HTTPError] keyword={kw}: {e}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values("date")

    out_csv = os.path.join(out_dir, f"youtube_search_keyword_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv} (rows={len(df)})")


if __name__ == "__main__":
    main()
