import os
import time
import argparse
from datetime import datetime
from typing import List, Optional, Dict

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


def first_match(text: str, keywords: List[str]) -> Optional[str]:
    for k in keywords:
        if k and k in text:
            return k
    return None


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def get_channel_meta(youtube, channel_id: str) -> Dict[str, str]:
    # handle/url 지원
    handle = None
    if "youtube.com/@" in channel_id:
        handle = "@" + channel_id.split("youtube.com/@", 1)[1].split("/", 1)[0]
    elif channel_id.startswith("@") or not channel_id.startswith("UC"):
        handle = channel_id if channel_id.startswith("@") else "@" + channel_id

    if handle:
        res = youtube.channels().list(part="contentDetails,snippet", forHandle=handle, maxResults=1).execute()
    else:
        res = youtube.channels().list(part="contentDetails,snippet", id=channel_id, maxResults=1).execute()

    if not res.get("items"):
        raise ValueError(f"채널 ID/핸들을 찾을 수 없음: {channel_id}")

    item = res["items"][0]
    uploads = item["contentDetails"]["relatedPlaylists"]["uploads"]
    title = item["snippet"]["title"]
    return {"uploads_playlist_id": uploads, "channel_title": title}


def list_video_ids(youtube, uploads_playlist_id: str, max_videos: int) -> List[str]:
    ids = []
    req = youtube.playlistItems().list(
        part="contentDetails",
        playlistId=uploads_playlist_id,
        maxResults=50
    )
    while req and len(ids) < max_videos:
        res = req.execute()
        for it in res.get("items", []):
            ids.append(it["contentDetails"]["videoId"])
            if len(ids) >= max_videos:
                break
        req = youtube.playlistItems().list_next(req, res)
    return ids


def get_video_title(youtube, video_id: str) -> str:
    res = youtube.videos().list(part="snippet", id=video_id, maxResults=1).execute()
    if not res.get("items"):
        return ""
    return res["items"][0]["snippet"]["title"]


def fetch_keyword_comments(
    youtube,
    video_id: str,
    keywords: List[str],
    channel_title: str,
    video_title: str,
    max_comments: int,
    sleep_sec: float,
) -> List[Dict]:
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
                    "source": channel_title,
                    "keyword": kw,
                    "video_id": video_id,
                    "video_title": video_title,
                    "text": text,
                })
            if len(rows) >= max_comments:
                break

        req = youtube.commentThreads().list_next(req, res)
        time.sleep(sleep_sec)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel_ids", nargs="+", required=True, help="UC로 시작하는 채널 ID들")
    parser.add_argument("--keywords", default="config/keywords.yaml", help="키워드 yaml 경로")
    parser.add_argument("--max_videos", type=int, default=30, help="채널당 최신 영상 개수")
    parser.add_argument("--max_comments", type=int, default=500, help="영상당 최대 수집 댓글(키워드 포함만 저장)")
    parser.add_argument("--sleep", type=float, default=0.1, help="API 호출 간 sleep(초)")
    parser.add_argument("--out_dir", default="data/raw/youtube", help="원본 저장 폴더")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY가 없습니다. .env 파일에 넣어주세요.")

    keywords = load_keywords(args.keywords)
    if not keywords:
        raise RuntimeError("keywords.yaml에서 키워드를 읽지 못했습니다.")

    youtube = build("youtube", "v3", developerKey=api_key)

    run_day = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(args.out_dir, run_day)
    ensure_dir(out_dir)

    all_rows = []

    for channel_id in args.channel_ids:
        meta = get_channel_meta(youtube, channel_id)
        uploads = meta["uploads_playlist_id"]
        channel_title = meta["channel_title"]

        video_ids = list_video_ids(youtube, uploads, args.max_videos)

        for vid in tqdm(video_ids, desc=f"{channel_title}", leave=False):
            try:
                vtitle = get_video_title(youtube, vid)
                rows = fetch_keyword_comments(
                    youtube=youtube,
                    video_id=vid,
                    keywords=keywords,
                    channel_title=channel_title,
                    video_title=vtitle,
                    max_comments=args.max_comments,
                    sleep_sec=args.sleep,
                )
                all_rows.extend(rows)
            except HttpError as e:
                print(f"[HTTPError] {channel_title} / {vid}: {e}")
                continue

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values("date")

    out_csv = os.path.join(out_dir, f"youtube_keyword_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv} (rows={len(df)})")


if __name__ == "__main__":
    main()
