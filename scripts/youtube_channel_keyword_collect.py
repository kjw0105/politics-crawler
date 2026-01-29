import os
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def today_dir(base: str) -> str:
    d = datetime.now().strftime("%Y-%m-%d")
    out = os.path.join(base, d)
    os.makedirs(out, exist_ok=True)
    return out


def require_channel_id(s: str) -> str:
    """
    이 스크립트는 쿼터 절약을 위해 search.list를 쓰지 않음.
    따라서 채널 입력은 반드시 UC... 형태의 채널ID여야 함.
    """
    s = s.strip()
    if not s.startswith("UC"):
        raise ValueError(
            "채널은 반드시 'UC...'로 시작하는 채널 ID를 넣어야 합니다.\n"
            "예: UCJb-2z4wH5dVJQ5q6kz8p6A"
        )
    return s


def get_uploads_playlist_id(youtube, channel_id: str) -> str:
    res = youtube.channels().list(
        part="contentDetails,snippet",
        id=channel_id
    ).execute()
    items = res.get("items", [])
    if not items:
        raise ValueError(f"channels.list 결과 0: {channel_id}")
    uploads = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    return uploads


def iter_channel_videos(youtube, uploads_playlist_id: str, max_videos: int) -> List[Dict[str, Any]]:
    """
    업로드 재생목록에서 videoId + publishedAt + title 수집
    """
    out = []
    req = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=uploads_playlist_id,
        maxResults=50
    )
    while req and len(out) < max_videos:
        res = req.execute()
        for it in res.get("items", []):
            cd = it.get("contentDetails", {})
            sn = it.get("snippet", {})
            vid = cd.get("videoId")
            if not vid:
                continue
            out.append({
                "videoId": vid,
                "videoTitle": sn.get("title", ""),
                "videoPublishedAt": sn.get("publishedAt", ""),
            })
            if len(out) >= max_videos:
                break
        req = youtube.playlistItems().list_next(req, res)
    return out


def keyword_hit(text: str, keywords: List[str]) -> str | None:
    for k in keywords:
        if k and (k in text):
            return k
    return None


def collect_video_comments_keyword_only(
    youtube,
    video_id: str,
    keywords: List[str],
    max_pages: int,
    sleep_sec: float
) -> List[Dict[str, Any]]:
    """
    댓글 최신순(order=time)으로 최대 max_pages만 탐색하며,
    키워드 포함 댓글만 반환.
    """
    rows: List[Dict[str, Any]] = []
    try:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            order="time",
        )

        pages = 0
        while req and pages < max_pages:
            res = req.execute()
            for item in res.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                text = top.get("textDisplay", "") or ""
                hit = keyword_hit(text, keywords)
                if hit:
                    rows.append({
                        "matched_keyword": hit,
                        "commentId": item["snippet"]["topLevelComment"]["id"],
                        "commentPublishedAt": top.get("publishedAt", ""),
                        "authorDisplayName": top.get("authorDisplayName", ""),
                        "likeCount": top.get("likeCount", None),
                        "text": text,
                    })
            req = youtube.commentThreads().list_next(req, res)
            pages += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    except HttpError:
        # commentsDisabled, removed, etc. 는 그냥 스킵
        return rows

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channel_id", required=True, help="반드시 UC... 채널ID (쿼터절약: search.list 안씀)")
    p.add_argument("--keywords", nargs="+", required=True, help='예: "설거지론" "이대남"')
    p.add_argument("--max_videos", type=int, default=300, help="채널 업로드 영상에서 가져올 최대 개수")
    p.add_argument("--max_comment_pages", type=int, default=20, help="영상당 댓글 페이지(100개=1페이지)")
    p.add_argument("--sleep", type=float, default=0.3)
    p.add_argument("--out_dir", default="data/raw/youtube_channel_keyword")
    args = p.parse_args()

    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY가 없습니다. .env 확인")

    channel_id = require_channel_id(args.channel_id)

    youtube = build("youtube", "v3", developerKey=api_key)

    uploads_id = get_uploads_playlist_id(youtube, channel_id)
    videos = iter_channel_videos(youtube, uploads_id, max_videos=args.max_videos)
    print(f"Channel={channel_id}, uploads_playlist={uploads_id}, videos_loaded={len(videos)}")

    out_base = today_dir(args.out_dir)
    out_csv = os.path.join(out_base, f"channel_keyword_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    all_rows: List[Dict[str, Any]] = []
    for i, v in enumerate(videos, 1):
        vid = v["videoId"]
        hits = collect_video_comments_keyword_only(
            youtube=youtube,
            video_id=vid,
            keywords=args.keywords,
            max_pages=args.max_comment_pages,
            sleep_sec=args.sleep,
        )

        if hits:
            for h in hits:
                all_rows.append({
                    "channelId": channel_id,
                    "videoId": vid,
                    "videoTitle": v["videoTitle"],
                    "videoPublishedAt": v["videoPublishedAt"],
                    **h
                })
            print(f"[{i}/{len(videos)}] HIT video={vid} hits={len(hits)} title={v['videoTitle'][:40]}")
        else:
            print(f"[{i}/{len(videos)}] no-hit video={vid}")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv} (rows={len(df)})")


if __name__ == "__main__":
    main()
