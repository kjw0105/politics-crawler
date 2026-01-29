# pipeline_cc_then_comments.py
# ------------------------------------------------------------
# 1) candidates.csv에서 video_id 목록 읽기
# 2) (공개 자막이 있으면) 자막을 가져와 키워드 매칭
# 3) 자막 매칭된 video_id만 YouTube Data API로 댓글/대댓글 스캔
# 4) 키워드 포함 댓글을 matched_comments.csv에 "누적 append" 저장 (중복 방지)
#
# 설치:
#   pip install google-api-python-client pandas python-dateutil youtube-transcript-api
#
# 실행(예):
#   python pipeline_cc_then_comments.py --api_key "YOUR_KEY" --candidates candidates.csv `
#     --keyword "여경" --keyword "퐁퐁남" --keyword "꼴페미" `
#     --matched_videos matched_videos.csv --comments_out matched_comments.csv
# ------------------------------------------------------------

import os
import re
import time
import random
import argparse
from typing import List, Dict, Optional, Set, Tuple

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)


# ---------------------------
# Utils
# ---------------------------
def extract_video_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", s)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", s)
    if m:
        return m.group(1)
    m = re.search(r"embed/([A-Za-z0-9_-]{6,})", s)
    if m:
        return m.group(1)
    return s


def contains_keyword(text: str, keywords: List[str], case_insensitive: bool = True) -> Optional[str]:
    if not text:
        return None
    hay = text.lower() if case_insensitive else text
    for kw in keywords:
        needle = kw.lower() if case_insensitive else kw
        if needle in hay:
            return kw
    return None


def contains_any_keyword(text: str, keywords: List[str]) -> List[str]:
    """자막은 여러 키워드가 동시에 잡힐 수 있으니 리스트로 반환"""
    if not text:
        return []
    t = text.lower()
    hits = []
    for kw in keywords:
        if kw.lower() in t:
            hits.append(kw)
    return hits


def rand_sleep(base: float, jitter: float = 0.3):
    time.sleep(base + random.uniform(0, jitter))


def _retryable_http(e: HttpError) -> bool:
    status = getattr(e.resp, "status", None)
    return status in (403, 429, 500, 502, 503, 504)


def youtube_api_call(func, max_retries: int = 7):
    for attempt in range(max_retries):
        try:
            return func()
        except HttpError as e:
            if attempt == max_retries - 1 or not _retryable_http(e):
                raise
            time.sleep(min(30, (2 ** attempt)) + random.uniform(0, 0.5))


# ---------------------------
# Transcript step (CC)
# ---------------------------
def get_transcript_text(video_id: str, languages: List[str]) -> Optional[str]:
    """
    공개 자막(자동 생성 포함)이 있으면 가져오고, 없으면 None.
    youtube-transcript-api는 공식 API 키를 안 쓰는 경우가 많지만,
    영상/국가/정책에 따라 실패할 수 있으니 예외는 정상으로 처리.
    """
    try:
        items = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        # items: [{"text":..., "start":..., "duration":...}, ...]
        text = " ".join([it.get("text", "") for it in items])
        return text
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript):
        return None
    except Exception:
        return None


# ---------------------------
# Comment fetching step
# ---------------------------
def iter_comment_threads_all(youtube, video_id: str, order: str = "time", sleep_sec: float = 0.02):
    token = None
    while True:
        def _call():
            return youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=token,
                textFormat="plainText",
                order=order,
            ).execute()

        resp = youtube_api_call(_call)
        for it in resp.get("items", []):
            yield it

        token = resp.get("nextPageToken")
        if sleep_sec:
            time.sleep(sleep_sec)
        if not token:
            break


def iter_replies_all(youtube, parent_comment_id: str, sleep_sec: float = 0.02):
    token = None
    while True:
        def _call():
            return youtube.comments().list(
                part="snippet",
                parentId=parent_comment_id,
                maxResults=100,
                pageToken=token,
                textFormat="plainText",
            ).execute()

        resp = youtube_api_call(_call)
        for it in resp.get("items", []):
            yield it

        token = resp.get("nextPageToken")
        if sleep_sec:
            time.sleep(sleep_sec)
        if not token:
            break


def load_seen_comment_ids(out_csv: str) -> Set[str]:
    if not os.path.exists(out_csv):
        return set()
    try:
        old = pd.read_csv(out_csv, usecols=["comment_id"])
        return set(old["comment_id"].dropna().astype(str).tolist())
    except Exception:
        return set()


def append_rows(out_csv: str, rows: List[Dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows).drop_duplicates(subset=["comment_id"])
    file_exists = os.path.exists(out_csv)
    df.to_csv(out_csv, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")
    return len(df)


def fetch_comments_for_video(
    youtube,
    video_id: str,
    keywords: List[str],
    out_csv: str,
    seen_ids: Set[str],
    order: str = "time",
    sleep_sec: float = 0.02,
    flush_every: int = 300,
) -> Dict[str, int]:
    rows: List[Dict] = []
    scanned_threads = 0
    matched_new = 0

    try:
        for th in iter_comment_threads_all(youtube, video_id, order=order, sleep_sec=sleep_sec):
            scanned_threads += 1

            top = th["snippet"]["topLevelComment"]
            top_id = str(top.get("id", "") or "")
            top_snip = top.get("snippet", {})
            top_text = top_snip.get("textDisplay", "") or ""

            mk = contains_keyword(top_text, keywords)
            if mk and top_id and top_id not in seen_ids:
                seen_ids.add(top_id)
                rows.append({
                    "video_id": video_id,
                    "matched_keyword": mk,
                    "comment_level": "top",
                    "comment_id": top_id,
                    "parent_id": "",
                    "author_name": top_snip.get("authorDisplayName", ""),
                    "published_at": top_snip.get("publishedAt", ""),
                    "text": top_text.replace("\n", " ").strip(),
                })
                matched_new += 1

            total_reply = int(th["snippet"].get("totalReplyCount", 0) or 0)
            if total_reply > 0 and top_id:
                for r in iter_replies_all(youtube, top_id, sleep_sec=sleep_sec):
                    rid = str(r.get("id", "") or "")
                    rs = r.get("snippet", {})
                    rtext = rs.get("textDisplay", "") or ""
                    mk2 = contains_keyword(rtext, keywords)
                    if mk2 and rid and rid not in seen_ids:
                        seen_ids.add(rid)
                        rows.append({
                            "video_id": video_id,
                            "matched_keyword": mk2,
                            "comment_level": "reply",
                            "comment_id": rid,
                            "parent_id": top_id,
                            "author_name": rs.get("authorDisplayName", ""),
                            "published_at": rs.get("publishedAt", ""),
                            "text": rtext.replace("\n", " ").strip(),
                        })
                        matched_new += 1

            if flush_every and len(rows) >= flush_every:
                appended = append_rows(out_csv, rows)
                rows.clear()
                print(f"[flush] video={video_id} scanned_threads={scanned_threads} new_matches={matched_new} appended={appended}")

    except HttpError as e:
        # 댓글 비활성화/권한 문제 등
        status = getattr(e.resp, "status", None)
        print(f"[warn] video={video_id} comments fetch failed (status={status}): {e}")
    except Exception as e:
        print(f"[warn] video={video_id} comments fetch failed: {e}")

    appended_last = append_rows(out_csv, rows)
    rows.clear()
    print(f"[done] video={video_id} scanned_threads={scanned_threads} new_matches={matched_new} appended_last={appended_last}")
    return {"video_id": video_id, "scanned_threads": scanned_threads, "new_matches": matched_new}


# ---------------------------
# Main pipeline
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--candidates", default="candidates.csv")
    ap.add_argument("--keyword", action="append", default=[], help="repeatable")
    ap.add_argument("--languages", default="ko,en", help="comma-separated transcript language preference")
    ap.add_argument("--matched_videos", default="matched_videos.csv")
    ap.add_argument("--comments_out", default="matched_comments.csv")
    ap.add_argument("--max_videos", type=int, default=0, help="0 means no limit")
    ap.add_argument("--sleep_transcript", type=float, default=0.25)
    ap.add_argument("--sleep_comments", type=float, default=0.02)
    ap.add_argument("--flush_every", type=int, default=300)
    args = ap.parse_args()

    keywords = [k.strip() for k in args.keyword if k and k.strip()]
    if not keywords:
        raise SystemExit("ERROR: provide at least one --keyword")

    langs = [x.strip() for x in args.languages.split(",") if x.strip()]

    df = pd.read_csv(args.candidates)
    if "video_id" not in df.columns:
        raise SystemExit("ERROR: candidates.csv must have video_id column")

    video_ids = [extract_video_id(v) for v in df["video_id"].astype(str).tolist()]
    # 중복 제거, 입력 순서 유지
    video_ids = list(dict.fromkeys([v for v in video_ids if v]))

    if args.max_videos and args.max_videos > 0:
        video_ids = video_ids[: args.max_videos]

    print(f"[stage1] transcript scan: videos={len(video_ids)} keywords={keywords} languages={langs}")

    matched_rows = []
    for i, vid in enumerate(video_ids, 1):
        txt = get_transcript_text(vid, langs)
        if txt:
            hits = contains_any_keyword(txt, keywords)
            if hits:
                matched_rows.append({
                    "video_id": vid,
                    "matched_keywords": ";".join(hits),
                    "has_transcript": True,
                })
                print(f"[hit] {i}/{len(video_ids)} video={vid} keywords={hits}")
            else:
                print(f"[nohit] {i}/{len(video_ids)} video={vid} transcript=Y")
        else:
            print(f"[skip] {i}/{len(video_ids)} video={vid} transcript=N")

        rand_sleep(args.sleep_transcript, jitter=0.35)

    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(args.matched_videos, index=False, encoding="utf-8-sig")
    print(f"[stage1 done] matched_videos={len(matched_df)} -> {args.matched_videos}")

    if matched_df.empty:
        print("[stage2] no matched videos -> stop")
        return

    # stage2: comments fetch only for matched videos
    youtube = build("youtube", "v3", developerKey=args.api_key)

    seen_ids = load_seen_comment_ids(args.comments_out)
    print(f"[stage2] comments scan: target_videos={len(matched_df)} seen_comment_ids={len(seen_ids)} out={args.comments_out}")

    total_new = 0
    for j, vid in enumerate(matched_df["video_id"].tolist(), 1):
        print(f"\n[video {j}/{len(matched_df)}] {vid}")
        stats = fetch_comments_for_video(
            youtube=youtube,
            video_id=vid,
            keywords=keywords,
            out_csv=args.comments_out,
            seen_ids=seen_ids,
            order="time",
            sleep_sec=args.sleep_comments,
            flush_every=args.flush_every,
        )
        total_new += stats["new_matches"]
        rand_sleep(0.05, jitter=0.2)

    print(f"\n[all reminds] matched_videos={len(matched_df)} total_new_comments={total_new} -> {args.comments_out}")


if __name__ == "__main__":
    main()
