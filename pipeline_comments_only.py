# pipeline_comments_only.py
# ------------------------------------------------------------
# 목적:
# - candidates.csv의 video_id 전부 순회
# - 각 영상의 댓글/대댓글을 최대한 스캔
# - 키워드가 포함된 댓글만 matched_comments.csv에 "누적 append" 저장
# - 중간에 꺼져도 이어서 가능(resume): 이미 저장된 comment_id는 중복 저장 안 함
#
# 실행 예:
#   python pipeline_comments_only.py --api_key "YOUR_KEY" --candidates candidates.csv `
#     --keyword "여경" --keyword "퐁퐁남" --keyword "꼴페미" `
#     --out matched_comments.csv --progress progress_videos.csv
# ------------------------------------------------------------

import os
import re
import time
import random
import argparse
from typing import List, Dict, Optional, Set

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


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


def rand_sleep(base: float, jitter: float = 0.25):
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
        seen = set(old["comment_id"].dropna().astype(str).tolist())
        print(f"[resume] loaded seen comment_ids={len(seen)} from {out_csv}")
        return seen
    except Exception as e:
        print(f"[resume] failed read {out_csv}: {e}")
        return set()


def append_rows(out_csv: str, rows: List[Dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows).drop_duplicates(subset=["comment_id"])
    file_exists = os.path.exists(out_csv)
    df.to_csv(out_csv, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")
    return len(df)


def load_done_videos(progress_csv: str) -> Set[str]:
    if not os.path.exists(progress_csv):
        return set()
    try:
        df = pd.read_csv(progress_csv, usecols=["video_id", "status"])
        done = set(df[df["status"] == "done"]["video_id"].astype(str).tolist())
        print(f"[resume] loaded done videos={len(done)} from {progress_csv}")
        return done
    except Exception:
        return set()


def append_progress(progress_csv: str, row: Dict) -> None:
    df = pd.DataFrame([row])
    file_exists = os.path.exists(progress_csv)
    df.to_csv(progress_csv, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")


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
    reply_parents = 0

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
                "like_count": top_snip.get("likeCount", 0),
                "text": top_text.replace("\n", " ").strip(),
            })
            matched_new += 1

        total_reply = int(th["snippet"].get("totalReplyCount", 0) or 0)
        if total_reply > 0 and top_id:
            reply_parents += 1
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
                        "like_count": rs.get("likeCount", 0),
                        "text": rtext.replace("\n", " ").strip(),
                    })
                    matched_new += 1

        if flush_every and len(rows) >= flush_every:
            appended = append_rows(out_csv, rows)
            rows.clear()
            print(f"[flush] video={video_id} scanned_threads={scanned_threads} new_matches={matched_new} appended={appended}")

    appended_last = append_rows(out_csv, rows)
    rows.clear()
    return {
        "video_id": video_id,
        "scanned_threads": scanned_threads,
        "reply_parents": reply_parents,
        "new_matches": matched_new,
        "appended_last": appended_last,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--candidates", default="candidates.csv")
    ap.add_argument("--keyword", action="append", default=[], help="repeatable")
    ap.add_argument("--out", default="matched_comments.csv")
    ap.add_argument("--progress", default="progress_videos.csv")
    ap.add_argument("--order", default="time", choices=["time", "relevance"])
    ap.add_argument("--sleep_comments", type=float, default=0.02)
    ap.add_argument("--sleep_between_videos", type=float, default=0.10)
    ap.add_argument("--flush_every", type=int, default=300)
    ap.add_argument("--max_videos", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip videos already done in progress csv")
    args = ap.parse_args()

    keywords = [k.strip() for k in args.keyword if k and k.strip()]
    if not keywords:
        raise SystemExit("ERROR: provide at least one --keyword")

    df = pd.read_csv(args.candidates)
    if "video_id" not in df.columns:
        raise SystemExit("ERROR: candidates.csv must have video_id column")

    video_ids = [extract_video_id(v) for v in df["video_id"].astype(str).tolist()]
    video_ids = list(dict.fromkeys([v for v in video_ids if v]))

    if args.max_videos and args.max_videos > 0:
        video_ids = video_ids[: args.max_videos]

    done_videos = load_done_videos(args.progress) if args.resume else set()
    seen_comment_ids = load_seen_comment_ids(args.out)

    youtube = build("youtube", "v3", developerKey=args.api_key)

    print(f"[start] videos={len(video_ids)} keywords={keywords} out={args.out} progress={args.progress} resume={args.resume}")

    total_new = 0
    for idx, vid in enumerate(video_ids, 1):
        if args.resume and vid in done_videos:
            print(f"[skip] {idx}/{len(video_ids)} video={vid} already done")
            continue

        print(f"\n[video] {idx}/{len(video_ids)} {vid}")
        t0 = time.time()

        status = "done"
        scanned_threads = 0
        new_matches = 0
        reply_parents = 0

        try:
            stats = fetch_comments_for_video(
                youtube=youtube,
                video_id=vid,
                keywords=keywords,
                out_csv=args.out,
                seen_ids=seen_comment_ids,
                order=args.order,
                sleep_sec=args.sleep_comments,
                flush_every=args.flush_every,
            )
            scanned_threads = stats["scanned_threads"]
            new_matches = stats["new_matches"]
            reply_parents = stats["reply_parents"]
            total_new += new_matches
            print(f"[done] video={vid} threads={scanned_threads} reply_parents={reply_parents} new_matches={new_matches}")
        except HttpError as e:
            status = "http_error"
            code = getattr(e.resp, "status", None)
            print(f"[warn] video={vid} http_error status={code}: {e}")
        except Exception as e:
            status = "error"
            print(f"[warn] video={vid} error: {e}")

        elapsed = round(time.time() - t0, 2)
        append_progress(args.progress, {
            "video_id": vid,
            "status": status,
            "scanned_threads": scanned_threads,
            "reply_parents": reply_parents,
            "new_matches": new_matches,
            "elapsed_sec": elapsed,
            "ts": pd.Timestamp.utcnow().isoformat(),
        })

        rand_sleep(args.sleep_between_videos, jitter=0.25)

    print(f"\n[all done] total_new_matches={total_new} -> {args.out}")


if __name__ == "__main__":
    main()
