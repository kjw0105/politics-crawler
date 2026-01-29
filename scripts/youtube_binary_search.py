import os
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Union

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def iso_z(dt: datetime) -> str:
    # YouTube API는 RFC3339 UTC(Z)를 선호
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def first_match(text: str, keywords: List[str]) -> Optional[str]:
    for k in keywords:
        if k and (k in text):
            return k
    return None


def search_video_ids_before(
    youtube,
    query: str,
    published_before_iso: str,
    max_videos: int,
    region: str
) -> List[str]:
    """
    cutoff 이전에 업로드된 영상들을 (최신부터) 검색으로 모음.
    주의: YouTube search 결과는 완전한 '전체'가 아니라 API가 반환하는 '검색 가능 결과'임.
    """
    ids = []
    req = youtube.search().list(
        part="id",
        q=query,
        type="video",
        maxResults=50,
        order="date",              # publishedBefore 이전의 최신 업로드부터
        regionCode=region,
        publishedBefore=published_before_iso,
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


def has_keyword_comment(
    youtube,
    video_id: str,
    keywords: List[str],
    max_pages: int,
    sleep_sec: float
) -> bool:
    """
    댓글을 오래된순으로 정렬은 불가.
    order="time"은 '최신 댓글부터'에 가까움.
    여기서는 "존재 여부(True/False)"만 확인.
    """
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
                s = item["snippet"]["topLevelComment"]["snippet"]
                text = s.get("textDisplay", "") or ""
                if first_match(text, keywords):
                    return True
            req = youtube.commentThreads().list_next(req, res)
            pages += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    except HttpError:
        # commentsDisabled, removed, etc.
        return False
    return False


def exists_before_cutoff(
    youtube,
    keyword: str,
    cutoff: datetime,
    keywords_all: List[str],
    max_videos: int,
    region: str,
    max_comment_pages: int,
    sleep_sec: float,
) -> bool:
    """
    cutoff(시점) 이전에 업로드된 영상들 중,
    키워드 포함 댓글이 '존재'하는지(하나라도 True면 True).
    """
    try:
        vids = search_video_ids_before(
            youtube=youtube,
            query=keyword,
            published_before_iso=iso_z(cutoff),
            max_videos=max_videos,
            region=region,
        )
    except HttpError:
        return False

    for vid in vids:
        if has_keyword_comment(
            youtube=youtube,
            video_id=vid,
            keywords=keywords_all,
            max_pages=max_comment_pages,
            sleep_sec=sleep_sec,
        ):
            return True
    return False


def binary_search_first_presence(
    youtube,
    keyword: str,
    start: datetime,
    end: datetime,
    keywords_all: List[str],
    max_videos: int,
    region: str,
    max_comment_pages: int,
    sleep_sec: float,
    max_iters: int = 18,
) -> Tuple[datetime, Optional[datetime], List[Tuple[str, bool]]]:
    """
    exists_before_cutoff를 이용해 [start, end]에서
    "처음 존재가 나타나는 구간"을 이진 탐색으로 좁힘.

    반환:
      low: low 이전에는 (대체로) 없음(no-presence)
      high: high 이전에는 있음(presence)으로 관찰되는 경계 (단, 관찰 실패 시 None)
      trace: (mid_date_iso, exists) 로그

    중요:
      탐색 중 True가 한 번도 안 나오면,
      "이 범위/조건에서는 관찰 실패"이므로 high=None 처리.
    """
    low = start
    high = end
    trace: List[Tuple[str, bool]] = []
    seen_true = False

    for _ in range(max_iters):
        if (high - low) <= timedelta(days=2):
            break

        mid = low + (high - low) / 2

        exists = exists_before_cutoff(
            youtube=youtube,
            keyword=keyword,
            cutoff=mid,
            keywords_all=keywords_all,
            max_videos=max_videos,
            region=region,
            max_comment_pages=max_comment_pages,
            sleep_sec=sleep_sec,
        )
        trace.append((mid.date().isoformat(), exists))

        if exists:
            seen_true = True
            high = mid   # 더 과거로 좁힘
        else:
            low = mid    # 더 최근으로 좁힘

    if not seen_true:
        # 범위 전체에서 True가 한 번도 안 나온 경우:
        # "등장"을 찾은 게 아니라 "관찰 실패"
        return low, None, trace

    return low, high, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keywords", nargs="+", required=True, help='예: "설거지론" "이대남" "드럼통" "김치녀"')
    p.add_argument("--start", default="2008-01-01", help="탐색 시작(YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="탐색 종료(YYYY-MM-DD), 기본은 오늘")
    p.add_argument("--region", default="KR")
    p.add_argument("--max_videos", type=int, default=120, help="시점마다 키워드 검색으로 가져올 영상 수")
    p.add_argument("--max_comment_pages", type=int, default=15, help="영상당 댓글 페이지 탐색(100개=1페이지)")
    p.add_argument("--sleep", type=float, default=0.1)
    p.add_argument("--iters", type=int, default=16, help="이진 탐색 반복 횟수")
    p.add_argument("--out", default="data/raw/youtube_binary_search", help="결과 저장 폴더")
    args = p.parse_args()

    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY가 없습니다. .env 확인")

    youtube = build("youtube", "v3", developerKey=api_key)

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc) if args.end is None else datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    os.makedirs(args.out, exist_ok=True)

    # 매칭용 키워드 목록: "완성 단어"만 들어가야 함
    keywords_all = args.keywords

    rows = []
    for kw in args.keywords:
        low, high, trace = binary_search_first_presence(
            youtube=youtube,
            keyword=kw,
            start=start,
            end=end,
            keywords_all=keywords_all,
            max_videos=args.max_videos,
            region=args.region,
            max_comment_pages=args.max_comment_pages,
            sleep_sec=args.sleep,
            max_iters=args.iters,
        )

        found = (high is not None)

        rows.append({
            "keyword": kw,
            "found_presence": found,
            "lower_bound_no_presence": low.date().isoformat(),
            "upper_bound_presence": None if high is None else high.date().isoformat(),
            "window_days": None if high is None else (high - low).days,
            "trace": str(trace),
        })

        if found:
            print(f"[{kw}] no-presence <= {low.date()}  |  presence <= {high.date()}  | window ~ {(high-low).days} days")
        else:
            print(f"[{kw}] presence NOT FOUND in range. last no-presence <= {low.date()}  | try increasing max_videos/pages or changing query.")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out, f"binary_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
