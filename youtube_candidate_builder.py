# youtube_candidate_builder.py
# ------------------------------------------------------------
# 목적:
# ① 키워드 검색(search API)으로 영상 후보 풀 만들기 (키워드 × 기간별)
# ② 채널 리스트(10~20개) 업로드(Uploads) 플레이리스트에서 영상 전수 수집
# ③ 두 풀을 합쳐 video 후보 CSV 생성:
#    video_id | channel_id | channel_name | title | published_at | source
#
# 준비:
#   pip install google-api-python-client pandas python-dateutil
#
# 실행 예시:
#   python youtube_candidate_builder.py \
#     --api_key "YOUR_KEY" \
#     --keyword "설거지론" --keyword "퐁퐁론" \
#     --period "2020-01-01,2020-12-31" --period "2021-01-01,2021-12-31" \
#     --channel "@SomeChannelHandle" --channel "UCxxxxxxxxxxxxxxxxxxxx" \
#     --out candidates.csv
#
# 참고:
# - 댓글은 여기서 안 긁음. "어떤 영상을 댓글 분석 대상으로 삼을지" 후보를 넓게 만들기만 함.
# - 채널 입력은 'UC...' 채널ID 또는 '@handle' 또는 채널명(검색) 모두 지원(정확도: 채널ID가 가장 좋음).
# ------------------------------------------------------------

import argparse
import re
import time
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
from dateutil import parser as dtparser
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def iso(dt_str: str) -> str:
    # 'YYYY-MM-DD' 같은 입력을 ISO 8601(Z)로 변환
    # YouTube API는 RFC3339/ISO8601 형식 요구
    dt = dtparser.parse(dt_str)
    if dt.tzinfo is None:
        # naive면 UTC로 간주
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.astimezone(dtparser.tz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def chunked(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def safe_api_call(func, max_retries: int = 6, base_sleep: float = 1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            # 429/5xx/일부 403(quota) 재시도
            if status in (403, 429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(base_sleep * (2 ** attempt))
                continue
            raise


def resolve_channel_id(youtube, channel_input: str) -> Optional[str]:
    """
    입력값이 채널ID(UC...), @handle, 또는 검색어(채널명)일 수 있음.
    가장 안정적인 건 UC... 채널ID를 직접 넣는 것.
    """
    s = channel_input.strip()

    # 이미 채널ID면 바로 반환
    if s.startswith("UC") and len(s) >= 10:
        return s

    # URL에서 UC... 추출
    m = re.search(r"(UC[A-Za-z0-9_-]{10,})", s)
    if m:
        return m.group(1)

    # @handle이면 채널 검색으로 해결
    # (정확도 높이려면 handle 그대로 q로 검색)
    q = s
    if s.startswith("@"):
        q = s

    def _call():
        return youtube.search().list(
            part="snippet",
            q=q,
            type="channel",
            maxResults=5,
        ).execute()

    resp = safe_api_call(_call)
    items = resp.get("items", [])
    if not items:
        return None

    # 가장 상단 결과를 채널로 채택(채널ID를 직접 넣는 게 가장 안전)
    return items[0]["snippet"].get("channelId")


def get_uploads_playlist_id(youtube, channel_id: str) -> Optional[str]:
    def _call():
        return youtube.channels().list(
            part="contentDetails",
            id=channel_id,
            maxResults=1,
        ).execute()

    resp = safe_api_call(_call)
    items = resp.get("items", [])
    if not items:
        return None

    return (
        items[0]
        .get("contentDetails", {})
        .get("relatedPlaylists", {})
        .get("uploads")
    )


def collect_videos_from_uploads_playlist(youtube, uploads_playlist_id: str) -> List[Dict]:
    """
    업로드 플레이리스트의 모든 영상ID 수집.
    playlistItems의 contentDetails.videoPublishedAt도 함께 얻음.
    """
    out = []
    token = None

    while True:
        def _call():
            return youtube.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=50,
                pageToken=token
            ).execute()

        resp = safe_api_call(_call)
        items = resp.get("items", [])

        for it in items:
            cd = it.get("contentDetails", {})
            sn = it.get("snippet", {})
            vid = cd.get("videoId")
            if not vid:
                continue
            out.append({
                "video_id": vid,
                # playlist item에서 제공되는 업로드 시각(없을 수도 있음)
                "published_at": cd.get("videoPublishedAt") or sn.get("publishedAt") or "",
                "title": sn.get("title") or "",
                "channel_id": sn.get("channelId") or "",
                "channel_name": sn.get("channelTitle") or "",
            })

        token = resp.get("nextPageToken")
        if not token:
            break

    return out


def collect_videos_by_keyword_search(
    youtube,
    keyword: str,
    published_after: str,
    published_before: str,
    max_pages: Optional[int] = None,
) -> List[Dict]:
    """
    search.list로 영상 후보 수집 (키워드 × 기간).
    search는 제목/설명(일부 메타) 기반.
    """
    out = []
    token = None
    page = 0

    while True:
        def _call():
            return youtube.search().list(
                part="snippet",
                q=keyword,
                type="video",
                maxResults=50,
                pageToken=token,
                publishedAfter=published_after,
                publishedBefore=published_before,
                order="date",          # 기간 내에서 최신부터 오지만, 후보 풀 만드는 용도라 OK
                safeSearch="none"
            ).execute()

        resp = safe_api_call(_call)
        items = resp.get("items", [])

        for it in items:
            sn = it.get("snippet", {})
            vid = it.get("id", {}).get("videoId")
            if not vid:
                continue
            out.append({
                "video_id": vid,
                "published_at": sn.get("publishedAt") or "",
                "title": sn.get("title") or "",
                "channel_id": sn.get("channelId") or "",
                "channel_name": sn.get("channelTitle") or "",
            })

        token = resp.get("nextPageToken")
        page += 1

        if not token:
            break
        if max_pages is not None and page >= max_pages:
            break

    return out


def hydrate_with_videos_list(youtube, records: List[Dict]) -> List[Dict]:
    """
    search/playlistItems의 snippet이 부정확하거나 누락될 수 있어서,
    videos.list(part=snippet)로 제목/채널명/게시일을 다시 한 번 정규화.
    """
    by_id = {r["video_id"]: r for r in records if r.get("video_id")}
    video_ids = list(by_id.keys())

    for batch in chunked(video_ids, 50):
        def _call():
            return youtube.videos().list(
                part="snippet",
                id=",".join(batch),
                maxResults=50
            ).execute()

        resp = safe_api_call(_call)
        items = resp.get("items", [])
        for it in items:
            vid = it.get("id")
            sn = it.get("snippet", {})
            if not vid or vid not in by_id:
                continue
            # 더 신뢰할 수 있는 값으로 업데이트
            by_id[vid]["title"] = sn.get("title") or by_id[vid].get("title", "")
            by_id[vid]["channel_id"] = sn.get("channelId") or by_id[vid].get("channel_id", "")
            by_id[vid]["channel_name"] = sn.get("channelTitle") or by_id[vid].get("channel_name", "")
            by_id[vid]["published_at"] = sn.get("publishedAt") or by_id[vid].get("published_at", "")

    return list(by_id.values())


def merge_sources(base: Dict, add_source: str):
    if "source" not in base or not base["source"]:
        base["source"] = add_source
    else:
        # 중복 방지
        existing = set([s.strip() for s in base["source"].split(";") if s.strip()])
        if add_source not in existing:
            base["source"] = base["source"] + ";" + add_source


def build_candidates(
    api_key: str,
    keywords: List[str],
    periods: List[Tuple[str, str]],
    channels: List[str],
    out_csv: str,
    max_search_pages: Optional[int] = None,
    hydrate: bool = True,
) -> pd.DataFrame:
    youtube = build("youtube", "v3", developerKey=api_key)

    candidates: Dict[str, Dict] = {}

    # ① 키워드 × 기간별 search
    for kw in keywords:
        for (start, end) in periods:
            pa = iso(start)
            pb = iso(end)
            items = collect_videos_by_keyword_search(
                youtube=youtube,
                keyword=kw,
                published_after=pa,
                published_before=pb,
                max_pages=max_search_pages,
            )
            for r in items:
                vid = r["video_id"]
                if vid not in candidates:
                    candidates[vid] = {
                        "video_id": vid,
                        "channel_id": r.get("channel_id", ""),
                        "channel_name": r.get("channel_name", ""),
                        "title": r.get("title", ""),
                        "published_at": r.get("published_at", ""),
                        "source": "",
                    }
                merge_sources(candidates[vid], f"keyword:{kw}@{start}..{end}")

    # ② 채널 업로드 플레이리스트 수집
    for ch in channels:
        ch_id = resolve_channel_id(youtube, ch)
        if not ch_id:
            print(f"[warn] channel not resolved: {ch}")
            continue

        upl = get_uploads_playlist_id(youtube, ch_id)
        if not upl:
            print(f"[warn] uploads playlist not found for channel: {ch_id}")
            continue

        vids = collect_videos_from_uploads_playlist(youtube, upl)
        for r in vids:
            vid = r["video_id"]
            if vid not in candidates:
                candidates[vid] = {
                    "video_id": vid,
                    "channel_id": r.get("channel_id", ""),
                    "channel_name": r.get("channel_name", ""),
                    "title": r.get("title", ""),
                    "published_at": r.get("published_at", ""),
                    "source": "",
                }
            merge_sources(candidates[vid], f"channel:{ch_id}")

    records = list(candidates.values())

    # (권장) videos.list로 제목/채널명/게시일 정규화
    if hydrate and records:
        records = hydrate_with_videos_list(youtube, records)

    df = pd.DataFrame(records)

    # 컬럼 순서 정렬
    cols = ["video_id", "channel_id", "channel_name", "title", "published_at", "source"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]

    # (선택) 게시일 기준 정렬 (오래된 순 / 최신 순 필요에 따라 바꾸면 됨)
    # 여기서는 오래된 순으로 정렬해서 "초기 등장" 탐색에 유리하게.
    df["published_at_sort"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.sort_values(["published_at_sort", "video_id"]).drop(columns=["published_at_sort"])

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def parse_periods(period_args: List[str]) -> List[Tuple[str, str]]:
    """
    입력 예: "2020-01-01,2020-12-31"
    """
    periods = []
    for p in period_args:
        if "," not in p:
            raise ValueError(f"period must be 'start,end' : {p}")
        a, b = [x.strip() for x in p.split(",", 1)]
        periods.append((a, b))
    return periods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", required=True, help="YouTube Data API v3 key")
    ap.add_argument("--keyword", action="append", default=[], help="Keyword (repeatable)")
    ap.add_argument("--period", action="append", default=[], help="Period 'start,end' (repeatable)")
    ap.add_argument("--channel", action="append", default=[], help="Channel (UC.. or @handle or name) repeatable")
    ap.add_argument("--out", default="candidates.csv", help="Output CSV path")
    ap.add_argument("--max_search_pages", type=int, default=0, help="0 means no limit (be careful with quota)")
    ap.add_argument("--no_hydrate", action="store_true", help="Skip videos.list normalization")
    args = ap.parse_args()

    keywords = [k.strip() for k in args.keyword if k and k.strip()]
    channels = [c.strip() for c in args.channel if c and c.strip()]
    if not args.period:
        raise SystemExit("ERROR: provide at least one --period 'YYYY-MM-DD,YYYY-MM-DD'")

    periods = parse_periods(args.period)

    if not keywords and not channels:
        raise SystemExit("ERROR: provide at least one --keyword or --channel")

    max_pages = None if args.max_search_pages == 0 else args.max_search_pages

    df = build_candidates(
        api_key=args.api_key,
        keywords=keywords,
        periods=periods,
        channels=channels,
        out_csv=args.out,
        max_search_pages=max_pages,
        hydrate=(not args.no_hydrate),
    )

    print(f"[done] candidates={len(df)} saved -> {args.out}")


if __name__ == "__main__":
    main()
