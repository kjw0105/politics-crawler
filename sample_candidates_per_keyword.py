# sample_candidates_per_keyword.py
# ------------------------------------------------------------
# candidates.csv에서 source 컬럼에 있는 "keyword:XXX@기간" 표시를 이용해
# 키워드별로 N개씩 영상 샘플링 -> candidates_sampled.csv 생성
#
# 샘플링 방식: 기본은 "오래된 순(확산 초기 탐색에 유리)"으로 N개
# (원하면 랜덤도 옵션으로 가능)
#
# 실행 예:
#   python sample_candidates_per_keyword.py --in candidates.csv --out candidates_sampled.csv --n 20
# ------------------------------------------------------------

import argparse
import re
import pandas as pd


def extract_keywords_from_source(source: str):
    """
    source 예시:
      keyword:여경@2020-01-01..2020-12-31;keyword:여경@2021-01-01..2021-12-31;channel:UC...
      keyword:"퐁퐁남"@...
    -> ["여경", "퐁퐁남", ...] 형태로 반환 (따옴표 제거)
    """
    if not isinstance(source, str):
        return []

    kws = []
    for part in source.split(";"):
        part = part.strip()
        if part.startswith("keyword:"):
            # keyword:XXX@.... 형식
            m = re.match(r"keyword:(.+?)@.+", part)
            if m:
                kw = m.group(1).strip()
                # keyword 값에 '"퐁퐁남"'처럼 따옴표가 들어있으면 제거
                kw = kw.strip('"').strip("'")
                kws.append(kw)
    # 중복 제거(순서 유지)
    return list(dict.fromkeys(kws))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="candidates.csv")
    ap.add_argument("--out", dest="out", default="candidates_sampled.csv")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", choices=["oldest", "newest", "random"], default="oldest")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    if "source" not in df.columns or "video_id" not in df.columns:
        raise SystemExit("ERROR: candidates.csv must contain 'video_id' and 'source' columns")

    # 각 행이 어떤 키워드로 뽑혔는지(복수 가능) 풀기
    rows = []
    for _, r in df.iterrows():
        kws = extract_keywords_from_source(r.get("source", ""))
        for kw in kws:
            rr = r.to_dict()
            rr["picked_by_keyword"] = kw
            rows.append(rr)

    if not rows:
        raise SystemExit("ERROR: no keyword:* found in source column. candidates.csv를 키워드로 만든 게 맞는지 확인해줘.")

    exp = pd.DataFrame(rows)

    # published_at 정규화(정렬용)
    if "published_at" in exp.columns:
        exp["_dt"] = pd.to_datetime(exp["published_at"], errors="coerce", utc=True)
    else:
        exp["_dt"] = pd.NaT

    out_parts = []
    for kw, g in exp.groupby("picked_by_keyword", dropna=True):
        g = g.drop_duplicates(subset=["video_id"])

        if args.mode == "oldest":
            g = g.sort_values(["_dt", "video_id"], ascending=[True, True])
            pick = g.head(args.n)
        elif args.mode == "newest":
            g = g.sort_values(["_dt", "video_id"], ascending=[False, True])
            pick = g.head(args.n)
        else:  # random
            pick = g.sample(n=min(args.n, len(g)), random_state=args.seed)

        out_parts.append(pick)

    sampled = pd.concat(out_parts, ignore_index=True)

    # 후보 파일 포맷 유지 + picked_by_keyword 컬럼 포함
    keep_cols = ["video_id", "channel_id", "channel_name", "title", "published_at", "source", "picked_by_keyword"]
    for c in keep_cols:
        if c not in sampled.columns:
            sampled[c] = ""
    sampled = sampled[keep_cols].drop_duplicates(subset=["video_id", "picked_by_keyword"])

    sampled.to_csv(args.out, index=False, encoding="utf-8-sig")

    # 요약 출력
    print(f"[done] saved -> {args.out}")
    print(sampled.groupby("picked_by_keyword")["video_id"].nunique().to_string())


if __name__ == "__main__":
    main()
