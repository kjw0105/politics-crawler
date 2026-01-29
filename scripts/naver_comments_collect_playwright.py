import os
import re
import time
import argparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

def clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def extract_comments_from_html(html: str):
    """
    네이버 뉴스 댓글 DOM은 자주 바뀜.
    그래서 '텍스트 후보 셀렉터'를 여러 개 두고 최대한 뽑아봄.
    """
    soup = BeautifulSoup(html, "lxml")
    comments = []

    # 댓글 텍스트 후보들(변경 대비)
    text_selectors = [
        "span.u_cbox_contents",
        "span.u_cbox_text",
        "span.u_cbox_comment_contents",
        "div.u_cbox_text_wrap span",
    ]
    date_selectors = [
        "span.u_cbox_date",
        "span.u_cbox_date_time",
        "span.u_cbox_info_base",
    ]

    texts = []
    for sel in text_selectors:
        found = soup.select(sel)
        if found:
            texts = found
            break

    # 날짜는 안 잡힐 수 있음(없으면 빈 값으로 둠)
    dates = []
    for sel in date_selectors:
        found = soup.select(sel)
        if found:
            dates = found
            break

    for i, t in enumerate(texts):
        txt = clean(t.get_text(" ", strip=True))
        dt = ""
        if i < len(dates):
            dt = clean(dates[i].get_text(" ", strip=True))
        if txt:
            comments.append((txt, dt))

    return comments

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", required=True, help="naver_only.csv 경로( nav­er_url 컬럼 필요 )")
    p.add_argument("--keywords", nargs="+", default=["설거지론"], help="필터 키워드(포함되는 댓글만 저장)")
    p.add_argument("--max_articles", type=int, default=50, help="테스트용: 앞에서 몇 개 기사만")
    p.add_argument("--max_more_clicks", type=int, default=30, help="댓글 더보기 클릭 횟수(많을수록 더 긁음)")
    p.add_argument("--sleep", type=float, default=0.6, help="클릭/로딩 대기")
    p.add_argument("--out_dir", default="data/raw/naver_comments")
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)
    if "naver_url" not in df.columns:
        raise RuntimeError("in_csv에 naver_url 컬럼이 없습니다. filter 단계부터 다시!")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"naver_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    rows = []

    with sync_playwright() as pwp:
        browser = pwp.chromium.launch(headless=True)  # 디버깅하려면 False
        page = browser.new_page()

        for _, r in tqdm(df.head(args.max_articles).iterrows(), total=min(len(df), args.max_articles), desc="Articles"):
            url = r["naver_url"]
            title = clean(str(r.get("title", "")))
            pubDate = clean(str(r.get("pubDate", "")))

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                time.sleep(args.sleep)

                # "댓글" 영역을 열기 시도 (버튼/탭이 있을 때만)
                # (없어도 그냥 진행)
                for sel in ["a[href*='comment']", "button:has-text('댓글')", "a:has-text('댓글')"]:
                    try:
                        if page.locator(sel).first.is_visible(timeout=1500):
                            page.locator(sel).first.click(timeout=1500)
                            time.sleep(args.sleep)
                            break
                    except:
                        pass

                # 댓글 "더보기" 버튼 반복 클릭(있을 때만)
                for _ in range(args.max_more_clicks):
                    clicked = False
                    for sel in [
                        "a.u_cbox_more",
                        "button.u_cbox_more",
                        "a:has-text('더보기')",
                        "button:has-text('더보기')",
                    ]:
                        try:
                            loc = page.locator(sel).first
                            if loc.is_visible(timeout=800):
                                loc.click(timeout=1500)
                                time.sleep(args.sleep)
                                clicked = True
                                break
                        except:
                            pass
                    if not clicked:
                        break

                html = page.content()
                comments = extract_comments_from_html(html)

                # 키워드 필터
                for (text, dt) in comments:
                    hit = next((k for k in args.keywords if k in text), None)
                    if hit:
                        rows.append({
                            "keyword": hit,
                            "article_title": title,
                            "article_url": url,
                            "article_pubDate": pubDate,
                            "comment_text": text,
                            "comment_date_raw": dt,
                            "collected_at": datetime.now().isoformat(timespec="seconds"),
                        })

            except Exception as e:
                rows.append({
                    "keyword": "",
                    "article_title": title,
                    "article_url": url,
                    "article_pubDate": pubDate,
                    "comment_text": "",
                    "comment_date_raw": "",
                    "collected_at": datetime.now().isoformat(timespec="seconds"),
                    "error": str(e),
                })

        browser.close()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved:", out_path, "rows=", len(out_df))

if __name__ == "__main__":
    main()
