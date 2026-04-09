# 프로젝트 작업 로그 (교수님/논문 대비)
최종 업데이트: 2026-04-08 (Asia/Seoul)

## 1) 사용한 데이터 파일
| 구분 | 파일 경로 | 용도 | 비고 |
|---|---|---|---|
| 학습(초기) | (파일명 미기록) | 이진분류 50샘플 초기 학습 | 출력 모델만 확인 가능 (`outputs/kcelectra_hate/final_model`) |
| 학습(2차) | `C:\Users\woozz\Downloads\50개씩 데이터 완성.xlsx` | 이진분류 이어학습 | `no` 50 + (`misogyny`,`derivation`) 100 |
| 학습(3차) | `C:\Users\woozz\Downloads\0321_dishwashing_labeled_330_binary_for_train.xlsx` | 이진분류 이어학습 | 순서정렬본(`no` 122 + hate 208) |
| 학습(4차) | `C:\Users\woozz\Downloads\0321_dishwashing_labeled_330_reviewed.xlsx` | 3클래스 학습 | 라벨컬럼 `judgment` 직접 사용 |
| 변환용 원본 | `C:\Users\woozz\Downloads\FM정치_kw1.csv` | 댓글 구조 변환(wide->long) | `comment_1_text`~`comment_49_text` |
| 변환 결과 | `C:\Users\woozz\Downloads\FM정치_kw1_template_format.csv` | 템플릿 구조 추론 입력 | 13,618행 |
| 변환 결과(엑셀) | `C:\Users\woozz\Downloads\FM정치_kw1_template_format.xlsx` | 검수/공유용 | 템플릿 컬럼 순서 동일 |
| 이진 추론 결과 | `C:\Users\woozz\Downloads\FM정치_kw1_template_format_predicted.csv` | 배치 분류 결과 | `hate` 11,493 / `non_hate` 2,125 |

## 2) 전처리 방식
### 공통 텍스트 구성 방식
- 우선순위:
1. `comment_text` 존재 시: `[KW] matched_keyword [COMMENT] comment_text`
2. 댓글 비어 있으면: `[KW] matched_keyword [TITLE] TITLE [POST] TEXT`
- 빈 텍스트는 학습/추론에서 제외.

### 이진분류용 라벨링 방식(초기~330모델)
- 기존 스크립트 레거시 모드 사용:
1. 상위 `non_hate_count` 행 -> `0 (non_hate)`
2. 그 다음 `hate_count` 행 -> `1 (hate)`
- 따라서 행 순서가 중요했고, 330 데이터는 학습 전에 순서 정렬 파일 생성.

### 3클래스 라벨링 방식(최신)
- `label_column=judgment` 사용.
- 클래스 순서(고정): `no, derivation, misogyny`.
- 매핑:
1. `no -> 0`
2. `derivation -> 1`
3. `misogyny -> 2`
- 제외 라벨: `cannot judgment`(해당 시 제거).

### FM CSV 구조 변환 (wide -> long)
- 입력은 게시글 1행에 댓글 컬럼이 다수(`comment_1_text`~`comment_49_text`)인 구조.
- 각 댓글을 1행으로 펼쳐 `comment_text`, `comment_up`로 정규화.
- 템플릿 컬럼 순서(`0321_dishwashing_labeled_330_binary_for_train.xlsx`)와 동일하게 맞춤.

## 3) 학습 방식/설정
### 모델 및 학습 기본값
- 베이스 모델: `beomi/KcELECTRA-base-v2022` 또는 이전 학습 결과 `final_model` 경로.
- 기본 하이퍼파라미터:
1. `epochs=8`
2. `batch_size=8`
3. `learning_rate=2e-5`
4. `weight_decay=0.01`
5. `eval_size=0.2` (stratified split)
6. `seed=42`

### 선정 지표
- 이진분류 학습 시: `f1` 기준 best checkpoint 선택.
- 3클래스 학습 시: `macro_f1` 기준 best checkpoint 선택.

## 4) 학습 실행 이력 (확정)
| 순번 | 학습 시각(모델 산출물 기준) | 입력 데이터 | 분류 | 출력 모델 | 주요 지표 |
|---|---|---|---|---|---|
| 1 | 2026-03-12 22:34:44 | 초기 50샘플(파일명 미기록) | 이진 | `outputs/kcelectra_hate/final_model` | best `eval_f1=0.8889` |
| 2 | 2026-03-30 16:43:07 | `50개씩 데이터 완성.xlsx` | 이진 | `outputs/kcelectra_hate_150/final_model` | best `eval_f1=0.9091` |
| 3 | 2026-04-01 20:13:34 | `0321_dishwashing_labeled_330_binary_for_train.xlsx` | 이진 | `outputs/kcelectra_hate_330/final_model` | best `eval_f1=0.8723` |
| 4 | 2026-04-03 15:47:40 | `0321_dishwashing_labeled_330_reviewed.xlsx` | 3클래스 | `outputs/kcelectra_hate_3class_330/final_model` | best `eval_macro_f1=0.7676`, best checkpoint `checkpoint-165` |

## 5) 재현용 핵심 명령어
### 3클래스 학습
```powershell
python "C:\Users\woozz\Documents\New project\train_kcelectra_hate_classifier.py" `
  --excel_path "C:\Users\woozz\Downloads\0321_dishwashing_labeled_330_reviewed.xlsx" `
  --text_column "comment_text" `
  --keyword_column "matched_keyword" `
  --title_column "TITLE" `
  --body_column "TEXT" `
  --label_column "judgment" `
  --class_labels "no,derivation,misogyny" `
  --model_name "C:\Users\woozz\Documents\New project\outputs\kcelectra_hate_330\final_model" `
  --output_dir "C:\Users\woozz\Documents\New project\outputs\kcelectra_hate_3class_330"
```

### 3클래스 모델 단일 문장 테스트
```powershell
python "C:\Users\woozz\Documents\New project\predict_kcelectra_hate_classifier.py" `
  --model_dir "C:\Users\woozz\Documents\New project\outputs\kcelectra_hate_3class_330\final_model" `
  --text "설거지론은 문제다"
```

### 3클래스 모델 배치 추론
```powershell
python "C:\Users\woozz\Documents\New project\predict_kcelectra_hate_classifier.py" `
  --model_dir "C:\Users\woozz\Documents\New project\outputs\kcelectra_hate_3class_330\final_model" `
  --input_path "C:\Users\woozz\Downloads\FM정치_kw1_template_format.csv" `
  --text_column "comment_text" `
  --keyword_column "matched_keyword" `
  --output_path "C:\Users\woozz\Downloads\FM정치_kw1_template_format_predicted_3class.csv"
```

## 6) 현재 권장 모델
- 이진 과제(`non_hate/hate`): `outputs/kcelectra_hate_330/final_model`
- 3클래스 과제(`no/derivation/misogyny`): `outputs/kcelectra_hate_3class_330/final_model`

## 7) 리스크/보완 메모
- 초기 50샘플 원본 파일명은 로그 누락(필수 보완).
- 앞으로는 매 학습마다 아래 4개를 동시에 기록할 것:
1. 입력 파일 절대경로 + 해시(MD5/SHA256)
2. 전처리 스크립트/옵션
3. 학습 명령 전체
4. 최종 메트릭/모델 출력 경로


## 8) 2026-04-08 추가 추론 로그
- 요청: 5000개 복사본_김진우.xlsx 분류
- 원본 입력: C:\Users\woozz\Desktop\5000개 복사본_김진우.xlsx
- 전처리: 헤더 없는 15열 파일을 컬럼명 복원하여 정규화
  - 정규화 파일: C:\Users\woozz\Desktop\5000개 복사본_김진우__normalized.xlsx
  - comment_text 기준 비어있지 않은 행: 5,000
- 사용 모델(3클래스): outputs/kcelectra_hate_3class_330/final_model
- 출력 파일: C:\Users\woozz\Desktop\5000개 복사본_김진우_predicted_3class.xlsx
- 예측 분포(총 5,000):
  - misogyny: 3,412
  - derivation: 911
  - 
o: 677
