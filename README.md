# Dishwashing Comment Classification Project

유튜브 댓글에서 `설거지` 관련 문맥을 분리하고, KcELECTRA 기반 분류 모델을 학습/추론하는 프로젝트입니다.

## 주요 폴더
- `training_data_csv/`: 모델 훈련에 실제 사용한 데이터셋의 CSV 사본
- `outputs/`: 학습 결과 모델(`final_model`)과 체크포인트
- `PROJECT_WORKLOG_KR.md`: 실험/작업 로그(국문)
- `MODEL_TRAINING_HISTORY.md`: 학습 히스토리 요약

## 작업 기록 정리 규칙 (필수)
교수님 보고 및 논문 작성 대비를 위해, **모든 학습/재학습/평가 작업마다 아래 3가지를 반드시 기록**합니다.

1. 어떤 파일을 썼는지  
- 원본 파일 절대경로
- 사용한 분할/샘플 수
- 가능하면 해시(SHA256 등)

2. 어떤 방식으로 전처리했는지  
- 텍스트 구성 규칙(예: `comment_text` 우선, fallback 규칙)
- 라벨 매핑 규칙(예: `no/derivation/misogyny`)
- 제외 조건(결측/불명 라벨 등)

3. 어떻게 학습했는지  
- 실행 명령어 전체
- 베이스 모델/초기 가중치 경로
- 하이퍼파라미터(epochs, batch size, lr, seed 등)
- 최종 성능 지표와 저장 경로

## 기록 위치
- 상세 로그는 `PROJECT_WORKLOG_KR.md`에 누적 기록합니다.
- 훈련 데이터 파일 목록은 `training_data_csv/MANIFEST.md`를 기준으로 유지합니다.

