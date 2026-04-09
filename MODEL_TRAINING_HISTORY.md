# MODEL TRAINING HISTORY
Updated: 2026-04-08 (Asia/Seoul)

Detailed Korean report for professor/paper:
- `C:\Users\woozz\Documents\New project\PROJECT_WORKLOG_KR.md`

## Summary
1. Initial binary model (50 samples, source filename not preserved):
   - `outputs/kcelectra_hate/final_model`
2. Continued binary training (150):
   - Input: `C:\Users\woozz\Downloads\50개씩 데이터 완성.xlsx`
   - Output: `outputs/kcelectra_hate_150/final_model`
3. Continued binary training (330, reordered binary file):
   - Input: `C:\Users\woozz\Downloads\0321_dishwashing_labeled_330_binary_for_train.xlsx`
   - Output: `outputs/kcelectra_hate_330/final_model`
4. Multiclass training (3-class: no/derivation/misogyny):
   - Input: `C:\Users\woozz\Downloads\0321_dishwashing_labeled_330_reviewed.xlsx`
   - Output: `outputs/kcelectra_hate_3class_330/final_model`

## Inference conversion record
- Source wide CSV: `C:\Users\woozz\Downloads\FM정치_kw1.csv`
- Converted long format:
  - `C:\Users\woozz\Downloads\FM정치_kw1_template_format.csv`
  - `C:\Users\woozz\Downloads\FM정치_kw1_template_format.xlsx`
- Binary prediction output:
  - `C:\Users\woozz\Downloads\FM정치_kw1_template_format_predicted.csv`
