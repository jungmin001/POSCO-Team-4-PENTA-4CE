============================================================
연속주조 자동 이상탐지 모델
============================================================

1. 이 버전의 특징
------------------------------------------------------------
- openpyxl 설치 필요 없음
- Excel(.xlsx) 파일 생성 안 함
- VS Code에서 실행 버튼(▶)만 눌러도 학습 가능
- 실행할 때마다 같은 결과 파일을 덮어씀
- 결과 파일이 계속 늘어나지 않음
- 모델 결과는 JSON + CSV만 사용

핵심 분석 방식:
- Robust Z-score
- 정상 데이터 Median + MAD
- 95 / 97 / 99% 경보 기준 비교
- 1 / 2 / 3분 지속시간 비교
- 조건에 맞는 본 경보 자동 선택
- quality.csv가 있으면 품질 결과까지 자동 연결


2. 파일 배치
------------------------------------------------------------
CH.4/
│
├─ model_clean_csv.py
├─ anomaly_config.json        ← 선택사항
│
└─ Data/
   └─ Processed/
      └─ Scenarios/
         ├─ S0001/
         │  ├─ C.csv
         │  └─ flow/
         │     └─ quality.csv
         │
         ├─ S0002/
         └─ ...


※ anomaly_config.json은 없어도 실행됩니다.
   파일이 없으면 Python 코드 안의 기본 설정을 사용합니다.


3. 필요한 Python 패키지
------------------------------------------------------------
numpy
pandas

설치가 안 되어 있다면 VS Code 터미널에서:

python -m pip install numpy pandas

※ openpyxl은 필요 없습니다.


4. 학습 실행 방법
------------------------------------------------------------
코드 아래쪽 설정:

RunMode = "train"

상태로 두고

model_clean_csv.py를 VS Code에서 ▶ 실행하면 됩니다.


5. 학습할 때 자동으로 하는 일
------------------------------------------------------------
C.csv 자동 탐색
    ↓
시간 컬럼 자동 탐색
    ↓
센서 컬럼 자동 탐색
    ↓
실제 Fault / Stop / Outlier 확인
    ↓
정상 Scenario 학습
    ↓
Median + MAD 정상범위 생성
    ↓
95 / 97 / 99% Threshold 비교
    ↓
1 / 2 / 3분 지속조건 비교
    ↓
본 경보 자동 선택
    ↓
주의 경보 자동 설정
    ↓
quality.csv가 있으면 품질 연결
    ↓
anomaly_model.json 저장


6. 학습 후 생성되는 결과
------------------------------------------------------------
model_output/
│
├─ anomaly_model.json
│
├─ threshold_persistence_comparison.csv
│
├─ quality_group_compare.csv
│
└─ details/
   ├─ ground_truth_summary.csv
   ├─ scenario_evaluation.csv
   └─ quality_summary.csv


각 파일 의미:

[anomaly_model.json]
실제 저장된 이상탐지 모델입니다.

포함 내용:
- 사용한 센서
- 시간 컬럼
- Feature 설정
- 정상 Median
- MAD 기반 Scale
- Threshold
- 본 경보 지속시간
- 주의 경보 지속시간
- 학습 결과 요약


[threshold_persistence_comparison.csv]
95 / 97 / 99%와
1 / 2 / 3분을 모두 비교한 결과입니다.

예:
- 탐지 건수
- 미탐 건수
- 탐지율
- 오경보/일
- 선행시간


[quality_group_compare.csv]
품질 데이터가 있을 때만 생성됩니다.

예:
- 3분에서 놓쳤지만 2분에서 추가 탐지한 그룹
- 본 경보에서도 미탐한 그룹
- 최종 불량 Scenario 비율
- 최종 불량 FLOW 비율


[details/]
발표에는 꼭 필요하지 않은 상세 분석 결과입니다.


7. 다시 실행하면?
------------------------------------------------------------
새 파일이 계속 추가되는 방식이 아닙니다.

같은 이름의 결과를 삭제/덮어쓰기 때문에
model_output 폴더가 계속 지저분해지지 않습니다.


8. 새 C.csv 예측
------------------------------------------------------------
먼저 학습이 끝나서 아래 파일이 있어야 합니다.

model_output/anomaly_model.json

그 다음 코드 아래쪽을:

RunMode = "predict"

로 변경합니다.

새 파일은 기본적으로:

CH.4/new_data/

폴더 아래에 둡니다.

실행하면:

prediction_output/
└─ prediction_results.csv

하나로 저장됩니다.


9. prediction_results.csv 의미
------------------------------------------------------------
TIME
- 시간

ANOMALY_SCORE
- 이상점수

TOP_FEATURE
- 가장 크게 이상했던 Feature

TOP_SENSOR
- 가장 크게 이상했던 센서

WARNING_ALARM
- 주의 경보
- 예: 1분 지속

ACTION_ALARM
- 행동 경보
- 예: 2분 지속


10. 팀장님께 설명할 때
------------------------------------------------------------
“기존에는 97%와 2분 같은 조건을 분석 후 직접 선택했지만,
모델링 버전에서는 새로운 학습 데이터가 들어오면
정상범위, 경보선, 지속시간을 다시 비교하고
그 결과를 anomaly_model.json으로 저장합니다.

이후 같은 센서 구조의 새로운 C.csv가 들어오면
저장된 모델을 이용해 자동으로 이상점수와 경보를 생성합니다.”


11. 중요
------------------------------------------------------------
이 모델은 Random Forest가 아닙니다.

사용 기법:
Robust Z-score 기반 정상범위 이상탐지

정상 데이터의
Median + MAD를 학습하고,
정상범위에서 얼마나 벗어났는지를 이상점수로 계산합니다.


12. 설정 변경
------------------------------------------------------------
anomaly_config.json을 수정하면
Python 코드를 직접 수정하지 않고도 조건을 바꿀 수 있습니다.

예:
"threshold_percentiles": [95, 97, 99]

"persistence_options": [1, 2, 3]

"target_recall_24h": 95.0


============================================================
권장 사용 순서
============================================================

1) model_clean_csv.py와 anomaly_config.json을 CH.4에 넣기
2) RunMode = "train"
3) ▶ 실행
4) model_output/anomaly_model.json 확인
5) threshold_persistence_comparison.csv 확인
6) 새 파일 예측 시 RunMode = "predict"
============================================================