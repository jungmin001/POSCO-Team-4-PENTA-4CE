============================================================
CH.4 설비 이상 분석·예측 및 정비 의사결정
============================================================

1. 이 챕터가 하는 일
------------------------------------------------------------
설비 7대의 센서와 정비 기록으로

- 지금 이상한지 알아채고 (이상 감지 · 상태 진단)
- 어디가 문제인지 원인 후보를 짚고
- 24시간 / 6시간 안에 고장 날 위험을 예측하고
- 비용을 비교해 정비 선택지를 추천하고
- 그 결과를 언리얼 디지털 트윈 화면이 읽는 파일로 내보냅니다.

설비 이름:
A 연속소둔로 / B 코일 용접기 / C 연주 주편 / D 조질압연기
E1, E2 소결 배풍기 / F 슬러리 교반기


2. 파트별 구성
------------------------------------------------------------
[전처리]
Main.py, src/
- 시나리오별 정지 · 결측 · 정비 구간 · 이상치 표시

[이상탐지]
AnomalyModel.py
- C 연주 정상범위 기반 이상 경보 (이 문서 아래쪽에 설명)

[상태 · 원인 진단]
Diagnosis.py, diagnosis/
- 설비 7대의 상태와 원인 후보 (자세한 방법은 diagnosis/README.md)

[고장 위험 예측]
jungmin/
- 설비 7대의 6 / 24시간 내 고장 확률
- 조질압연기(D) 상세 모델 (mill_d.py)
- 언리얼용 파일 생성 (unreal.py)

[최종 결함 위험]
chanhee/Analysis.py
- C 검사 직후 최종(D 공정) 결함 위험, 생산 흐름 단위

[잔여수명]
hee.py
- CC1 잔여수명 (생존분석은 scikit-survival이 있을 때만 실행)

[언리얼 연동]
Bundle.py
- 파트별 결과를 Results/Unreal 한 곳에 모음


3. 폴더 구조
------------------------------------------------------------
CH.4/
│
├─ Main.py                    전체 실행 진입점
├─ Bundle.py                  언리얼용 결과 모으기
├─ AnomalyModel.py            이상탐지 모델
├─ anomaly_config.json        이상탐지 설정 (선택사항)
├─ Diagnosis.py               진단 실행 파일
├─ hee.py                     잔여수명
│
├─ src/                       전처리
├─ diagnosis/                 상태 · 원인 진단
├─ jungmin/                   고장 위험 예측, 조질압연기 모델, 언리얼 파일 생성
├─ chanhee/                   최종 결함 위험
│
├─ Data/
│  ├─ Raw/                    원본 (manifest.json 포함)
│  └─ Processed/Scenarios/    전처리 결과 (S0001 ~ S0360)
│
├─ model_output/              이상탐지 모델 결과
│
├─ Results/
│  ├─ Diagnosis/              진단 결과 · 모델 · 예측
│  ├─ Quality/                최종 결함 위험 모델 · 예측
│  └─ Unreal/                 언리얼이 읽는 결과 (SCHEMA.md 참고)
│
└─ Reports/                   발표 자료


4. 필요한 Python 패키지
------------------------------------------------------------
numpy
pandas
scikit-learn
joblib
matplotlib

선택:
scikit-survival  (hee.py 생존분석)

설치가 안 되어 있다면 VS Code 터미널에서:

python -m pip install numpy pandas scikit-learn joblib matplotlib

진단(Diagnosis.py)은 아래 파일의 버전으로 맞추는 것을 권장합니다.

python -m pip install -r requirements-diagnosis.txt

※ scikit-learn 버전이 다르면 진단 성능이 소수점 단위로 달라질 수 있습니다.


5. 전체 실행 방법
------------------------------------------------------------
CH.4 폴더에서:

python Main.py all

이미 만든 단계는 건너뛸 수 있습니다.

python Main.py all skip=preprocess

python Main.py all skip=preprocess,anomaly,diagnosis

한 단계가 실패하면 그 단계 이름을 알려 주고 멈춥니다.
시간이 오래 걸리는 단계는 전처리, 진단, 고장 위험 예측입니다.


단계 순서:

preprocess
    ↓
anomaly
    ↓
diagnosis
    ↓
diagnosis_demo
    ↓
risk
    ↓
unreal
    ↓
quality
    ↓
rul
    ↓
bundle


단계별 하는 일:

[preprocess]
시나리오별 전처리 데이터 저장

[anomaly]
C 연주 이상탐지 모델 학습

[diagnosis]
설비 7대의 상태 · 원인 진단 학습 · 예측

[diagnosis_demo]
저장된 진단 모델로 validation 시나리오의 화면용 예측 생성
(재학습 없음, 시연용)

[risk]
설비 7대의 고장 위험 학습 · 평가 · 모델 저장

[unreal]
언리얼용 온톨로지 · 비용 · 위험 예측 파일 생성

[quality]
C 검사 직후 최종 결함 위험 예측 (생산 흐름 단위)

[rul]
CC1 잔여수명

[bundle]
언리얼 출력을 Results/Unreal에 모으기


단계 하나만 실행할 때:

python Main.py                       전처리만 (설비 A~F, flow, 정답, history)
python Main.py check [설비...]       판정 결과를 화면에 출력 (저장하지 않음)
python Diagnosis.py                  진단
python jungmin/Main.py A B C D E1 E2 F noshow    고장 위험 예측
python jungmin/unreal.py             언리얼용 파일 생성
python chanhee/Analysis.py           최종 결함 위험
python Bundle.py                     언리얼 출력 모으기
python jungmin/mill_d.py             조질압연기(D) 모델 (Main.py all에는 포함되지 않음)
python jungmin/mill_report.py        조질압연기(D) 리포트


6. 전처리 기준
------------------------------------------------------------
기준값은 src/config.py에 있습니다.

- 정지:       가동 여부 RUNNING = 0
- 결측:       센서 값이 비었거나 품질 표시가 ok가 아님
              연속 3행 이하만 직전 값으로 채움
- 정비 구간:  정비 이력의 시작 이상, 종료 미만
- 이상치:     직전 11개 값의 중앙값에서 10 × 척도(MAD) 이상 벗어남
              (E1, E2는 시나리오당 행이 적어 직전 5개)

행은 삭제하지 않고
IS_MISSING / IS_STOPPED / IS_MAINTENANCE / IS_OUTLIER 열로 표시합니다.

정지 · 결측 · 정비 구간의 행은 이상치 판정에서 제외합니다.

정답(GT_ 열)은 모델 입력으로 쓰지 않습니다.


7. 결과 파일
------------------------------------------------------------
Results/Unreal/
│
├─ manifest.json                 어떤 파일이 있는지 (먼저 읽기)
├─ SCHEMA.md                     파일 형식
├─ UNREAL_PROMPT.md              언리얼 작업 지시문
├─ UNREAL_PROMPT_ADDON.md        진단 · 결함 위험 표시 추가 지시문
│
├─ ontology.json, costs.json     설비 · 부품 구조, 비용 가정값
├─ risk_{설비}.json              고장 위험 예측 (설비별)
├─ decisions_{설비}_check.csv    비용 계산 확인표
├─ diagnosis_{설비}.csv          상태 · 원인 진단 예측
├─ diagnosis_metrics.csv         진단 성능
├─ flow_quality.csv              최종 결함 위험
├─ quality_summary.json          최종 결함 위험 성능
└─ anomaly_model.json            이상탐지 모델

언리얼 프로젝트의 DSSData 폴더로 이 폴더를 복사해서 사용합니다.


8. 주요 성능 (검증 시나리오 72개 기준)
------------------------------------------------------------
- 실제 고장 47건 중 40건에서 고장 전에 경보
  첫 경보부터 고장까지 보통 약 30시간
- 고장이 없는 설비에서 경보가 울린 비율 약 3%
- 조질압연기(D): 고장 구간 11개 중 10개 감지, 오경보 하루 0.03회
- 상태 진단 정확도 설비별 92 ~ 99.9%
- 최종 결함 위험: 결함 308개 중 299개 발견 (오경보 96개)

※ 합성 데이터의 validation 결과입니다.
   실제 설비 데이터로는 다시 확인해야 합니다.


9. 주의사항
------------------------------------------------------------
- 데이터는 합성 데이터셋(pdm-v2.2.0)입니다.
  72시간 시나리오 360개 (train 216 / validation 72 / test 72)

- 언리얼 시연은 미리 계산한 예측을 시간순으로 재생하는 방식입니다.
  실시간 입력이 아닙니다.

- 고장 위험, 진단, 최종 결함 위험은 모두
  validation 시나리오(S0217 ~ S0288) 기준입니다.

- 비용 금액은 가정값입니다.

- 6시간 예측과 A · C · F 설비의 원인 진단은 참고용입니다.
  원인은 확정이 아니라 "후보"로 표시합니다.

- Results/Diagnosis의 models / features / predictions와
  jungmin/outputs/models 는 용량이 커서 없을 수 있습니다.
  없으면 Diagnosis.py, jungmin/Main.py를 다시 실행해서 만듭니다.


아래부터는 이상탐지 모델(AnomalyModel.py) 설명입니다.


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
├─ AnomalyModel.py
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

AnomalyModel.py를 VS Code에서 ▶ 실행하면 됩니다.


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

1) AnomalyModel.py와 anomaly_config.json을 CH.4에 넣기
2) RunMode = "train"
3) ▶ 실행
4) model_output/anomaly_model.json 확인
5) threshold_persistence_comparison.csv 확인
6) 새 파일 예측 시 RunMode = "predict"
============================================================