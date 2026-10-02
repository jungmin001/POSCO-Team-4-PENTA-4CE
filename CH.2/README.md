# PENTA-4CE — CH_2
### 주제F 터보팬 엔진 고장 임박 예측
---

## 챕터 개요

> NASA C-MAPSS 터보팬 엔진 데이터를 활용해 센서 기반 남은 수명(RUL)을 예측하고, 고장 임박 엔진과 정비 우선순위를 도출하는 프로젝트
>
> 열화 속도가 다른 100대 엔진을 대상으로 잔여 수명 기준에 따른 위험 엔진 선별을 핵심 과제로 수행

---

### 담당 파트

> 서정민 : 팀  장 | 깃 관리자 | 총괄

> 임재우 : 부팀장 | 데이터 담당

> 박은진 : 팀  원 | 해석·문서 담당 | Miro 관리

> 함수연 : 팀  원 | 시계열·특성 담당 | Notion 관리

> 김희락 : 팀  원 | 모델링 담당
 
---

### 주요 기능 요약

> - 센서 데이터 전처리 및 열화 추세 탐색
> - 핵심 센서 선정 및 Rolling Mean / Std 시계열 Feature 생성
> - 엔진 단위 Train / Validation 분리로 데이터 누수 방지
> - RandomForestRegressor 기반 RUL 예측 및 기준 모델 비교
> - RUL 10·20·30·50 기준의 고장 임박 탐지 성능 비교
> - Recall 중심의 최종 위험 기준 선정 및 Z-score 비교
> - 공식 Test 엔진별 현재 위험도 판정 및 정비 우선순위 산출

---

## 데이터셋

### 데이터 출처 | 파일명

> NASA C-MAPSS FD001 데이터셋
> - Data/Raw/train_fd001.csv : Train 데이터
> - Data/Raw/test_fd001.csv : Test 데이터
> - Data/Raw/rul_fd001.csv : Test 엔진 실제 RUL
> - Data/Processed/cleaned_data.csv : data 전처리 과정을 통해 생성된 데이터
> - Data/Processed/timeseries_features.csv : 시계열 분석 과정을 통해 생성된 데이터

### 데이터 개요

| 항목 | 내용 |
|---|---|
| Train | 20,631행, 100개 엔진 |
| Test | 13,096행, 100개 엔진|
| 센서 |sensor_measurement_1 ~ 21 |
| Target |RUL(Remaining Useful Life) |
| 특징 | 엔진별 여러 Cycle이 존재하는 시계열 데이터 |

### RUL 정의
> RUL = 엔진의 최대 Cycle - 현재 Cycle
> 공식 Test는 별도로 제공되는 rul_fd001.csv를 최종 정답으로 사용

### 전처리 및 Feature
> - 결측치·중복·이상치 확인 후 센서별 변동성과 Cycle에 따른 열화 추세를 분석
> - 주요 열화 후보 센서를 선정하고, 최근 상태와 변동성을 반영하기 위해 엔진별 Rolling Mean / Rolling Std를 추가

주요 후보 센서:
> 11, 12, 4, 7, 15, 20, 21

## 모델링
### 분석 방향
> - 정확한 RUL 자체보다 현재 점검이 필요한 엔진을 선별하는 것을 최종 목적으로 설정
> - RandomForestRegressor로 RUL을 예측한 뒤 예측 RUL에 위험 기준을 적용하여 고장 임박 여부를 판단

### 데이터 분할
> - 동일 엔진이 학습·검증 데이터에 동시에 포함되지 않도록 엔진 번호 기준 80:20 분할을 사용
> - GroupKFold로 모델 안정성을 추가 확인

### 사용 모델

| 모델 | 역할 |
|---|---|
| DummyRegressor | 기준 모델 |
| RandomForestRegressor | 메인 RUL 예측 모델 |
| Z-score | 단순 이상탐지 비교 모델 |

### 평가 지표
> - 회귀: MAE, RMSE, R²
> - 위험 탐지: Precision, Recall, F1, FP, FN

### 위험 기준 선정
> - Validation 데이터에서 RUL ≤ 10 / 20 / 30 / 50을 비교
> - 위험 엔진 누락을 줄이기 위해 Recall을 우선하고, Recall 0.9 이상인 후보 중 F1이 가장 높은 기준을 최종 위험 기준으로 선택
> - 위험 기준은 Validation에서만 결정하고 공식 Test에서는 변경하지 않음

### 공식 Test 및 정비 우선순위
> Train 전체 데이터로 최종 모델을 학습한 뒤 공식 Test 엔진의 마지막 관측 시점 RUL을 예측
> - PredRUL ≤ 위험 기준 → 점검 대상
> - PredRUL > 위험 기준 → 정상

> PredRUL이 작은 엔진부터 정렬하여 Maintenance_Rank를 부여하고 정비 우선순위를 산출
---

## 개발 환경

### 개발 언어 및 도구

| 구분 | 내용 |
|---|---|
| Language | Python 3.14 |
| IDE | VS Code, PyCharm |
| Interpreter | Python 3.14 |

### 실행 순서
> 전처리 → 시계열 분석 → Rolling Feature 생성 → RUL 모델링 → 위험 기준 선정 → Test 평가 및 정비 순위

### 실행 방법

> 1. Main.py 실행
> 2. train.csv 파일 선택
> 3. 이상치 처리 방법 선택(터미널): n(no)

### 의존 라이브러리 / 패키지
> pandas

> numpy

> scikit-learn

> matplotlib

> seaborn

> tkinter

> sys

> filedialog


## 분석 시 주의사항

> - Train / Validation은 반드시 엔진 단위로 분리

> - Rolling Feature도 엔진별로 계산

> - 위험 기준은 Validation에서만 선정

> - C-MAPSS는 시뮬레이션 데이터이므로 센서 중요도를 실제 물리적 고장 원인으로 단정하지 않음