# PENTA-4CE — CH_1

---

### 챕터 개요

> 팀 프로젝트를 위한 각 파트별 학습

---

### 담당 파트

> 서정민 : 팀  장 | 깃 관리자 | 총괄

> 임재우 : 부팀장 | 데이터 담당

> 박은진 : 팀  원 | 해석·문서 담당 | Miro 관리

> 함수연 : 팀  원 | 시계열·특성 담당 | Notion 관리

> 김희락 : 팀  원 | 모델링 담당
 
---

### 주요 기능 요약

> - 원본 센서 데이터 전처리 (결측치·중복 제거, 시간 기반 보간)
> - IQR 기반 이상치 탐지 및 고장 직전(사전징후) 구간과의 상관관계 분석
> - IsolationForest 기반 이상탐지 모델로 펌프 고장 사전 경보 (진행중)

---

## 데이터셋

### 데이터 출처 | 파일명

> `CH.1/Data/Row/sensor.csv` (원본), 전처리 결과는 `CH.1/Data/Preprocessing_Data_File/`에 저장

### 데이터 개요

| 항목 | 내용 |
|---|---|
| 데이터 크기 | (220,320행, 55열) |
| 결측치 비율 | sensor_15 100%, sensor_50 약 34.96% (30% 이상은 이 2개 센서뿐) |
| 중복 데이터 여부 | 없음 |
| 기타 특이사항 | `machine_status` 라벨 극단적 불균형 — NORMAL 205,836 / RECOVERING 14,477 / **BROKEN 7건(0.003%)** |

### 전처리 방법

> 결측치 30% 이상인 센서(sensor_15, sensor_50) 제거 → 시간 기반 보간(interpolate)으로 내부 결측 채움 → 남은 가장자리 결측은 앞/뒤 값으로 채움 → 완전 중복 행 제거

---

## 개발 환경

### 개발 언어 및 도구

| 구분 | 내용 |
|---|---|
| Language | Python 3.14 |
| IDE | VS Code, PyCharm |
| Interpreter | Python 3.14 |

### 실행 방법

> 1. `1_preprocessing.py` → `2_timeseries_check.py` → '3_modeling.py' → '4_evaluation_report.py'

### 의존 라이브러리 / 패키지

```text
pandas
numpy
scikit-learn
matplotlib