# 상태·원인 진단 실행 안내

## 실행

CH.4 폴더에서 실행한다. 기존 Main.py는 전처리용으로 그대로 유지했다.

```powershell
python Diagnosis.py
```

입력이 바뀌지 않은 재실행:

```powershell
python Diagnosis.py --reuse-features
```

학습 없이 보고서만 다시 만들기:

```powershell
python Diagnosis.py --report-only
```

설비 C만 별도 결과 폴더에 실행:

```powershell
python Diagnosis.py --equipment C --output Results/Diagnosis_C
```

저장된 모델로 시나리오 하나를 추론(정답을 읽지 않음):

```powershell
python Diagnosis.py --predict-scenario S0289 --equipment C
```

검증:

```powershell
python -m unittest diagnosis.test_integrity
```

필요 라이브러리: numpy, pandas, scipy, scikit-learn, matplotlib, joblib. 사용한 버전은 requirements-diagnosis.txt 및 Results/Diagnosis/run_manifest.json에 기록했다. joblib 모델/캐시는 직접 만든 로컬 파일만 읽는다.

## 파일 역할

- Diagnosis.py: 전체 실행 진입점.
- diagnosis/features.py: 설비별 센서 허용 목록, 과거 관측 요약, 시간 구간 정답 연결.
- diagnosis/model.py: 모델 학습, 확률 보정, 판단 보류, 관측 설명, 시나리오 단위 부트스트랩.
- diagnosis/pipeline.py: 특징 캐시, 설비별 실행, 예측·표·그림·보고서 저장.
- diagnosis/test_integrity.py: 미래 정보 차단과 정답 시간 경계 검증.

## 학습·검증 설계

원본 train 216개 상황을 고정 seed로 학습 162개와 보정 54개로 나눈다. validation 72개는 최종 결과 평가에만 사용한다. test 72개는 정답 없는 예측만 저장한다. 모델 후보를 여러 번 바꾸며 같은 validation 성능을 최적화하면 그 자료도 더 이상 순수한 평가 자료가 아니라는 점에 유의한다.

입력은 센서/운전조건 허용 목록으로만 만든다. GT_ 칼럼, ID, 경과 시간, 정비 여부, 이상치 여부는 입력에서 제외한다. upstream의 IS_MISSING 표시 행에서는 센서 관측을 보수적으로 다시 마스킹한다. 이는 기존 결측 대체가 전체 결측 길이를 본 뒤 채울지 결정하는 방식에 모델이 의존하지 않기 위한 처리다. 정지 값은 보존한다.

상태: normal/degraded/failed/maintenance. 원인: 원인이 알려진 degraded/failed 시점의 부품::고장유형. 정상·정비 시점의 공란 원인은 별도 고장 클래스로 바꾸지 않는다. 원인 모델의 성능은 조건부이며, 별도로 상태 감지까지 포함한 end_to_end 결과를 저장한다.

A/C는 5분마다 진단하므로 5분 미만의 짧은 상태를 놓칠 수 있다. B는 같은 시각의 공간 관측 평균·표준편차를 쓴다. D/E/F는 파형 표본 취득 시간(N/FS_HZ) 뒤에 관측이 이용 가능하다고 가정한다. 과거 특징은 1시간 또는 6시간 창을 사용한다. 상태 전환 전 관측이 창에 남는 것은 실시간 진단에서 존재하는 현상이며 혼동 사례로 확인한다.

## 출력 읽기

먼저 Results/Diagnosis/분석결과_과정설명.md를 연다. 결과_한눈에보기.html에는 성능표와 혼동행렬이 있다. predictions의 gzip CSV는 pandas.read_csv로 바로 읽는다. 검증 CSV의 정답 칼럼은 화면 평가용이며 서비스 입력으로 사용하지 않는다.

간단한 결론부터 보려면 Results/Diagnosis/먼저읽기_진단결과.md를 연다. 전체 저장 결과 검증은 `python -m diagnosis.validate_results`로 실행한다. models/features/predictions는 용량 때문에 해당 결과 폴더의 .gitignore로 Git 추적에서 제외하지만 실제 파일은 로컬에 저장되어 있다.

시험 출력의 cause_probability_conditional은 실제 이상이라는 조건 아래의 후보 점수다. 상태 또는 원인이 needs_review이면 확정 진단 대신 점검 필요를 표시해야 한다. outside_training_envelope는 관측 범위 경고일 뿐 새 고장 검출 보장이 아니다. D/E/F는 측정 간격이 길어 마지막 관측 시각을 함께 표시해야 한다.

관측 설명은 정상 학습 참조값으로 바꿨을 때 모델 점수가 변하는 정도다. 부하를 완전히 일치시킨 인과 실험이 아니며 센서 조작이나 부품 교체의 효과를 보장하지 않는다. 학습에 없는 원인에 대한 open-set 분류기는 구현하지 않았다.
