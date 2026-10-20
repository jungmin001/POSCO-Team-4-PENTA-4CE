import json
import shutil

import pandas as pd
import sys
from datetime import datetime, timezone
from pathlib import Path

BaseDir = Path(__file__).resolve().parent
OutDir = BaseDir / "Results" / "Unreal"

# 언리얼이 읽는 출력 모음: 파트별 결과 파일을 Results/Unreal 한 곳에 복사하고 manifest.json에 무엇이 들어 있는지 기록
#   python Bundle.py
# 파일 형식은 Results/Unreal/SCHEMA.md 참고 (이 파일 목록이 곧 언리얼과 주고받는 약속)
# 원본이 없는 항목은 건너뛰고 manifest에 missing으로 표시 (예: 진단 모델 결과는 git에 없어 Diagnosis.py를 먼저 돌려야 함)
Sources = [
    {"Name": "UNREAL_PROMPT.md", "Part": "언리얼 작업 지시문 (정비 의사결정 솔루션)", "From": "jungmin/outputs/unreal/UNREAL_PROMPT.md", "Producer": "직접 작성"},
    {"Name": "ontology.json", "Part": "설비·부품·공정 구조", "From": "jungmin/outputs/unreal/ontology.json", "Producer": "jungmin/unreal.py"},
    {"Name": "costs.json", "Part": "정비 비용 가정값", "From": "jungmin/outputs/unreal/costs.json", "Producer": "jungmin/unreal.py"},
    {"Name": "anomaly_model.json", "Part": "이상탐지 모델 (C 연주)", "From": "model_output/anomaly_model.json", "Producer": "AnomalyModel.py"},
    {"Name": "flow_quality.csv", "Part": "최종 결함 위험 예측 (흐름 단위)", "From": "Results/Quality/flow_quality.csv", "Producer": "chanhee/Analysis.py"},
    {"Name": "quality_summary.json", "Part": "최종 결함 위험 모델 성능", "From": "Results/Quality/quality_summary.json", "Producer": "chanhee/Analysis.py"},
    {"Name": "diagnosis_metrics.csv", "Part": "상태·원인 진단 성능", "From": "Results/Diagnosis/metrics_summary.csv", "Producer": "Diagnosis.py"},
]

# 설비마다 파일이 하나씩: 고장 위험 예측과 기대 비용 확인표 (jungmin/unreal.py), 진단 예측 (Diagnosis.py, git에 없으므로 있으면 복사)
for Equip in ["A", "B", "C", "D", "E1", "E2", "F"]:
    Sources.append({"Name": f"risk_{Equip}.json", "Part": f"고장 위험 예측 (설비 {Equip})",
                    "From": f"jungmin/outputs/unreal/risk_{Equip}.json", "Producer": "jungmin/unreal.py"})
    Sources.append({"Name": f"decisions_{Equip}_check.csv", "Part": f"기대 비용 계산 확인표 (설비 {Equip})",
                    "From": f"jungmin/outputs/unreal/decisions_{Equip}_check.csv", "Producer": "jungmin/unreal.py"})
    Sources.append({"Name": f"diagnosis_{Equip}.csv", "Part": f"상태·원인 진단 예측 (설비 {Equip})",
                    "From": f"Results/Diagnosis/predictions/{Equip}_dashboard_validation.csv.gz", "Producer": "Diagnosis.py", "Convert": "diagnosis"})


# 진단 예측: 압축을 풀고 시각을 다른 파일과 같은 ISO 형식(UTC, 밀리초까지)으로 맞춰 일반 CSV로 저장
def ConvertDiagnosis(From, To):
    Df = pd.read_csv(From)
    Df["MEAS_DT"] = pd.to_datetime(Df["MEAS_DT"], utc=True, format="mixed").dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    Df.to_csv(To, index=False, encoding="utf-8")


def Main():
    OutDir.mkdir(parents=True, exist_ok=True)
    Items = []

    for Source in Sources:
        Path_ = BaseDir / Source["From"]
        Item = {"File": Source["Name"], "Part": Source["Part"], "Producer": Source["Producer"]}

        if Path_.exists():
            if Source.get("Convert") == "diagnosis":
                ConvertDiagnosis(Path_, OutDir / Source["Name"])
            else:
                shutil.copy2(Path_, OutDir / Source["Name"])
            Item.update(Status="ok", Bytes=(OutDir / Source["Name"]).stat().st_size,
                        SourceModified=datetime.fromtimestamp(Path_.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        else:
            Item.update(Status="missing", Hint=f"{Source['Producer']}를 먼저 실행")

        Items.append(Item)

    Manifest = {"Version": 1, "GeneratedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "Schema": "SCHEMA.md", "Files": Items}
    (OutDir / "manifest.json").write_text(json.dumps(Manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for Item in Items:
        print(f"    {Item['Status']:7s} {Item['File']}")
    print(f"저장 위치: {OutDir}")


if __name__ == "__main__":
    Main()
