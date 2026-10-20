import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jungmin import config


# 조질압연기(D) 리포트 만들기, CH.4 폴더에서 실행 (jungmin/mill_d.py를 먼저 실행해야 함)
#   python jungmin/mill_report.py
#
# 저장 (jungmin/outputs/mill_D)
#   report.html          브라우저에서 바로 여는 단독 파일 (인터넷 연결 시 글꼴을 불러옴)
#   report_artifact.html 게시용 본문 (문서 틀 없이 내용만)
OutDir = config.OutDir / "mill_D"
StepHours = 5 / 60
NormalScenarios = 3                    # 고장이 없는 시나리오도 몇 개 넣어서 평상시와 비교
FamilyCode = {"충격형": 1, "마찰형": 2}
ModeLabel = {"bearing::bearing-damage": "베어링 손상", "roll::roll-wear": "롤 마모",
             "bearing::background-wear": "베어링 일반 마모", "roll::background-wear": "롤 일반 마모"}


def Pack(Values, Digits):
    return [None if pd.isna(Value) else round(float(Value), Digits) for Value in Values]


def BuildData():
    Metrics = json.loads((OutDir / "metrics.json").read_text(encoding="utf-8"))
    Extra = json.loads((OutDir / "report_data.json").read_text(encoding="utf-8"))
    Rows = pd.read_csv(OutDir / "validation_rows.csv", keep_default_na=False, na_values=[""])
    Risk = pd.read_csv(OutDir / "validation_risk.csv")
    Families = Extra["families"]

    Rows["Mode"] = Rows["Mode"].fillna("")
    Rows["family"] = Rows["family"].fillna("")
    Rows["TruthFamily"] = Rows["Mode"].map(Families).fillna("")

    Chosen, Normal = [], []
    for Id, Part in Rows.groupby("SCENARIO_ID"):
        (Chosen if (Part["Mode"] != "").any() else Normal).append(Id)
    Selected = sorted(Chosen) + sorted(Normal)[:NormalScenarios]

    Scenarios = []
    for Id in Selected:
        Part = Rows[Rows["SCENARIO_ID"] == Id].reset_index(drop=True)
        Hours = Part["T_SEC"].to_numpy() / 3600
        Alarm = Part["alarm"].to_numpy().astype(int)

        Episodes = []
        Flag = (Part["Mode"] != "").to_numpy()
        Edges = np.flatnonzero(np.diff(np.r_[0, Flag.astype(int), 0]))
        for Start, End in zip(Edges[::2], Edges[1::2]):
            Hits = np.flatnonzero(Alarm[Start:End])
            Mode = Part["Mode"].iloc[Start]
            Episodes.append({"s": round(float(Hours[Start]), 3), "e": round(float(Hours[End - 1] + StepHours), 3), "mode": Mode,
                             "label": ModeLabel.get(Mode, Mode), "fam": FamilyCode[Families[Mode]],
                             "delay": round(float(Hits[0] * StepHours), 3) if len(Hits) else None})

        Grid = Risk[Risk["SCENARIO_ID"] == Id].sort_values("T_SEC")
        Scenarios.append({
            "id": Id, "start": str(Part["MEAS_DT"].iloc[0]), "episodes": Episodes,
            "p": Pack(Part["p_fault"], 3), "alarm": Alarm.tolist(),
            "pred": [FamilyCode.get(Value, 0) for Value in Part["family"]],
            "truth": [FamilyCode.get(Value, 0) for Value in Part["TruthFamily"]],
            "cl": Pack(Part["cur_load"], 1), "ku": Pack(Part["VIB-TOP_KUR"], 2),
            "risk": [{"h": float(Row.T_SEC / 3600), "p6": None if pd.isna(Row.P6H) else round(float(Row.P6H), 3),
                      "p24": None if pd.isna(Row.P24H) else round(float(Row.P24H), 3),
                      "y6": None if pd.isna(Row.GT_RISK_6H) else int(Row.GT_RISK_6H),
                      "y24": None if pd.isna(Row.GT_RISK_24H) else int(Row.GT_RISK_24H)} for Row in Grid.itertuples()],
        })

    Known24, Known6 = Risk["GT_RISK_24H"].notna(), Risk["GT_RISK_6H"].notna()
    Metrics["risk"]["24H"]["baseline"] = float(Risk.loc[Known24, "GT_RISK_24H"].mean())
    Metrics["risk"]["6H"]["baseline"] = float(Risk.loc[Known6, "GT_RISK_6H"].mean())

    return {"metrics": Metrics, "fingerprint": Extra["fingerprint"], "normal": Extra["normal"], "order": Extra["order"],
            "labels": ModeLabel, "families": {Key: FamilyCode[Value] for Key, Value in Families.items()},
            "scenarios": Scenarios, "validationScenarios": int(Rows["SCENARIO_ID"].nunique()),
            "detThreshold": Metrics["detect"]["validation"]["threshold"],
            "p24Threshold": Metrics["risk"]["24H"]["validation"]["threshold"], "p6Threshold": Metrics["risk"]["6H"]["validation"]["threshold"]}


Template = Path(__file__).with_name("mill_report_template.html")


def Main():
    Data = BuildData()
    Body = Template.read_text(encoding="utf-8").replace("__DATA__", json.dumps(Data, ensure_ascii=False, separators=(",", ":")))

    (OutDir / "report_artifact.html").write_text(Body, encoding="utf-8")
    Wrapped = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
               '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
               "<style>:root{color-scheme:light;padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}"
               "body{margin:0;font:14px system-ui,sans-serif}[hidden]{display:none!important}</style>\n"
               "</head>\n<body>\n" + Body + "\n</body>\n</html>\n")
    (OutDir / "report.html").write_text(Wrapped, encoding="utf-8")

    print(f"시나리오 {len(Data['scenarios'])}개 포함 | report.html {len(Wrapped) / 1e6:.2f}MB")
    print(f"저장 위치: {OutDir}")


if __name__ == "__main__":
    Main()
