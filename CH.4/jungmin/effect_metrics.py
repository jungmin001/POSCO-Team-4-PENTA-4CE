import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jungmin import config


# 경제적 효과 슬라이드용 숫자, CH.4 폴더에서 실행 (jungmin/unreal.py가 risk_*.json을 먼저 만들어 둬야 함)
#   python jungmin/effect_metrics.py
#
# 저장: jungmin/outputs/ppt/effect_metrics.json
#
# 비용 가정 없이 측정하는 것 (validation 시나리오 72개, 설비 7대)
#   - 실제 고장(정비 이력의 repair 기록) 중 고장 전에 경보가 한 번이라도 울린 비율
#   - 첫 경보부터 고장까지 걸린 시간
#   - 고장이 없는 설비·시나리오에서 경보가 울린 비율
# 가정 비용으로 환산하는 것 (costs.json 기본 가정값)
#   - 고장 난 뒤 대응 비용 F, 즉시 교체 비용 R, 건당 절감액 F - R
# 교체가 고장을 막는다는 보장은 데이터에 없으므로, 총 절감액은 계산하지 않고 건당 값만 낸다.
OutDir = config.OutDir / "unreal"
Tags = {"A": "P1-CR1-CAL01", "B": "P1-CR1-WLD01", "C": "P1-CC1-INS01", "D": "P1-CR1-SPM01",
        "E1": "P1-SP1-EXF01", "E2": "P1-SP2-EXF01", "F": "P1-SP1-AGT01"}


def Main():
    Index = pd.read_csv(config.ScenarioDir / "index.csv").set_index("scenario")["split"]
    Valid = [Id for Id, Split in Index.items() if Split == "validation"]

    Maint = pd.concat(pd.read_csv(File) for File in glob.glob(str(config.ScenarioDir / "S*" / "history" / "maintenance.csv")))
    Maint = Maint[(Maint["MNT_TYPE"] == "repair") & Maint["SCENARIO_ID"].isin(Valid)]
    Failure = {(Row.SCENARIO_ID, Row.EQP_TAG): Row.T_FAILURE_SEC for Row in Maint.itertuples()}

    Rows = []
    for Equip, Tag in Tags.items():
        Risk = json.loads((OutDir / f"risk_{Equip}.json").read_text(encoding="utf-8"))
        for Scenario in Risk["Scenarios"]:
            Id, Time = Scenario["Scenario"], Failure.get((Scenario["Scenario"], Tag))
            Points = [Point for Point in Scenario["Points"] if Time is None or Point["TSec"] < Time]
            Alarms = [Point for Point in Points if Point["Alarm24H"] or Point["Alarm6H"]]
            Lead = (Time - Alarms[0]["TSec"]) / 3600 if Time is not None and Alarms else None
            Rows.append({"equip": Equip, "scenario": Id, "failure": Time is not None, "alarm": bool(Alarms), "lead_h": Lead})

    Df = pd.DataFrame(Rows)
    Fail, Quiet = Df[Df["failure"]], Df[~Df["failure"]]
    Lead = Fail["lead_h"].dropna()

    Costs = json.loads((OutDir / "costs.json").read_text(encoding="utf-8"))
    Eq = Costs["Equipments"][0]
    Part = next(Item for Item in Eq["Parts"] if Item["Part"] == Eq["DefaultPart"])
    After = Eq["RepairCost"] + Eq["RepairHours"] * Eq["DowntimeCostPerHour"] + Eq["FailureLoss"]
    Replace = Part["Price"] + Eq["ReplaceHours"] * Eq["DowntimeCostPerHour"]

    Result = {
        "scenarios": len(Valid), "equipments": len(Tags), "failures": int(len(Fail)),
        "warned": int(Fail["alarm"].sum()), "warned_rate": float(Fail["alarm"].mean()),
        "lead_median_h": float(Lead.median()), "lead_mean_h": float(Lead.mean()), "lead_min_h": float(Lead.min()), "lead_max_h": float(Lead.max()),
        "lead_ge_6h": int((Lead >= 6).sum()), "lead_ge_24h": int((Lead >= 24).sum()),
        "quiet_pairs": int(len(Quiet)), "quiet_alarmed": int(Quiet["alarm"].sum()), "quiet_alarmed_rate": float(Quiet["alarm"].mean()),
        "cost_after_failure": float(After), "cost_replace_now": float(Replace), "saving_per_failure": float(After - Replace), "currency": Costs["Currency"],
        "by_equip": {Equip: {"failures": int(Group["failure"].sum()), "warned": int(Group.loc[Group["failure"], "alarm"].sum())} for Equip, Group in Df.groupby("equip")},
    }

    Path_ = config.OutDir / "ppt"
    Path_.mkdir(parents=True, exist_ok=True)
    (Path_ / "effect_metrics.json").write_text(json.dumps(Result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"고장 {Result['failures']}건 중 {Result['warned']}건({Result['warned_rate']:.0%})에서 고장 전에 경보")
    print(f"예고 시간 중앙값 {Result['lead_median_h']:.1f}시간, 6시간 이상 전 {Result['lead_ge_6h']}건, 24시간 이상 전 {Result['lead_ge_24h']}건")
    print(f"고장 없는 {Result['quiet_pairs']}쌍 중 경보 {Result['quiet_alarmed']}쌍({Result['quiet_alarmed_rate']:.1%})")
    print(f"가정 비용: 고장 후 {After:,.0f}만원, 즉시 교체 {Replace:,.0f}만원, 건당 절감 {After - Replace:,.0f}만원")


if __name__ == "__main__":
    Main()
