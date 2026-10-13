import json
from pathlib import Path

import pandas as pd

from . import posko

BaseDir = Path(__file__).resolve().parent
RawDir = BaseDir.parent / "Data" / "Raw"

# 설비 이름 -> (설비 ID, 원자료 id), D·E·F는 구간별 특징 파일을 읽음
Equipment = {
    "A": ("P1-CR1-CAL01", "annealing"),
    "B": ("P1-CR1-WLD01", "welding"),
    "C": ("P1-CC1-INS01", "casting"),
    "D": ("P1-CR1-SPM01", "temper-features"),
    "E1": ("P1-SP1-EXF01", "fan-features"),
    "E2": ("P1-SP2-EXF01", "fan-features"),
    "F": ("P1-SP1-AGT01", "agitator-features"),
}

# 숫자로 바꾸지 않는 컬럼 (식별자, 시각, 문자 코드)
TextCols = {"SCENARIO_ID", "MEAS_DT", "EQP_CD", "SLAB_NO", "LOT_NO", "COIL_ID",
            "BURST_ID", "SESS_ID", "GRADE", "QUALITY_FLAG"}


MaintenanceCols = ["SCENARIO_ID", "WO_NO", "EQP_TAG", "PART_CD", "DETC_DT", "FAILURE_DT",
                   "STRT_DT", "END_DT", "MNT_TYPE", "DOWN_MIN", "COST_SCORE"]


# manifest.json 로드
def LoadManifest(Root=RawDir):
    return json.loads((Path(Root) / "manifest.json").read_text())


# 시나리오별 분할 {"S0001": "train", ...}
def GetScenarioSplit(Root=RawDir):
    return {S["id"]: S["split"] for S in LoadManifest(Root)["scenarios"]}


# 분할(train / validation / test)에 속한 시나리오 ID 목록
def GetScenarios(Split, Root=RawDir):
    return [Id for Id, Name in GetScenarioSplit(Root).items() if Name == Split]


# 한 시나리오, 한 설비의 센서 표 (숫자·시간형 변환, 빈 값은 NaN)
def LoadSensor(ScenarioId, Equip, Root=RawDir):
    EqpCd, Source = Equipment[Equip]

    Df = pd.DataFrame(posko.source_rows(Root, ScenarioId, Source))
    Df = Df[Df["EQP_CD"] == EqpCd].drop(columns=["_SOURCE_CHUNK", "_SOURCE_ROW"])

    NumCols = [Col for Col in Df.columns if Col not in TextCols]
    Df[NumCols] = Df[NumCols].apply(pd.to_numeric, errors="coerce")
    Df["MEAS_DT"] = pd.to_datetime(Df["MEAS_DT"], utc=True)

    return Df.reset_index(drop=True)


# 한 시나리오의 정비 이력 (모든 설비), 정비가 없는 시나리오는 빈 표 반환
def LoadMaintenance(ScenarioId, Root=RawDir):
    Df = pd.DataFrame(posko.source_rows(Root, ScenarioId, "maintenance"), columns=MaintenanceCols + ["_SOURCE_CHUNK", "_SOURCE_ROW"])
    Df = Df.drop(columns=["_SOURCE_CHUNK", "_SOURCE_ROW"])

    for Col in ["DETC_DT", "FAILURE_DT", "STRT_DT", "END_DT"]:
        Df[Col] = pd.to_datetime(Df[Col], utc=True)

    return Df


# 정답 CSV 로드 (test 시나리오는 정답 파일이 없음), 설비를 주면 해당 설비 행만 반환
def LoadTruth(Kind, TimeCols, ScenarioId, Equip, Root):
    Path_ = Path(Root) / Kind / f"{ScenarioId}.csv"
    if not Path_.exists():
        raise FileNotFoundError(f"{Kind} 정답이 없습니다 (test 시나리오): {ScenarioId}")

    Df = pd.read_csv(Path_)

    if Equip is not None:
        Df = Df[Df["EQP_CD"] == Equipment[Equip][0]]

    for Col in TimeCols:
        Df[Col] = pd.to_datetime(Df[Col], utc=True)

    return Df.reset_index(drop=True)


# 상태·원인 구간 정답 (입력 특징으로 쓰지 않음)
def LoadDiagnostics(ScenarioId, Equip=None, Root=RawDir):
    return LoadTruth("diagnostics", ["START_AT", "END_AT"], ScenarioId, Equip, Root)


# 6·24시간 고장 위험 정답 (공란은 NaN으로 유지)
def LoadLabels(ScenarioId, Equip=None, Root=RawDir):
    return LoadTruth("labels", ["OBSERVED_AT"], ScenarioId, Equip, Root)


# 잔여수명·검열·정비 시점 정답 (RUL_HOURS 공란은 NaN으로 유지)
def LoadTargets(ScenarioId, Equip=None, Root=RawDir):
    return LoadTruth("targets", ["OBSERVED_AT"], ScenarioId, Equip, Root)
