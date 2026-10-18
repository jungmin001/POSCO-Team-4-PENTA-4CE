import json
from pathlib import Path

import pandas as pd

from . import config

BaseDir = Path(__file__).resolve().parent
RawDir = BaseDir.parent / "Data" / "Raw"
OutDir = BaseDir.parent.joinpath(*config.OutDirParts)


# manifest.json 로드
def LoadManifest(Root=RawDir):
    return json.loads((Path(Root) / "manifest.json").read_text(encoding="utf-8"))


# 시나리오별 분할 {"S0001": "train", ...}
def GetScenarioSplit(Root=RawDir, Manifest=None):
    Manifest = LoadManifest(Root) if Manifest is None else Manifest

    return {S["id"]: S["split"] for S in Manifest["scenarios"]}


# 원자료 종류 하나를 가진 시나리오 ID 목록 (manifest 순서)
def GetSourceScenarios(Source, Root=RawDir, Manifest=None):
    Manifest = LoadManifest(Root) if Manifest is None else Manifest

    return [F["scenarioId"] for F in Manifest["files"] if F["id"] == Source]


# 시나리오 하나의 청크를 모두 열어 순서대로 이어 붙인 표 (청크는 해시 이름의 CSV)
def ReadScenario(Source, ScenarioId, Root=RawDir, Manifest=None, AsText=False):
    Manifest = LoadManifest(Root) if Manifest is None else Manifest
    File = next(F for F in Manifest["files"] if F["id"] == Source and F["scenarioId"] == ScenarioId)

    # 행이 없는 원자료(예: 정비가 없는 시나리오)는 청크가 0개
    if not File["chunks"]:
        return pd.DataFrame(columns=File["columns"])

    # AsText: 값을 숫자로 바꾸지 않고 원본 글자 그대로 읽음 (빈칸은 빈 문자열)
    Options = {"dtype": str, "keep_default_na": False} if AsText else {}
    Parts = [pd.read_csv(Path(Root) / "chunks" / Chunk["sha256"], encoding="utf-8-sig", **Options) for Chunk in File["chunks"]]

    return pd.concat(Parts, ignore_index=True)


# 정답 CSV 로드 (분석 확인용, 입력 특징으로 쓰지 않음), test 시나리오는 정답 파일이 없음
def LoadTruth(Kind, ScenarioId, EqpCode=None, Root=RawDir):
    Path_ = Path(Root) / Kind / f"{ScenarioId}.csv"
    if not Path_.exists():
        raise FileNotFoundError(f"{Kind} 정답이 없습니다 (test 시나리오): {ScenarioId}")

    Df = pd.read_csv(Path_)

    if EqpCode is not None:
        Df = Df[Df[config.EqpCol] == EqpCode]

    for Col in config.TruthTimeCols[Kind]:
        Df[Col] = pd.to_datetime(Df[Col], utc=True)

    return Df.reset_index(drop=True)
