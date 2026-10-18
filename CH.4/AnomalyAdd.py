from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# 경로
# =========================================================

BaseDir = Path(__file__).resolve().parent

ScenarioDir = BaseDir / "Data" / "Processed" / "Scenarios"


# =========================================================
# Scenario
# =========================================================

StartScenario = 1
EndScenario = 360


# =========================================================
# 연속주조 설비
# =========================================================

CastingEqp = "P1-CC1-INS01"
CastingStage = "C"


# =========================================================
# 앞선 이상탐지 결과
#
# 97% Threshold + 2분 지속 기준
# =========================================================

FaultScenarioCount = 81


# 3분에서는 놓쳤지만
# 2분에서는 잡힌 Scenario 26개
RecoveredBy2Min = {
    "S0002",
    "S0004",
    "S0013",
    "S0016",
    "S0024",
    "S0034",
    "S0040",
    "S0051",
    "S0123",
    "S0124",
    "S0128",
    "S0129",
    "S0139",
    "S0152",
    "S0155",
    "S0164",
    "S0168",
    "S0174",
    "S0199",
    "S0215",
    "S0233",
    "S0235",
    "S0260",
    "S0277",
    "S0284",
    "S0286",
}


# 97% + 2분에서도 최종적으로 놓친 4개
Missed2Min = {"S0011", "S0193", "S0209", "S0237"}


# =========================================================
# quality.csv 찾기
# =========================================================


def GetQualityFiles():

    Files = []

    for Number in range(StartScenario, EndScenario + 1):

        ScenarioId = f"S{Number:04d}"

        File = ScenarioDir / ScenarioId / "flow" / "quality.csv"

        if File.exists():

            Files.append((ScenarioId, File))

    print("=" * 100)

    print("Quality 파일 탐색")

    print("=" * 100)

    print("찾은 quality.csv:", len(Files))

    if Files:

        print("첫 Scenario:", Files[0][0])

        print("마지막 Scenario:", Files[-1][0])

    return Files


# =========================================================
# 숫자 변환
# =========================================================


def Numeric(Series):

    return pd.to_numeric(Series, errors="coerce")


# =========================================================
# Scenario 하나의 품질 분석
# =========================================================


def AnalyzeQuality(ScenarioId, File):

    Df = pd.read_csv(File, low_memory=False)

    Required = [
        "FLOW_ID",
        "STAGE",
        "EQP_CD",
        "GT_STAGE_COMPLETED",
        "GT_DEFECT_FLAG",
        "GT_ORIGIN_EQP_CD",
        "GT_FINAL_DEFECT_FLAG",
        "GT_FINAL_CENSORED",
    ]

    Missing = [Col for Col in Required if Col not in Df.columns]

    if Missing:

        print(ScenarioId, "필수 컬럼 없음:", Missing)

        return None

    # =====================================================
    # 숫자 컬럼 정리
    # =====================================================

    for Col in [
        "GT_STAGE_COMPLETED",
        "GT_DEFECT_FLAG",
        "GT_FINAL_DEFECT_FLAG",
        "GT_FINAL_CENSORED",
    ]:

        Df[Col] = Numeric(Df[Col])

    # =====================================================
    # FLOW 단위로 정리
    #
    # quality.csv에는
    # 한 FLOW가 여러 STAGE로 존재함
    # =====================================================

    FlowIds = Df["FLOW_ID"].dropna().unique()

    TotalFlows = len(FlowIds)

    # =====================================================
    # 1. 공정 중 어느 단계에서든 불량
    # =====================================================

    StageDefectFlowCount = Df.loc[Df["GT_DEFECT_FLAG"] == 1, "FLOW_ID"].nunique()

    # =====================================================
    # 2. 연속주조 C 단계의 불량
    # =====================================================

    CastingStageDf = Df[Df["STAGE"].astype(str).str.upper().eq(CastingStage)]

    CastingDefectFlowCount = CastingStageDf.loc[
        CastingStageDf["GT_DEFECT_FLAG"] == 1, "FLOW_ID"
    ].nunique()

    # =====================================================
    # 3. 최종 불량의 원인 설비가
    #    연속주조인가?
    # =====================================================

    OriginCastingFlowCount = Df.loc[
        Df["GT_ORIGIN_EQP_CD"].astype(str).eq(CastingEqp), "FLOW_ID"
    ].nunique()

    # =====================================================
    # 4. 최종 품질
    #
    # 예시 quality.csv를 보면
    #
    # GT_FINAL_CENSORED == 0
    # → FINAL_DEFECT_FLAG 존재
    #
    # GT_FINAL_CENSORED == 1
    # → FINAL_DEFECT_FLAG NaN
    #
    # 따라서 CENSORED=0인 FLOW만
    # 최종 품질 판정에 사용
    # =====================================================

    FinalRows = Df[Df["GT_FINAL_CENSORED"].notna()].copy()

    # 한 FLOW당 최종값 하나만 사용
    FinalByFlow = FinalRows.sort_values("FLOW_ID").drop_duplicates(
        subset=["FLOW_ID"], keep="last"
    )

    ObservedFinal = FinalByFlow[
        (FinalByFlow["GT_FINAL_CENSORED"] == 0)
        & FinalByFlow["GT_FINAL_DEFECT_FLAG"].notna()
    ]

    CensoredFinal = FinalByFlow[FinalByFlow["GT_FINAL_CENSORED"] == 1]

    ObservedFlowCount = len(ObservedFinal)

    CensoredFlowCount = len(CensoredFinal)

    FinalDefectFlows = int((ObservedFinal["GT_FINAL_DEFECT_FLAG"] == 1).sum())

    FinalNormalFlows = int((ObservedFinal["GT_FINAL_DEFECT_FLAG"] == 0).sum())

    # =====================================================
    # 최종 불량률
    # =====================================================

    FinalDefectRate = (
        FinalDefectFlows / ObservedFlowCount * 100 if ObservedFlowCount else np.nan
    )

    # =====================================================
    # 품질 판정 가능 비율
    # =====================================================

    QualityCoverage = ObservedFlowCount / TotalFlows * 100 if TotalFlows else np.nan

    # =====================================================
    # Scenario 단위 품질 분류
    # =====================================================

    if FinalDefectFlows > 0:

        QualityClass = "최종 불량 발생"

    elif CastingDefectFlowCount > 0 or OriginCastingFlowCount > 0:

        QualityClass = "연속주조 이상 있으나 최종 불량 없음"

    elif ObservedFlowCount > 0:

        QualityClass = "관측된 최종 불량 없음"

    else:

        QualityClass = "최종 품질 판정 불가"

    # =====================================================
    # 이상탐지 결과 분류
    # =====================================================

    if ScenarioId in Missed2Min:

        AlarmResult = "97%-2분 미탐"

    elif ScenarioId in RecoveredBy2Min:

        AlarmResult = "2분 기준 추가 탐지"

    else:

        AlarmResult = "기타 Scenario"

    return {
        "SCENARIO_ID": ScenarioId,
        "ALARM_RESULT": AlarmResult,
        # ---------------------------------------------
        # 전체 FLOW
        # ---------------------------------------------
        "TOTAL_FLOWS": TotalFlows,
        # ---------------------------------------------
        # 공정 중 불량
        # ---------------------------------------------
        "ANY_STAGE_DEFECT_FLOWS": StageDefectFlowCount,
        # ---------------------------------------------
        # 연속주조 C 단계 불량
        # ---------------------------------------------
        "CASTING_DEFECT_FLOWS": CastingDefectFlowCount,
        # ---------------------------------------------
        # 연속주조가 원인인 FLOW
        # ---------------------------------------------
        "CASTING_ORIGIN_FLOWS": OriginCastingFlowCount,
        # ---------------------------------------------
        # 최종 품질
        # ---------------------------------------------
        "FINAL_OBSERVED_FLOWS": ObservedFlowCount,
        "FINAL_CENSORED_FLOWS": CensoredFlowCount,
        "FINAL_NORMAL_FLOWS": FinalNormalFlows,
        "FINAL_DEFECT_FLOWS": FinalDefectFlows,
        "FINAL_DEFECT_RATE_%": FinalDefectRate,
        "QUALITY_COVERAGE_%": QualityCoverage,
        # ---------------------------------------------
        # Boolean
        # ---------------------------------------------
        "HAS_FINAL_DEFECT": int(FinalDefectFlows > 0),
        "HAS_CASTING_DEFECT": int(CastingDefectFlowCount > 0),
        "HAS_CASTING_ORIGIN": int(OriginCastingFlowCount > 0),
        # ---------------------------------------------
        # 최종 설명
        # ---------------------------------------------
        "QUALITY_CLASS": QualityClass,
    }


# =========================================================
# 전체 Scenario 품질 분석
# =========================================================


def BuildQualitySummary(Files):

    Rows = []

    for Index, (ScenarioId, File) in enumerate(Files, 1):

        Result = AnalyzeQuality(ScenarioId, File)

        if Result is not None:

            Rows.append(Result)

        if Index % 30 == 0 or Index == len(Files):

            print(f"품질 분석: " f"{Index}/{len(Files)}")

    return pd.DataFrame(Rows)


# =========================================================
# 미탐 4개 상세 분석
# =========================================================


def AnalyzeMissedQuality(Summary):

    Missed = Summary[Summary["SCENARIO_ID"].isin(Missed2Min)].sort_values("SCENARIO_ID")

    print("\n" + "=" * 130)

    print("97% + 2분 미탐 4건의 품질 결과")

    print("=" * 130)

    Display = [
        "SCENARIO_ID",
        "TOTAL_FLOWS",
        "ANY_STAGE_DEFECT_FLOWS",
        "CASTING_DEFECT_FLOWS",
        "CASTING_ORIGIN_FLOWS",
        "FINAL_OBSERVED_FLOWS",
        "FINAL_CENSORED_FLOWS",
        "FINAL_DEFECT_FLOWS",
        "FINAL_DEFECT_RATE_%",
        "QUALITY_CLASS",
    ]

    print(Missed[Display].round({"FINAL_DEFECT_RATE_%": 2}).to_string(index=False))

    # =====================================================
    # 미탐 중 최종 품질 불량 Scenario
    # =====================================================

    MissedFinalDefect = Missed["HAS_FINAL_DEFECT"].sum()

    MissedCastingDefect = Missed["HAS_CASTING_DEFECT"].sum()

    MissedCastingOrigin = Missed["HAS_CASTING_ORIGIN"].sum()

    print("\n미탐 Scenario:", len(Missed))

    print("미탐 중 최종 불량 Scenario:", MissedFinalDefect)

    print("미탐 중 연속주조 C 단계 불량 Scenario:", MissedCastingDefect)

    print("미탐 중 연속주조가 원인인 Scenario:", MissedCastingOrigin)

    print("\n고장 Event 기준 미탐률:")

    print(
        f"{len(Missed)} / "
        f"{FaultScenarioCount}"
        f" = "
        f"{len(Missed) / FaultScenarioCount * 100:.2f}%"
    )

    if len(Missed):

        print("\n미탐 4건 중 최종 품질 불량 비율:")

        print(
            f"{MissedFinalDefect} / "
            f"{len(Missed)}"
            f" = "
            f"{MissedFinalDefect / len(Missed) * 100:.2f}%"
        )

    return Missed


# =========================================================
# 2분에서 새로 잡힌 26개와
# 남은 미탐 4개 비교
# =========================================================


def CompareRecoveredVsMissed(Summary):

    Compare = Summary[Summary["SCENARIO_ID"].isin(RecoveredBy2Min | Missed2Min)].copy()

    Compare["GROUP"] = np.where(
        Compare["SCENARIO_ID"].isin(Missed2Min), "2분에서도 미탐", "2분에서 추가 탐지"
    )

    GroupSummary = (
        Compare.groupby("GROUP")
        .agg(
            SCENARIOS=("SCENARIO_ID", "nunique"),
            FINAL_DEFECT_SCENARIOS=("HAS_FINAL_DEFECT", "sum"),
            CASTING_DEFECT_SCENARIOS=("HAS_CASTING_DEFECT", "sum"),
            CASTING_ORIGIN_SCENARIOS=("HAS_CASTING_ORIGIN", "sum"),
            TOTAL_FINAL_OBSERVED=("FINAL_OBSERVED_FLOWS", "sum"),
            TOTAL_FINAL_DEFECT=("FINAL_DEFECT_FLOWS", "sum"),
        )
        .reset_index()
    )

    GroupSummary["SCENARIO_FINAL_DEFECT_RATE_%"] = (
        GroupSummary["FINAL_DEFECT_SCENARIOS"] / GroupSummary["SCENARIOS"] * 100
    )

    GroupSummary["FLOW_FINAL_DEFECT_RATE_%"] = (
        GroupSummary["TOTAL_FINAL_DEFECT"] / GroupSummary["TOTAL_FINAL_OBSERVED"] * 100
    )

    print("\n" + "=" * 130)

    print("2분 추가 탐지 vs 2분 미탐 품질 비교")

    print("=" * 130)

    print(GroupSummary.round(2).to_string(index=False))

    return GroupSummary


# =========================================================
# 전체 품질 분포
# =========================================================


def PrintOverallQuality(Summary):

    print("\n" + "=" * 130)

    print("전체 Scenario 품질 상태")

    print("=" * 130)

    print("\nQUALITY_CLASS")

    print(Summary["QUALITY_CLASS"].value_counts().to_string())

    TotalObserved = Summary["FINAL_OBSERVED_FLOWS"].sum()

    TotalDefect = Summary["FINAL_DEFECT_FLOWS"].sum()

    OverallFinalDefectRate = (
        TotalDefect / TotalObserved * 100 if TotalObserved else np.nan
    )

    print("\n전체 최종 품질 판정 FLOW:", f"{TotalObserved:,}")

    print("전체 최종 불량 FLOW:", f"{TotalDefect:,}")

    print("전체 최종 불량률:", f"{OverallFinalDefectRate:.2f}%")


# =========================================================
# 미탐 품질 리스크 해석
# =========================================================


def PrintDecisionGuide(Missed):

    FinalDefectMiss = int(Missed["HAS_FINAL_DEFECT"].sum())

    CastingDefectMiss = int(Missed["HAS_CASTING_DEFECT"].sum())

    print("\n" + "=" * 130)

    print("의사결정 해석")

    print("=" * 130)

    print("\n97% + 2분 기준")

    print("- 실제 이상 81건 중 77건 탐지")

    print("- 미탐 4건")

    print("- 미탐률 4.94%")

    print(f"- 미탐 중 최종 불량 발생 Scenario: " f"{FinalDefectMiss}건")

    print(f"- 미탐 중 연속주조 C 단계 불량 Scenario: " f"{CastingDefectMiss}건")

    if FinalDefectMiss == 0:

        print("\n해석:")

        print("미탐 4건에서 관측된 최종 품질 불량이 없었습니다.")

        print("따라서 4건의 설비 이상을 놓쳤지만,")

        print("제품 품질 손실까지 이어진 사례는 확인되지 않았습니다.")

        print("이 경우 2분 경보 기준을 수용할 근거가 강해집니다.")

    elif FinalDefectMiss < len(Missed):

        print("\n해석:")

        print("미탐 4건이 모두 품질 불량으로 이어진 것은 아닙니다.")

        print("따라서 단순 미탐 건수보다")

        print("실제 품질 영향을 함께 보는 것이 중요합니다.")

    else:

        print("\n해석:")

        print("미탐 Scenario가 모두 최종 품질 불량과 연결됩니다.")

        print("이 경우 2분 기준의 비용 절감보다")

        print("미탐 품질비용을 더 크게 고려해야 합니다.")


# =========================================================
# Main
# =========================================================


def Main():

    # =====================================================
    # 1. quality.csv 찾기
    # =====================================================

    Files = GetQualityFiles()

    if not Files:

        raise ValueError("quality.csv를 찾지 못했습니다.")

    # =====================================================
    # 2. 전체 품질 분석
    # =====================================================

    Summary = BuildQualitySummary(Files)

    # =====================================================
    # 3. 전체 품질 분포
    # =====================================================

    PrintOverallQuality(Summary)

    # =====================================================
    # 4. 미탐 4건 품질
    # =====================================================

    Missed = AnalyzeMissedQuality(Summary)

    # =====================================================
    # 5. 2분 추가 탐지 vs 미탐
    # =====================================================

    CompareRecoveredVsMissed(Summary)

    # =====================================================
    # 6. 의사결정 해석
    # =====================================================

    PrintDecisionGuide(Missed)


if __name__ == "__main__":

    Main()

from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# 경로
# =========================================================

BaseDir = Path(__file__).resolve().parent

ScenarioDir = BaseDir / "Data" / "Processed" / "Scenarios"


CastingEqp = "P1-CC1-INS01"

MissedScenarios = {"S0011", "S0193", "S0209", "S0237"}


# =========================================================
# Scenario 하나 분석
# =========================================================


def AnalyzeScenario(ScenarioId):

    QualityFile = ScenarioDir / ScenarioId / "flow" / "quality.csv"

    CastingFile = ScenarioDir / ScenarioId / "C.csv"

    if not QualityFile.exists():

        return None

    Q = pd.read_csv(QualityFile, low_memory=False)

    # -----------------------------------------------------
    # 숫자 변환
    # -----------------------------------------------------

    for Col in ["GT_FINAL_DEFECT_FLAG", "GT_FINAL_CENSORED"]:

        Q[Col] = pd.to_numeric(Q[Col], errors="coerce")

    # =====================================================
    # FLOW 단위로 변환
    # =====================================================

    FlowRows = []

    for FlowId, G in Q.groupby("FLOW_ID"):

        FinalObserved = (
            (G["GT_FINAL_CENSORED"] == 0) & G["GT_FINAL_DEFECT_FLAG"].notna()
        ).any()

        FinalDefect = G["GT_FINAL_DEFECT_FLAG"].eq(1).any()

        CastingOrigin = G["GT_ORIGIN_EQP_CD"].astype(str).eq(CastingEqp).any()

        CastingStageDefect = (
            (G["STAGE"].astype(str).str.upper() == "C")
            & (pd.to_numeric(G["GT_DEFECT_FLAG"], errors="coerce") == 1)
        ).any()

        FlowRows.append(
            {
                "FLOW_ID": FlowId,
                "FINAL_OBSERVED": int(FinalObserved),
                "FINAL_DEFECT": int(FinalDefect),
                "CASTING_ORIGIN": int(CastingOrigin),
                "CASTING_STAGE_DEFECT": int(CastingStageDefect),
            }
        )

    Flow = pd.DataFrame(FlowRows)

    # =====================================================
    # 최종 판정 가능한 FLOW
    # =====================================================

    Observed = Flow[Flow["FINAL_OBSERVED"] == 1].copy()

    # =====================================================
    # ★ 핵심
    #
    # 최종 불량이면서
    # 연주 설비가 원인인 FLOW
    # =====================================================

    CastingFinalLoss = Observed[
        (Observed["FINAL_DEFECT"] == 1) & (Observed["CASTING_ORIGIN"] == 1)
    ]

    FinalDefects = Observed[Observed["FINAL_DEFECT"] == 1]

    GoodFlows = Observed[Observed["FINAL_DEFECT"] == 0]

    # =====================================================
    # C.csv의 SLAB 개수도 확인
    # =====================================================

    SlabCount = np.nan

    if CastingFile.exists():

        C = pd.read_csv(CastingFile, usecols=["SLAB_NO"], low_memory=False)

        SlabCount = C["SLAB_NO"].nunique()

    return {
        "SCENARIO_ID": ScenarioId,
        "TOTAL_FLOW": Flow["FLOW_ID"].nunique(),
        "SLAB_COUNT": SlabCount,
        "FINAL_OBSERVED": len(Observed),
        "FINAL_GOOD": len(GoodFlows),
        "FINAL_DEFECT": len(FinalDefects),
        "CASTING_FINAL_LOSS": len(CastingFinalLoss),
    }


# =========================================================
# 전체 Scenario
# =========================================================

Rows = []


for Number in range(1, 361):

    ScenarioId = f"S{Number:04d}"

    Result = AnalyzeScenario(ScenarioId)

    if Result is not None:

        Rows.append(Result)


Summary = pd.DataFrame(Rows)


# =========================================================
# 전체 품질
# =========================================================

TotalObserved = Summary["FINAL_OBSERVED"].sum()


TotalGood = Summary["FINAL_GOOD"].sum()


TotalDefect = Summary["FINAL_DEFECT"].sum()


CastingLoss = Summary["CASTING_FINAL_LOSS"].sum()


CastingLossRate = CastingLoss / TotalObserved * 100


CurrentYield = TotalGood / TotalObserved * 100


YieldWithoutCastingLoss = (TotalGood + CastingLoss) / TotalObserved * 100


# =========================================================
# 2분 미탐 Scenario
# =========================================================

Missed = Summary[Summary["SCENARIO_ID"].isin(MissedScenarios)]


MissObserved = Missed["FINAL_OBSERVED"].sum()


MissGood = Missed["FINAL_GOOD"].sum()


MissCastingLoss = Missed["CASTING_FINAL_LOSS"].sum()


MissLossRate = MissCastingLoss / MissObserved * 100


# =========================================================
# 출력
# =========================================================

print("\n" + "=" * 100)

print("전체 최종 생산량 영향")

print("=" * 100)


print("최종 품질 판정 FLOW:", f"{TotalObserved:,}")


print("정상 최종 FLOW:", f"{TotalGood:,}")


print("전체 최종 불량 FLOW:", f"{TotalDefect:,}")


print("연속주조 원인 최종 불량:", f"{CastingLoss:,}")


print(f"연속주조 원인 생산손실 비율: " f"{CastingLossRate:.3f}%")


print(f"현재 정상 생산수율: " f"{CurrentYield:.3f}%")


print(f"연주 원인 불량을 모두 제거할 경우: " f"{YieldWithoutCastingLoss:.3f}%")


print(f"수율 개선 가능폭: " f"{YieldWithoutCastingLoss - CurrentYield:.3f}%p")


print("\n" + "=" * 100)

print("97% + 2분 미탐 4건의 생산량 영향")

print("=" * 100)


print("최종 판정 FLOW:", MissObserved)


print("정상 FLOW:", MissGood)


print("연속주조 원인 최종 불량:", MissCastingLoss)


print(f"미탐 구간 생산손실 비율: " f"{MissLossRate:.2f}%")


print("\n" + "=" * 100)

print("FLOW ↔ SLAB 개수 확인")

print("=" * 100)


Check = Summary[["SCENARIO_ID", "TOTAL_FLOW", "SLAB_COUNT"]].copy()


Check["COUNT_MATCH"] = Check["TOTAL_FLOW"] == Check["SLAB_COUNT"]


print(Check.head(30).to_string(index=False))


print("\nFLOW 수 = SLAB 수 Scenario:", Check["COUNT_MATCH"].sum(), "/", len(Check))
