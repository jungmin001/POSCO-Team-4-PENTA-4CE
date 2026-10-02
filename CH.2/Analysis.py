import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BaseDir = os.path.dirname(os.path.abspath(__file__))
CleanedPath = os.path.join(BaseDir, "Data", "Processed", "cleaned_data.csv")

# rolling window (단기 5 + 장기 30, 교차검증으로 선정)
RollingWindows = (5, 30)


# CSV 파일 로드 및 Df 변환하여 반환 (rolling 등 순서에 의존하는 계산이 많아서 미리 정렬)
def LoadData(FileLocation):
    Df = pd.read_csv(FileLocation)
    Df = Df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)

    return Df


# 엔진별 총 수명(사이클 수) 분포 확인
def CheckLifeDistribution(Df):
    LifeCycles = Df.groupby("unit_number")["time_in_cycles"].max()
    print(LifeCycles.describe())

    plt.figure(figsize=(8, 5))
    sns.histplot(LifeCycles, bins=20, color=sns.color_palette()[0])
    plt.title("엔진별 수명(사이클 수) 분포")
    plt.xlabel("수명(사이클)")
    plt.ylabel("엔진 수")
    plt.tight_layout()
    plt.show()

    return LifeCycles


# 시각화 비교용 RUL 계산 (엔진별 최대 사이클 - 현재 사이클)
def ComputeRUL(Df):
    MaxCycle = Df.groupby("unit_number")["time_in_cycles"].transform("max")
    Rul = MaxCycle - Df["time_in_cycles"]

    return Rul


# 센서끼리, 센서와 RUL 사이의 상관관계 히트맵 (센서 선정용이 아니라 다중공선성 확인용 - 핵심 센서끼리
# 상관관계가 너무 높으면 사실상 같은 정보를 중복으로 갖고 있다는 뜻이라 모델링 단계에 참고가 됨)
def PlotCorrelationHeatmap(Df):
    Df = Df.copy()
    Df["RUL"] = ComputeRUL(Df)
    SensorCols = [Col for Col in Df.columns if Col.startswith("sensor_measurement_")]

    Corr = Df[SensorCols + ["RUL"]].corr(method="spearman")

    plt.figure(figsize=(14, 11))
    sns.heatmap(Corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("센서 + RUL 상관관계 (Spearman, 전체 데이터 기준)")
    plt.tight_layout()
    plt.show()

    return Corr


# 센서별 Cycle 추세 강도 확인 (Spearman 상관계수 - 값의 스케일과 무관하게 얼마나 일관되게 증가/감소하는지 측정)
# +1에 가까움: Cycle이 증가할수록 센서도 꾸준히 증가 / -1에 가까움: 꾸준히 감소 / 0에 가까움: 뚜렷한 방향성 없음
def CheckSensorTrend(Df):
    SensorCols = [Col for Col in Df.columns if Col.startswith("sensor_measurement_")]
    Report = []

    for Col in SensorCols:
        EarlyValues = []
        UnitCorrelations = []

        for _, Group in Df.groupby("unit_number"):
            Group = Group.sort_values("time_in_cycles")
            SplitPoint = len(Group) // 3
            EarlyValues.append(Group[Col].iloc[:SplitPoint])

            Correlation = Group["time_in_cycles"].corr(Group[Col], method="spearman")
            if pd.notna(Correlation):
                UnitCorrelations.append(Correlation)

        EarlyValues = pd.concat(EarlyValues)
        MeanCorrelation = (
            sum(UnitCorrelations) / len(UnitCorrelations) if UnitCorrelations else 0
        )

        Report.append(
            {
                "sensor": Col,
                "mean_correlation": MeanCorrelation,
                "trend_strength": abs(MeanCorrelation),
                "normal_low": EarlyValues.quantile(0.05),
                "normal_high": EarlyValues.quantile(0.95),
            }
        )

    TrendDf = pd.DataFrame(Report).sort_values("trend_strength", ascending=False)

    return TrendDf


# 추세 강도 순위 막대그래프 시각화
def PlotSensorTrend(TrendDf):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=TrendDf, x="trend_strength", y="sensor", color=sns.color_palette()[0]
    )
    plt.title("센서별 Cycle 추세 강도 (Spearman 상관계수 절댓값)")
    plt.xlabel("추세 강도")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


# 패턴 확인된 핵심 열화 지표 센서 목록 확정 (추세 강도가 기준값 이상인 센서만 선택)
def SelectKeySensors(TrendDf, Threshold=0.7):
    KeySensors = TrendDf[TrendDf["trend_strength"] >= Threshold]["sensor"].tolist()
    print(f"핵심 센서(추세 강도 {Threshold} 이상) {len(KeySensors)}개:", KeySensors)

    return KeySensors


# 센서별로 서브플롯 하나씩, 그 안에 엔진 여러 개를 겹쳐서 그래프 (창 하나로 전부 보기)
def PlotTopSensorsByUnit(Df, TrendDf, TopSensors, SampleUnits=5):
    Df = Df.copy()
    Df["RUL"] = ComputeRUL(Df)

    NormDf = Df.copy()
    NormalRanges = {}
    for Col in TopSensors:
        Min = Df[Col].min()
        Max = Df[Col].max()
        NormDf[Col] = (Df[Col] - Min) / (Max - Min)

        Row = TrendDf.loc[TrendDf["sensor"] == Col].iloc[0]
        NormalRanges[Col] = (
            (Row["normal_low"] - Min) / (Max - Min),
            (Row["normal_high"] - Min) / (Max - Min),
        )

    Units = Df["unit_number"].unique()[:SampleUnits]
    UnitColors = dict(zip(Units, sns.color_palette(n_colors=len(Units))))

    # squeeze=False: 센서 1개여도 Axes를 배열로 받음
    Fig, Axes = plt.subplots(
        len(TopSensors), 1, figsize=(10, 2.5 * len(TopSensors)), sharex=True, squeeze=False
    )
    Axes = Axes[:, 0]

    for Ax, Col in zip(Axes, TopSensors):
        Low, High = NormalRanges[Col]
        Ax.axhspan(
            Low, High, color="seagreen", alpha=0.15, label="정상 범위(초반 5~95%)"
        )

        for Unit in Units:
            UnitDf = NormDf[NormDf["unit_number"] == Unit].sort_values(
                "RUL", ascending=False
            )
            Ax.plot(
                UnitDf["RUL"],
                UnitDf[Col],
                color=UnitColors[Unit],
                linewidth=1.8,
                marker="o",
                markersize=4,
                markevery=10,
                label=f"엔진 {Unit}",
            )

        Ax.set_ylabel(Col, fontsize=10)
        Ax.grid(True, alpha=0.3)

    Axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    Axes[-1].set_xlabel("RUL(잔여 수명)")
    Axes[-1].invert_xaxis()
    Fig.suptitle("핵심 센서별 추세 (0~1 정규화, RUL 기준, 엔진별 비교)")
    Fig.tight_layout()
    plt.show()


# 수명 진행률(%) 구간별 핵심 센서 평균값 시각화 (엔진 100개 전체 평균 - 개별 노이즈 없이 전체 경향만 보기)
def PlotSensorTrendByProgress(Df, KeySensors):
    Df = Df.copy()
    MaxCycle = Df.groupby("unit_number")["time_in_cycles"].transform("max")
    Df["CycleProgress"] = Df["time_in_cycles"] / MaxCycle * 100
    Df["ProgressBin"] = (Df["CycleProgress"] / 5).round() * 5

    TrendMeanDf = Df.groupby("ProgressBin")[KeySensors].mean()

    NumCols = 3
    NumRows = -(-len(KeySensors) // NumCols)
    Fig, Axes = plt.subplots(NumRows, NumCols, figsize=(15, 4 * NumRows))
    Axes = Axes.flatten()

    for Ax, Col in zip(Axes, KeySensors):
        Ax.plot(
            TrendMeanDf.index,
            TrendMeanDf[Col],
            color=sns.color_palette()[0],
            linewidth=2,
        )
        Ax.set_title(Col, fontsize=10)
        Ax.set_xlabel("수명 진행률(%)")
        Ax.grid(alpha=0.3)

    for Ax in Axes[len(KeySensors) :]:
        Ax.axis("off")

    Fig.suptitle("수명 진행률별 핵심 센서 평균 추세 (엔진 100개 평균)")
    Fig.tight_layout()
    plt.show()


# 원본 값과 Rolling Mean 비교 시각화 (엔진 하나 예시로)
def PlotRawVsRolling(Df, ExampleSensor, ExampleUnit=1, Windows=RollingWindows):
    ExampleDf = Df[Df["unit_number"] == ExampleUnit].sort_values("time_in_cycles")

    plt.figure(figsize=(12, 5))
    plt.plot(
        ExampleDf["time_in_cycles"], ExampleDf[ExampleSensor], label="원본", alpha=0.5
    )
    for Window in Windows:
        plt.plot(
            ExampleDf["time_in_cycles"],
            ExampleDf[f"{ExampleSensor}_rolling_mean_{Window}"],
            label=f"Rolling Mean ({Window})",
            linewidth=2,
        )
    plt.title(f"엔진 {ExampleUnit} - {ExampleSensor} 원본 vs Rolling Mean")
    plt.xlabel("Cycle")
    plt.ylabel("센서값")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# 핵심 센서에 대해 엔진별 rolling 평균/표준편차 특성 추가 (엔진 경계 안 넘게 unit_number별로 계산)
def AddRollingFeatures(Df, TopSensors, Windows=RollingWindows):
    ResultDf = Df.copy()

    for Window in Windows:
        for Col in TopSensors:
            ResultDf[f"{Col}_rolling_mean_{Window}"] = ResultDf.groupby("unit_number")[
                Col
            ].transform(lambda X: X.rolling(Window, min_periods=1).mean())
            ResultDf[f"{Col}_rolling_std_{Window}"] = (
                ResultDf.groupby("unit_number")[Col]
                .transform(lambda X: X.rolling(Window, min_periods=1).std())
                .fillna(0)
            )

    print(
        f"rolling 특성 추가 완료: 센서 {len(TopSensors)}개 x window {len(Windows)}개 x 평균/표준편차"
        f" = {len(TopSensors) * len(Windows) * 2}개 컬럼"
    )

    return ResultDf


# 자동 실행 함수
def Main():
    # 전처리 단계(Processing.py)가 항상 같은 경로에 저장하므로 파일 선택 없이 바로 로드
    Df = LoadData(CleanedPath)

    CheckLifeDistribution(Df)

    PlotCorrelationHeatmap(Df)

    TrendDf = CheckSensorTrend(Df)
    print(TrendDf)

    PlotSensorTrend(TrendDf)

    KeySensors = SelectKeySensors(TrendDf)

    # 핵심 센서가 없으면 이후 단계 불가 → 종료
    if not KeySensors:
        sys.exit("핵심 센서가 0개라 종료합니다. SelectKeySensors의 Threshold를 낮춰보세요.")

    PlotTopSensorsByUnit(Df, TrendDf, KeySensors)

    PlotSensorTrendByProgress(Df, KeySensors)

    Df = AddRollingFeatures(Df, KeySensors)

    PlotRawVsRolling(Df, KeySensors[0])

    FeaturePath = os.path.join(BaseDir, "Data", "Processed", "timeseries_features.csv")
    Df.to_csv(FeaturePath, index=False)
    print("저장 완료:", FeaturePath)


if __name__ == "__main__":
    Main()
