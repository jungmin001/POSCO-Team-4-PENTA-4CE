import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BaseDir = os.path.dirname(os.path.abspath(__file__))
CleanedPath = os.path.join(BaseDir, "Data", "Processed", "cleaned_data.csv")

KeyCols = ["EQP_CD", "SLAB_NO", "MEAS_DT"]

# rolling window (단기 5 + 장기 30)
RollingWindows = (5, 30)


# CSV 파일 로드 및 Df 변환하여 반환 (rolling 등 순서에 의존하는 계산이 많아서 미리 정렬)
def LoadData(FileLocation):
    Df = pd.read_csv(FileLocation, parse_dates=["MEAS_DT"])
    Df = Df.sort_values(KeyCols).reset_index(drop=True)

    return Df


# 센서 컬럼 목록 (설비·슬라브·시각을 뺀 나머지)
def GetSensorCols(Df):
    return [Col for Col in Df.columns if Col not in KeyCols]


# 슬라브별 측정 행 수 분포 확인 (CH.2의 엔진별 수명 분포 확인과 같은 역할)
def CheckSlabDistribution(Df):
    SlabRows = Df.groupby(["EQP_CD", "SLAB_NO"]).size()
    print("설비별 슬라브 수:", SlabRows.groupby(level="EQP_CD").size().to_dict())
    print(SlabRows.describe())

    plt.figure(figsize=(8, 5))
    sns.histplot(SlabRows, bins=15, color=sns.color_palette()[0])
    plt.title("슬라브별 측정 행 수 분포")
    plt.xlabel("측정 행 수")
    plt.ylabel("슬라브 수")
    plt.tight_layout()
    plt.show()

    return SlabRows


# 설비(CC1 / CC2)별 센서값 분포 비교 (두 설비는 정상 운전 범위 자체가 다름)
def CompareEquipment(Df):
    SensorCols = GetSensorCols(Df)
    print(Df.groupby("EQP_CD")[SensorCols].mean().round(1))

    Fig, Axes = plt.subplots(1, len(SensorCols), figsize=(4 * len(SensorCols), 5))
    for Ax, Col in zip(Axes, SensorCols):
        sns.boxplot(data=Df, x="EQP_CD", y=Col, ax=Ax)
        Ax.set_title(Col)
        Ax.set_xlabel("")
    Fig.suptitle("설비별 센서값 분포")
    Fig.tight_layout()
    plt.show()


# 센서끼리 상관관계 히트맵 (다중공선성 확인용 - 센서끼리 상관관계가 너무 높으면
# 사실상 같은 정보를 중복으로 갖고 있다는 뜻이라 모델링 단계에 참고가 됨)
def PlotCorrelationHeatmap(Df):
    SensorCols = GetSensorCols(Df)
    Corr = Df[SensorCols].corr(method="spearman")

    plt.figure(figsize=(8, 6))
    sns.heatmap(Corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("센서 상관관계 (Spearman, 전체 데이터 기준)")
    plt.tight_layout()
    plt.show()

    return Corr


# 센서별로 서브플롯 하나씩, 그 안에 슬라브 여러 개를 겹쳐서 그래프 (x축: 슬라브 안 측정 순번)
def PlotSensorsBySlab(Df, SampleSlabs=5):
    SensorCols = GetSensorCols(Df)
    Slabs = Df["SLAB_NO"].unique()[:SampleSlabs]
    SlabColors = dict(zip(Slabs, sns.color_palette(n_colors=len(Slabs))))

    Fig, Axes = plt.subplots(len(SensorCols), 1, figsize=(10, 2.5 * len(SensorCols)), sharex=True)

    for Ax, Col in zip(Axes, SensorCols):
        for Slab in Slabs:
            SlabDf = Df[Df["SLAB_NO"] == Slab]
            Ax.plot(range(len(SlabDf)), SlabDf[Col], color=SlabColors[Slab], linewidth=1, label=Slab)

        Ax.set_ylabel(Col, fontsize=10)
        Ax.grid(True, alpha=0.3)

    Axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    Axes[-1].set_xlabel("슬라브 안 측정 순번")
    Fig.suptitle("센서별 추세 (슬라브별 비교)")
    Fig.tight_layout()
    plt.show()


# 원본 값과 Rolling Mean 비교 시각화 (슬라브 하나 예시로)
def PlotRawVsRolling(Df, ExampleSensor, ExampleSlab, Windows=RollingWindows):
    ExampleDf = Df[Df["SLAB_NO"] == ExampleSlab]
    Order = range(len(ExampleDf))

    plt.figure(figsize=(12, 5))
    plt.plot(Order, ExampleDf[ExampleSensor], label="원본", alpha=0.5)
    for Window in Windows:
        plt.plot(Order, ExampleDf[f"{ExampleSensor}_rolling_mean_{Window}"],
                 label=f"Rolling Mean ({Window})", linewidth=2)
    plt.title(f"슬라브 {ExampleSlab} - {ExampleSensor} 원본 vs Rolling Mean")
    plt.xlabel("측정 순번")
    plt.ylabel("센서값")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# 센서에 대해 슬라브별 rolling 평균/표준편차 특성 추가 (슬라브 경계 안 넘게 SLAB_NO별로 계산)
def AddRollingFeatures(Df, SensorCols, Windows=RollingWindows):
    ResultDf = Df.copy()

    for Window in Windows:
        for Col in SensorCols:
            ResultDf[f"{Col}_rolling_mean_{Window}"] = ResultDf.groupby("SLAB_NO")[Col].transform(
                lambda X: X.rolling(Window, min_periods=1).mean()
            )
            ResultDf[f"{Col}_rolling_std_{Window}"] = (
                ResultDf.groupby("SLAB_NO")[Col]
                .transform(lambda X: X.rolling(Window, min_periods=1).std())
                .fillna(0)
            )

    print(f"rolling 특성 추가 완료: 센서 {len(SensorCols)}개 x window {len(Windows)}개 x 평균/표준편차"
          f" = {len(SensorCols) * len(Windows) * 2}개 컬럼")

    return ResultDf


# 자동 실행 함수
def Main():
    # 전처리 단계(Processing.py)가 항상 같은 경로에 저장하므로 파일 선택 없이 바로 로드
    Df = LoadData(CleanedPath)

    CheckSlabDistribution(Df)

    CompareEquipment(Df)

    PlotCorrelationHeatmap(Df)

    PlotSensorsBySlab(Df)

    # 센서가 5개뿐이라 CH.2처럼 핵심 센서를 고르지 않고 전부 사용
    SensorCols = GetSensorCols(Df)
    Df = AddRollingFeatures(Df, SensorCols)

    PlotRawVsRolling(Df, SensorCols[0], Df["SLAB_NO"].iloc[0])

    FeaturePath = os.path.join(BaseDir, "Data", "Processed", "timeseries_features.csv")
    Df.to_csv(FeaturePath, index=False)
    print("저장 완료:", FeaturePath)


if __name__ == "__main__":
    Main()
