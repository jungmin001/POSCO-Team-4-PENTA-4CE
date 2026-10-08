import os
import platform
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 폰트 설정 - Windows / Mac
# =========================================================

Font = "Malgun Gothic" if platform.system() == "Windows" else "AppleGothic"

plt.rcParams["font.family"] = Font
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 전역 설정
# =========================================================

BaseDir = os.path.dirname(os.path.abspath(__file__))
CleanedPath = os.path.join(BaseDir, "Data", "Processed", "cleaned_data.csv")

SensorCols = ["OSC-FRQ", "OSC-STK", "MLD-LVL"]
UseCols = ["EQP_CD", "SLAB_NO", "MEAS_DT"] + SensorCols

IQRMultiplier = 3.0

# IQR 정상 범위 계산
GetBounds = lambda X: (
    X.quantile(0.25) - IQRMultiplier * (X.quantile(0.75) - X.quantile(0.25)),
    X.quantile(0.75) + IQRMultiplier * (X.quantile(0.75) - X.quantile(0.25)),
)


# =========================================================
# 정제 데이터 로드
# =========================================================


def LoadData(FilePath):
    Df = pd.read_csv(FilePath, usecols=UseCols, parse_dates=["MEAS_DT"])

    Df["EQP_CD"] = Df["EQP_CD"].astype("category")

    return Df.sort_values(["EQP_CD", "SLAB_NO", "MEAS_DT"]).reset_index(drop=True)


# =========================================================
# 설비별 센서 정상 범위 계산
# =========================================================


def GetNormalRanges(LineDf):
    return {Sensor: GetBounds(LineDf[Sensor]) for Sensor in SensorCols}


# =========================================================
# SLAB별 센서 대표값 생성
# 중앙값을 사용해 전체적인 주편 흐름 확인
# =========================================================


def GetSlabProfile(LineDf):
    return LineDf.groupby("SLAB_NO", observed=True)[SensorCols].median().reset_index()


# =========================================================
# 정상 범위를 벗어난 실제 측정값 반환
# =========================================================


def GetOutliers(LineDf, Sensor, Lower, Upper):
    return LineDf[(LineDf[Sensor] < Lower) | (LineDf[Sensor] > Upper)][
        ["SLAB_NO", Sensor]
    ]


# =========================================================
# 설비별 차트 생성
#
# 사용:
# PlotEquipment(Df, "CC1")
# PlotEquipment(Df, "CC2")
# =========================================================


def PlotEquipment(Df, Line):
    LineDf = Df[Df["EQP_CD"] == Line]
    SlabDf = GetSlabProfile(LineDf)
    Ranges = GetNormalRanges(LineDf)

    X = range(len(SlabDf))

    Fig, Axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    for Ax, Sensor in zip(Axes, SensorCols):
        Lower, Upper = Ranges[Sensor]

        # SLAB별 중앙값 흐름
        Ax.plot(
            X,
            SlabDf[Sensor],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label="SLAB 중앙값",
        )

        # 정상 운전 범위
        Ax.axhspan(Lower, Upper, alpha=0.15, label="정상 범위")

        # 실제 이상치가 발생한 SLAB 표시
        Outliers = GetOutliers(LineDf, Sensor, Lower, Upper)

        if not Outliers.empty:
            SlabIndex = {Slab: Index for Index, Slab in enumerate(SlabDf["SLAB_NO"])}

            OutlierX = Outliers["SLAB_NO"].map(SlabIndex)

            Ax.scatter(
                OutlierX, Outliers[Sensor], marker="x", s=45, label="이상치", zorder=3
            )

        Ax.set_ylabel(Sensor)
        Ax.set_title(f"{Sensor}  |  정상범위 {Lower:.1f} ~ {Upper:.1f}")

        Ax.grid(alpha=0.2)
        Ax.legend(loc="upper right")

    # SLAB 번호가 너무 많으면 일부만 표시
    Step = max(1, len(SlabDf) // 15)

    Axes[-1].set_xticks(list(X)[::Step])

    Axes[-1].set_xticklabels(
        SlabDf["SLAB_NO"].astype(str).iloc[::Step], rotation=45, ha="right"
    )

    Axes[-1].set_xlabel("SLAB_NO")

    Fig.suptitle(f"{Line} 주편별 공정 센서 흐름", fontsize=16)

    Fig.tight_layout()

    plt.show()


# =========================================================
# 실행
# =========================================================


def Main():
    Df = LoadData(CleanedPath)

    # 차트 1
    PlotEquipment(Df, "CC1")

    # 차트 2
    PlotEquipment(Df, "CC2")


if __name__ == "__main__":
    Main()
