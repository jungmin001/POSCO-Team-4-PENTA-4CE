import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BaseDir = os.path.dirname(os.path.abspath(__file__))
FilePath = os.path.join(BaseDir, "Data", "Processed", "cleaned_data.csv")

Sensors = ["WD-SPD", "WD-LOD", "OSC-FRQ", "OSC-STK", "MLD-LVL"]
CastingSensors = ["OSC-FRQ", "OSC-STK", "MLD-LVL"]


# =========================================================
# 데이터 로드
# =========================================================

Df = pd.read_csv(FilePath, parse_dates=["MEAS_DT"])

Df = Df.sort_values(["EQP_CD", "SLAB_NO", "MEAS_DT"]).reset_index(drop=True)


# =========================================================
# 슬라브별 공정 진행률 생성
#
# 목적:
# 슬라브마다 길이가 다르므로
# 시작 0% ~ 종료 100%로 맞춰 공통 흐름 확인
# =========================================================

Df["STEP"] = Df.groupby(["EQP_CD", "SLAB_NO"]).cumcount()

Df["MAX_STEP"] = Df.groupby(["EQP_CD", "SLAB_NO"])["STEP"].transform("max")

Df["PROGRESS"] = Df["STEP"] / Df["MAX_STEP"].replace(0, np.nan) * 100

Df["PROGRESS_BIN"] = Df["PROGRESS"] // 10 * 10


# =========================================================
# 1. CC1 / CC2 기본 센서 상태
# =========================================================

print("\n===== CC1 / CC2 기본 센서 상태 =====")

print(Df.groupby("EQP_CD")[Sensors].agg(["mean", "std", "min", "max"]).round(2))


# =========================================================
# 2. 공정 초반 → 후반 변화
#
# 목적:
# 공정이 진행되면서 센서 중심값이 변하는지 확인
# =========================================================

for Line in ["CC1", "CC2"]:

    Temp = Df[Df["EQP_CD"] == Line]

    Early = Temp[Temp["PROGRESS"] <= 20][Sensors].median()

    Middle = Temp[Temp["PROGRESS"].between(40, 60)][Sensors].median()

    Late = Temp[Temp["PROGRESS"] >= 80][Sensors].median()

    Result = pd.DataFrame(
        {
            "초반": Early,
            "중반": Middle,
            "후반": Late,
            "변화율(%)": (Late - Early) / Early.abs() * 100,
        }
    ).round(2)

    print(f"\n[{Line}]")
    print(Result)


# =========================================================
# 3. 공정 진행 흐름 그래프
#
# 센서마다 단위가 다르므로
# 각 센서의 중앙값 대비 변화율(%)로 비교
# =========================================================

for Line in ["CC1", "CC2"]:

    Temp = Df[Df["EQP_CD"] == Line]

    Profile = Temp.groupby("PROGRESS_BIN")[Sensors].median()

    Base = Temp[Sensors].median()

    Profile = (Profile - Base) / Base.abs() * 100

    plt.figure(figsize=(10, 5))

    for Sensor in Sensors:
        plt.plot(Profile.index, Profile[Sensor], marker="o", label=Sensor)

    plt.axhline(0, linestyle="--", alpha=0.5)

    plt.title(f"{Line} 주조 공정 센서 흐름")
    plt.xlabel("공정 진행률 (%)")
    plt.ylabel("평소 수준 대비 변화율 (%)")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================================================
# 4. OSC 계열 센서 상관관계
#
# 목적:
# OSC-FRQ / OSC-STK / MLD-LVL이
# 서로 같이 움직이는 센서인지 확인
# =========================================================

for Line in ["CC1", "CC2"]:

    Temp = Df[Df["EQP_CD"] == Line]

    Corr = Temp[CastingSensors].corr(method="spearman")

    print(f"\n[{Line} 상관관계]")
    print(Corr.round(2))

    plt.figure(figsize=(5, 4))

    sns.heatmap(Corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)

    plt.title(f"{Line} 주조 센서 상관관계")
    plt.tight_layout()
    plt.show()
