import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

BaseDir = os.path.dirname(os.path.abspath(__file__))

Df = pd.read_csv(os.path.join(BaseDir, "cleaned_data.csv"))
Df["timestamp"] = pd.to_datetime(Df["timestamp"])

BrokenTimes = Df.loc[Df["machine_status"] == "BROKEN", "timestamp"]
print("고장 이벤트 개수:", len(BrokenTimes))

# 일단 이 두 센서만 먼저 봄, 필요하면 목록 늘려서 다시 실행
SensorsToCheck = ["sensor_12", "sensor_04"]

for Sensor in SensorsToCheck:
    print(f"\n{Sensor} 기초 통계")
    print(Df[Sensor].describe())

    plt.figure(figsize=(14, 4))
    plt.plot(Df["timestamp"], Df[Sensor], linewidth=0.5)
    for T in BrokenTimes:
        plt.axvline(T, color="red", linestyle="--")
    plt.title(f"{Sensor} 흐름 (빨간 선 = 고장 시점)")
    plt.xlabel("시간")
    plt.ylabel(Sensor)
    plt.tight_layout()
    plt.savefig(os.path.join(BaseDir, f"{Sensor}_trend.png"))
    print(f"그래프 저장: {Sensor}_trend.png")

print("\n고장 발생 시점:")
print(BrokenTimes.to_string(index=False))
