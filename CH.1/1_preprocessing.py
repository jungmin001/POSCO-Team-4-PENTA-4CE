import os
import pandas as pd

BaseDir = os.path.dirname(os.path.abspath(__file__))
RawPath = os.path.join(BaseDir, "Data", "Row", "sensor.csv")
SavePath = os.path.join(BaseDir, "cleaned_data.csv")

Df = pd.read_csv(RawPath)
print("원본 크기:", Df.shape)

if "Unnamed: 0" in Df.columns:
    Df = Df.drop(columns=["Unnamed: 0"])

Df["timestamp"] = pd.to_datetime(Df["timestamp"])
Df = Df.sort_values("timestamp").drop_duplicates().reset_index(drop=True)
print("중복 제거 후 크기:", Df.shape)

SensorCols = [Col for Col in Df.columns if Col.startswith("sensor_")]
print("센서 개수:", len(SensorCols))

# 결측치 30% 넘는 센서는 그냥 버림
MissingRate = Df[SensorCols].isna().mean() * 100
DropCols = MissingRate[MissingRate >= 30].index.tolist()
print("제거 대상 센서:", DropCols)

Df = Df.drop(columns=DropCols)
SensorCols = [Col for Col in SensorCols if Col not in DropCols]
print("남은 센서 개수:", len(SensorCols))

# 나머지 결측치는 시간 기준으로 채우기
Df = Df.set_index("timestamp")
Df[SensorCols] = Df[SensorCols].interpolate(method="time").ffill().bfill()
Df = Df.reset_index()

Df["machine_status"] = Df["machine_status"].str.strip().str.upper()
print("machine_status 종류:", Df["machine_status"].unique())

print("남은 결측치:", Df[SensorCols].isna().sum().sum())
print("데이터 크기:", Df.shape)

Df.to_csv(SavePath, index=False)
print("저장 완료:", SavePath)
