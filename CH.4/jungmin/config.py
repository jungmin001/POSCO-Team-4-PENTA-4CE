import sys
from pathlib import Path

# 그림을 파일로 저장하면서 창으로도 보여줌, 명령에 noshow를 붙이면 저장만 함 (예: python jungmin/Main.py noshow)
Show = "noshow" not in sys.argv

BaseDir = Path(__file__).resolve().parent
ScenarioDir = BaseDir.parent / "Data" / "Processed" / "Scenarios"
OutDir = BaseDir / "outputs"

# 설비 이름 (Scenarios/{시나리오}/{설비}.csv)과 위험 구간, 설비를 지정하지 않으면 기본 설비(연주 C)만 실행
Equipments = ["A", "B", "C", "D", "E1", "E2", "F"]
DefaultEquipments = ["C"]
Horizons = ["6H", "24H"]

# 특징: 정답 시점 이전 WindowHours시간 구간을 요약 (시점 이전 데이터만 사용)
WindowHours = [6, 12]
MinSlopePoints = 3

# 센서가 아닌 열 (식별·가동 여부·표본 설정·정답·판정 표시), 나머지 숫자 열은 모두 센서로 봄
ExcludeCols = {"ROW_ID", "T_SEC", "RUNNING", "N", "FS_HZ", "SEQ_NO", "GRADE"}
ExcludePrefixes = ("IS_", "GT_")

# 정답 시점마다 붙는 열 (특징이 아님)
MetaCols = ["scenario", "split", "equip", "time", "T_SEC", "RISK_6H", "RISK_24H"]

# 모델
RandomState = 42
Folds = 5
ForestTrees = 300
ForestLeaf = 5
Thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
