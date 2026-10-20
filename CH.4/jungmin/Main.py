import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jungmin import config, features, models

if config.Show:
    import matplotlib.pyplot as plt


# 고장 위험 예측 전체 실행 (CH.4 폴더에서 실행, 전처리는 하지 않고 Data/Processed/Scenarios를 읽기만 함)
#   python jungmin/Main.py                 연주 C × 6H·24H: 특징 만들기 → 학습·평가 → 비교표
#   python jungmin/Main.py C 24H           위험 구간을 지정 (하나만 고르면 상세 결과도 출력)
#   python jungmin/Main.py A B C           다른 설비도 지정하면 실행 (설비 이름 A, B, C, D, E1, E2, F)
#   python jungmin/Main.py reuse           만들어 둔 특징 표(outputs/features_*.csv)가 있으면 다시 만들지 않음
#   python jungmin/Main.py features        특징 표만 만듦
#   python jungmin/Main.py noshow          그림을 창으로 띄우지 않고 저장만 함
def Main(Args):
    Reuse = "reuse" in Args
    FeaturesOnly = "features" in Args
    Names = [Name for Name in Args if Name not in ("reuse", "features", "noshow")]

    Equips = [Name for Name in Names if Name in config.Equipments] or config.DefaultEquipments
    Horizons = [Name for Name in Names if Name in config.Horizons] or config.Horizons
    Verbose = len(Equips) * len(Horizons) == 1

    Summaries = []

    for Equip in Equips:
        if not (Reuse and features.TablePath(Equip).exists()):
            features.Build(Equip)

        if FeaturesOnly:
            continue

        for Horizon in Horizons:
            if not Verbose:
                print(f"    {Equip} {Horizon} 학습·평가 중", flush=True)

            Summaries.append(models.Run(Equip, Horizon, Verbose)[1])

    if not Summaries:
        return

    SummaryDf = pd.DataFrame(Summaries)
    SummaryDf.to_csv(config.OutDir / "summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n===== 전체 비교 (validation, 임계값은 train 교차검증으로 선택) =====")
    print(SummaryDf.round(3).to_string(index=False))
    print(f"\n저장 위치: {config.OutDir}")

    # 저장한 그림을 창으로도 보여줌 (창을 모두 닫으면 종료)
    if config.Show:
        plt.show()


if __name__ == "__main__":
    Main(sys.argv[1:])
