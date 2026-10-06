import Processing
import Analysis
import Modeling

Steps = [("전처리", Processing), ("시계열 분석 및 특성 생성", Analysis), ("모델링", Modeling)]

# 자동 실행 함수
def Main():
    for Index, (StepName, Module) in enumerate(Steps, start=1):
        print(f"\n===== {Index}단계: {StepName} ({Module.__name__}.py) =====")
        Module.Main()

    print("\n전체 파이프라인 실행 완료")


if __name__ == "__main__":
    Main()
