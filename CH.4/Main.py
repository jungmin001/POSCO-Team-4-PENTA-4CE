import sys

from src import config, detect, export, screen, zoom

# 설비 하나의 결측·이상치 판정 결과를 화면에 출력 (파일은 저장하지 않음)
# 점검 → 훑어보기 → 주변 탐지 순서, 앞 단계 결과를 파일이 아니라 값으로 다음 단계에 넘김
def Check(Name):
    config.Use(Name)

    print(f"\n===== 설비 {Name} ({config.Source}) =====")
    print(f"\n--- 1단계: 전수 점검 ({detect.__name__}.py) ---")
    SummaryDf, FlagDf = detect.Main()

    print(f"\n--- 2단계: 시나리오 훑어보기 ({screen.__name__}.py) ---")
    ScreenDf = screen.Main(SummaryDf)

    print(f"\n--- 3단계: 의심 지점 주변 탐지 ({zoom.__name__}.py) ---")
    zoom.Main(FlagDf, ScreenDf)


# 자동 실행 함수: 시나리오별 전처리 데이터셋을 Data/Processed/Scenarios에 저장
#   python Main.py                 설비 A~F, flow, 정답, history 전체 저장
#   python Main.py A C flow        지정한 것만 저장 (설비 이름, flow, answers, history)
#   python Main.py check [설비...]  판정 결과를 화면에 출력 (저장하지 않음, 설비를 안 주면 전체)
def Main(Names=None):
    export.Main(Names)

    print("\n전체 파이프라인 실행 완료")


if __name__ == "__main__":
    if sys.argv[1:2] == ["check"]:
        for Name in (sys.argv[2:] or list(config.Profiles)):
            Check(Name)
    else:
        Main(sys.argv[1:] or None)
