import subprocess
import sys
from pathlib import Path

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


# 전체 파이프라인: 파트별 스크립트를 순서대로 별도 프로세스로 실행 (각 파트가 명령행 인자를 직접 읽으므로 함수 호출 대신 프로세스로 분리)
#   python Main.py all                    전처리 → 이상탐지 → 상태·원인 진단 → 고장 위험 → 최종 결함 위험 → 언리얼 파일 → 모음
#   python Main.py all skip=preprocess    전처리를 건너뜀 (이미 Data/Processed/Scenarios가 있을 때)
#   python Main.py all skip=diagnosis,anomaly   여러 단계를 건너뜀
#   python Main.py all rul                잔여수명(hee.py)도 실행 (scikit-survival 설치 필요)
# 한 단계가 실패하면 거기서 멈추고 어느 단계인지 알려 줌
BaseDir = Path(__file__).resolve().parent

PipelineSteps = [
    ("preprocess", [sys.executable, "Main.py"], "시나리오별 전처리 데이터 저장"),
    ("anomaly", [sys.executable, "AnomalyModel.py"], "C 연주 이상탐지 모델 학습"),
    ("diagnosis", [sys.executable, "Diagnosis.py"], "7개 설비 상태·원인 진단 학습·예측"),
    ("diagnosis_demo", [sys.executable, "Diagnosis.py", "--validation-dashboard"], "진단 모델로 validation 시나리오 화면용 예측 생성 (시연용, 재학습 없음)"),
    ("risk", [sys.executable, "jungmin/Main.py", "A", "B", "C", "D", "E1", "E2", "F", "noshow"], "7개 설비 고장 위험 학습·평가·모델 저장"),
    ("unreal", [sys.executable, "jungmin/unreal.py"], "언리얼용 온톨로지·비용·위험 예측 파일 생성"),
    ("quality", [sys.executable, "chanhee/Analysis.py"], "C 검사 직후 최종 결함 위험 예측 (흐름 단위)"),
    ("rul", [sys.executable, "hee.py"], "CC1 잔여수명 (옵션)"),
    ("bundle", [sys.executable, "Bundle.py"], "언리얼 출력을 Results/Unreal에 모으기"),
]


def RunAll(Args):
    Skip = {Name for Arg in Args if Arg.startswith("skip=") for Name in Arg[5:].split(",")}
    Optional = {"rul"} - set(Args)
    Unknown = Skip - {Name for Name, _, _ in PipelineSteps}

    if Unknown:
        raise SystemExit(f"알 수 없는 단계: {sorted(Unknown)} (가능: {[Name for Name, _, _ in PipelineSteps]})")

    for Index, (Name, Command, Note) in enumerate(PipelineSteps, start=1):
        if Name in Skip or Name in Optional:
            print(f"\n[{Index}/{len(PipelineSteps)}] {Name} 건너뜀", flush=True)
            continue

        print(f"\n[{Index}/{len(PipelineSteps)}] {Name}: {Note}", flush=True)

        if subprocess.run(Command, cwd=BaseDir).returncode != 0:
            raise SystemExit(f"\n{Name} 단계에서 실패했습니다. 해결 후 'python Main.py all skip={','.join(n for n, _, _ in PipelineSteps[:Index - 1])}'로 이어서 실행하세요.")

    print("\n전체 파이프라인 완료")


# 자동 실행 함수: 시나리오별 전처리 데이터셋을 Data/Processed/Scenarios에 저장
#   python Main.py                 설비 A~F, flow, 정답, history 전체 저장
#   python Main.py A C flow        지정한 것만 저장 (설비 이름, flow, answers, history)
#   python Main.py check [설비...]  판정 결과를 화면에 출력 (저장하지 않음, 설비를 안 주면 전체)
def Main(Names=None):
    export.Main(Names)

    print("\n전체 파이프라인 실행 완료")


if __name__ == "__main__":
    if sys.argv[1:2] == ["all"]:
        RunAll(sys.argv[2:])
    elif sys.argv[1:2] == ["check"]:
        for Name in (sys.argv[2:] or list(config.Profiles)):
            Check(Name)
    else:
        Main(sys.argv[1:] or None)
