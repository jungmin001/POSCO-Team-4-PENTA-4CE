import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import ImageFont


# PPT 이미지 공통 글꼴: 맥 기본 한글 글꼴 Apple SD Gothic Neo (굵기별로 들어 있음)
#   matplotlib 그림: ppt_font.UseMatplotlib() 한 번 호출한 뒤 fontweight="bold" 등으로 굵기 지정
#   PIL 그림: ppt_font.Pil(크기, "bold")
# 글꼴 파일은 시스템 것을 쓰고, matplotlib용 복사본은 임시 폴더에만 만든다 (저장소에 넣지 않음)
Source = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
Faces = {"regular": 0, "medium": 2, "semibold": 4, "bold": 6}
Family = "Apple SD Gothic Neo"
CacheDir = Path(tempfile.gettempdir()) / "ppt_fonts"


def Split():
    from fontTools.ttLib import TTCollection

    CacheDir.mkdir(parents=True, exist_ok=True)
    Collection = TTCollection(str(Source))
    Paths = {}

    for Weight, Index in Faces.items():
        Path_ = CacheDir / f"AppleSDGothicNeo-{Weight}.ttf"
        if not Path_.exists():
            Collection.fonts[Index].save(str(Path_))
        Paths[Weight] = Path_

    return Paths


def UseMatplotlib():
    for Path_ in Split().values():
        font_manager.fontManager.addfont(str(Path_))

    plt.rcParams["font.family"] = Family
    plt.rcParams["axes.unicode_minus"] = False


def Pil(Size, Weight="regular"):
    return ImageFont.truetype(str(Source), Size, index=Faces[Weight])
