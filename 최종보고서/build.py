"""report.html → PDF. 1차로 찍어 각 장의 쪽 번호를 찾고, 목차에 채워 다시 찍는다."""
import os, re, subprocess, pymupdf
D = os.path.dirname(os.path.abspath(__file__))
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
OUT = os.path.join(D, 'PENTA-4CE_제철공정예지보전분석_결과보고서.pdf')
TITLES = {'c1': '1프로젝트 개요', 'c2': '2데이터와 품질', 'c3': '3공통 전처리와 시간순 분할', 'c4': '4고장 위험 예측', 'c5': '5이상탐지와 조기경보',
          'c6': '6상태·원인 진단', 'c7': '7잔여수명 추정', 'c8': '8공정 품질·최종 결함 예측', 'c9': '9통합과 현장 활용',
          'c10': '10분석 한계와 향후 과제', 'c11': '11역할과 분석 회고'}


def render(html, pdf):
    subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', '--print-to-pdf=' + pdf,
                    '--allow-file-access-from-files', 'file://' + html], check=True, capture_output=True)


src = open(os.path.join(D, 'report.html'), encoding='utf-8').read()
tmp = os.path.join(D, '_build.html')
open(tmp, 'w', encoding='utf-8').write(src)
render(tmp, OUT)
doc = pymupdf.open(OUT)
pages = {}
for i in range(2, len(doc)):
    txt = doc[i].get_text().replace('PENTA-4CE_제철 공정 예지보전 분석 결과보고서', '')
    head = re.sub(r'\s+', '', txt)[len(str(i + 1)):][:40]
    for k, t in TITLES.items():
        if k not in pages and head.startswith(t.replace(' ', '')):
            pages[k] = i + 1
print('쪽', len(doc), pages)
for k in TITLES:
    src = src.replace('<span class="pg" data-for="%s"></span>' % k, '<span class="pg" data-for="%s">%s</span>' % (k, pages.get(k, '')))
open(tmp, 'w', encoding='utf-8').write(src)
render(tmp, OUT)
os.remove(tmp)
print(OUT)
