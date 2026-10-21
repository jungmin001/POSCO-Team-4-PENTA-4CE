"""최종보고서 그림 생성. 숫자는 각 파트 결과 파일·재실행 로그에서 옮겨 적었다."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
plt.rcParams.update({'font.family': 'Apple SD Gothic Neo', 'axes.unicode_minus': False, 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold'})
BLUE, ORANGE, GREY, GREEN, RED = '#3D6FB6', '#E08A3C', '#B8C0CC', '#3C9D6E', '#C9504A'
EQ = ['A', 'B', 'C', 'D', 'E1', 'E2', 'F']


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=200); plt.close(fig)


def label(ax, bars, fmt='%.2f', dy=0.01):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt % b.get_height(), ha='center', va='bottom', fontsize=9)


# 1 데이터: 시나리오 1개당 행 수 (로그)와 시간순 분할
rows = [8640, 5616, 51840, 864, 36, 36, 144]
iv = ['30초', '10분', '5초', '5분', '2시간', '2시간', '30분']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1.3, 1]})
b = a1.bar(EQ, rows, color=BLUE); a1.set_yscale('log'); a1.set_title('설비별 시나리오 1개(72시간)당 행 수')
for bb, r, t in zip(b, rows, iv):
    a1.text(bb.get_x() + bb.get_width() / 2, r * 1.15, '%s행\n(%s)' % (format(r, ','), t), ha='center', fontsize=8.5)
a1.set_ylim(10, 400000); a1.set_ylabel('행 수 (로그)')
sp = [('학습', 216, '2025.01 ~ 2027.05', BLUE), ('검증', 72, '2027.05 ~ 2028.02', ORANGE), ('평가', 72, '2028.02 ~ 2028.12', GREY)]
left = 0
for n, v, per, c in sp:
    a2.barh([0], [v], left=left, color=c, height=0.5)
    a2.text(left + v / 2, 0, '%s %d개\n%s' % (n, v, per), ha='center', va='center', fontsize=9, color='white' if c != GREY else '#333')
    left += v
a2.set_xlim(0, 360); a2.set_yticks([]); a2.set_xlabel('시나리오 수 (시작 시각 순)'); a2.set_title('시간순 분할 (360개)')
a2.spines['left'].set_visible(False)
save(fig, 'data_overview.png')

# 2 고장 위험: 설비별 PR-AUC
p24 = [0.688, 0.835, 0.768, 0.788, 0.899, 0.845, 0.873]
p6 = [0.35, 0.30, 0.575, 0.69, 0.66, 0.63, 0.35]
x = np.arange(7); w = 0.38
fig, ax = plt.subplots(figsize=(10, 3.8))
label(ax, ax.bar(x - w / 2, p24, w, color=BLUE, label='24시간 이내 고장'))
label(ax, ax.bar(x + w / 2, p6, w, color=ORANGE, label='6시간 이내 고장'))
ax.set_xticks(x, EQ); ax.set_ylim(0, 1.05); ax.set_ylabel('검증 PR-AUC'); ax.legend(frameon=False, loc='upper left', ncol=2)
ax.set_title('설비별 고장 위험 예측 성능 (검증 72개 시나리오)')
save(fig, 'risk_prauc.png')

# 3 이상탐지: 기준 × 지속시간
th = ['95%', '97%', '99%']; per = ['1분', '2분', '3분']
rec = np.array([[100, 97.5, 69.1], [100, 87.7, 46.9], [96.3, 54.3, 17.3]])
fa = np.array([[23.3, 9.32, 4.28], [18.3, 6.03, 2.47], [9.98, 2.31, 1.03]])
fig, ax = plt.subplots(figsize=(8.6, 4.2))
cols = [BLUE, ORANGE, GREEN]; mk = ['o', 's', '^']
for i in range(3):
    ax.plot(fa[i], rec[i], '-', color=cols[i], alpha=.5)
    for j in range(3):
        sel = (i == 1 and j == 1)
        ax.scatter(fa[i, j], rec[i, j], s=160 if sel else 60, marker=mk[i], color=cols[i], edgecolor='black' if sel else 'none', lw=1.5, zorder=3,
                   label=('기준 %s' % th[i]) if j == 0 else None)
        ax.annotate(per[j], (fa[i, j], rec[i, j]), textcoords='offset points', xytext=(6, -12), fontsize=8.5, color='#444')
ax.annotate('선택: 97% · 2분\n6시간 재현율 87.7%, 오경보 6.0건/일', (6.03, 87.7), xytext=(11, 70), fontsize=9.5,
            arrowprops=dict(arrowstyle='->', color='#333'))
ax.axhline(95, ls=':', color=GREY); ax.text(14, 96, '탐지율 95% 기준', ha='center', fontsize=8.5, color='#666')
ax.set_xlabel('하루 오경보 수'); ax.set_ylabel('고장 6시간 전 탐지율 (%)'); ax.legend(frameon=False, loc='lower right')
ax.set_title('경보 기준과 지속시간에 따른 탐지율·오경보 (FAULT 81건)')
save(fig, 'anomaly_tradeoff.png')

# 4 진단: 설비별 상태 F1, 원인 F1, top-2 포함률
sf1 = [0.960, 0.997, 0.838, 0.740, 0.969, 0.951, 0.981]
cf1 = [0.323, 0.452, 0.163, 0.453, 0.761, 0.837, 0.296]
top2 = [0.818, 0.951, 0.515, 0.943, 1.0, 1.0, 0.869]
w = 0.27
fig, ax = plt.subplots(figsize=(10, 3.8))
label(ax, ax.bar(x - w, sf1, w, color=BLUE, label='상태 macro-F1'))
label(ax, ax.bar(x, cf1, w, color=ORANGE, label='원인 macro-F1'))
label(ax, ax.bar(x + w, top2, w, color=GREEN, label='원인 상위 2개 포함률'))
ax.set_xticks(x, EQ); ax.set_ylim(0, 1.15); ax.legend(frameon=False, ncol=3, loc='upper left')
ax.set_title('설비별 상태·원인 진단 성능 (검증)')
save(fig, 'diag_summary.png')

# 5 잔여수명: 단계별 MAE
st = ['기준선\n(중앙값)', '센서 3개', '현재값 7개', '+12시간\n변화량', '튜닝', '임박 가중\n4배', '+초기 변화·\n부하 (최종)']
mae = [13.56, 11.81, 10.73, 8.44, 8.39, 8.40, 8.32]
imm = [22.50, 14.85, 13.66, 10.50, 10.34, 8.09, 7.71]
x7 = np.arange(len(st)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 3.9))
label(ax, ax.bar(x7 - w / 2, mae, w, color=BLUE, label='전체 152행 MAE'), '%.2f', 0.2)
label(ax, ax.bar(x7 + w / 2, imm, w, color=RED, label='임박(6시간 이하) 16행 MAE'), '%.2f', 0.2)
ax.set_xticks(x7, st, fontsize=9.5); ax.set_ylabel('MAE (시간)'); ax.set_ylim(0, 25); ax.legend(frameon=False)
ax.set_title('C 연주설비 잔여수명 모델 개선 단계별 오차')
save(fig, 'rul_steps.png')

# 6 품질: 예측 시점별 AP, 입력 추가별 AP
pts = ['E 직후\n(8시간 전)', 'C 직후\n(6시간 전)', 'B 직후\n(4시간 전)', 'A 직후\n(2시간 전)']
ap = [0.454, 0.740, 0.842, 0.932]
inp = ['검사값만', '+B', '+A', '+D', '+B·A·D', '최종\n(145개 특징)']
ap2 = [0.740, 0.813, 0.807, 0.767, 0.863, 0.9635]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
bb = a1.bar(pts, ap, color=[GREY, ORANGE, GREY, GREY]); label(a1, bb, '%.3f')
a1.set_ylim(0, 1.08); a1.set_ylabel('AP'); a1.set_title('예측 시점별 AP (검사값만)'); a1.tick_params(axis='x', labelsize=9)
bb = a2.bar(inp, ap2, color=[GREY] * 4 + [ORANGE, BLUE]); label(a2, bb, '%.3f')
a2.set_ylim(0, 1.08); a2.set_title('C 직후 시점에 지난 공정 센서를 더했을 때'); a2.tick_params(axis='x', labelsize=9)
save(fig, 'quality_ap.png')

# 7 품질: 임계값에 따른 찾은 결함·오경보
fig, ax = plt.subplots(figsize=(6.4, 3.4))
b1 = ax.bar([0 - .2, 1 - .2], [269, 299], .4, color=BLUE, label='찾은 결함 (308건 중)')
b2 = ax.bar([0 + .2, 1 + .2], [30, 96], .4, color=ORANGE, label='오경보')
label(ax, b1, '%d', 3); label(ax, b2, '%d', 3)
ax.set_xticks([0, 1], ['임계값 0.5', '임계값 0.131 (선택)']); ax.set_ylim(0, 340); ax.legend(frameon=False, loc='upper left')
ax.set_title('임계값을 낮췄을 때 (검증 1,978개 흐름)')
save(fig, 'quality_threshold.png')

# 8 언리얼 화면: 에디터 테두리와 알림을 잘라 낸다
W = '/Users/jungmin/Documents/UnrealProject/team_ue5/Saved/MeshGen/renders/widget/'
Image.open(W + 'style_flat_main.png').crop((0, 22, 1280, 632)).save(os.path.join(OUT, 'ue_main.png'))
Image.open(W + 'style_flat_evidence.png').crop((0, 22, 1280, 632)).save(os.path.join(OUT, 'ue_evidence.png'))
Image.open(W + 'dashboard_full_after.png').crop((278, 512, 962, 698)).save(os.path.join(OUT, 'ue_timeline.png'))
print('ok')
