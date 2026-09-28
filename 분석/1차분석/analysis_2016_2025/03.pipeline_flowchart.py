"""
03.pipeline_flowchart.py
========================
분석 파이프라인 흐름도 생성 (PNG)

출력: results/02_descriptive/pipeline_flowchart.png
실행: python 03.pipeline_flowchart.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# ── 한글 폰트 설정 ──────────────────────────────────────────────────
_font_candidates = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",          # macOS
    "/Library/Fonts/NanumGothic.ttf",                      # macOS (나눔)
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # macOS 구버전
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",     # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for _fp in _font_candidates:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).parent
OUT  = BASE / "results" / "02_descriptive"
OUT.mkdir(parents=True, exist_ok=True)

# ── 노드 정의 ────────────────────────────────────────────────────────
# (cx, cy, w, h, 번호, 제목, 설명, 배경색, 글자색, 챕터)
NODES = [
    # 단일 블록
    dict(cx=0.50, cy=0.930, w=0.42, h=0.080,
         title="① 데이터 수집",
         desc="KCI 법학 AI 논문 2,014편 (2016–2025)\n인용 관계 포함",
         bg="#1D3557", fg="white", ch="1장"),

    dict(cx=0.50, cy=0.812, w=0.42, h=0.080,
         title="② 기술 통계 분석",
         desc="연도별 성장 추이 · 학술지 분포\n단일 1,410편 / 융복합 604편",
         bg="#2A9D8F", fg="white", ch="2장"),

    # 왼쪽 경로
    dict(cx=0.23, cy=0.644, w=0.37, h=0.080,
         title="③-A  키워드 전처리",
         desc="SYNONYM_MAP 정규화 (~90개 규칙)\nSTOP_KEYWORDS 제거 · 한국어 우선 정규화",
         bg="#E63946", fg="white", ch="3–4장"),

    dict(cx=0.23, cy=0.508, w=0.37, h=0.080,
         title="④  키워드 공출현 네트워크",
         desc="Min. freq ≥ 5 · Min. co-occ ≥ 3\n거대 컴포넌트: 224 노드 / 1,268 엣지",
         bg="#E07B39", fg="white", ch="4장"),

    dict(cx=0.23, cy=0.372, w=0.37, h=0.080,
         title="⑤  Leiden 클러스터링",
         desc="Resolution = 1.0 · Modularity = 0.64\n14개 커뮤니티 도출",
         bg="#C4882B", fg="white", ch="4장"),

    dict(cx=0.23, cy=0.236, w=0.37, h=0.080,
         title="⑥  논문 → 커뮤니티 매핑",
         desc="키워드 다수결 투표 방식\n1,333편 매핑 (66.2%)",
         bg="#9B4521", fg="white", ch="4장"),

    # 오른쪽 경로
    dict(cx=0.77, cy=0.644, w=0.37, h=0.080,
         title="③-B  인용 네트워크 구축",
         desc="논문 간 인용 관계 → 방향 그래프\n내부(AI↔AI) / 확장(AI↔전체 KCI)",
         bg="#6A4C93", fg="white", ch="3장"),

    dict(cx=0.77, cy=0.508, w=0.37, h=0.080,
         title="④  인용 커뮤니티 탐지",
         desc="Leiden 클러스터링 적용\n커뮤니티별 상위 키워드 3개 레이블링",
         bg="#4D6B8A", fg="white", ch="3장"),

    dict(cx=0.77, cy=0.372, w=0.37, h=0.080,
         title="⑤  커뮤니티 구조 분석",
         desc="중심성 지표 (PageRank, Betweenness)\n허브·브리지 논문 식별",
         bg="#3A5570", fg="white", ch="3장"),

    # 최종 통합
    dict(cx=0.50, cy=0.080, w=0.72, h=0.090,
         title="⑦  시계열 누적 분석  &  연구 트렌드 종합",
         desc="키워드 커뮤니티 성장 곡선 · 등장 시점 타임라인 · 연도×커뮤니티 히트맵\n"
              "인용 커뮤니티 누적 분석 · 핵심 키워드 시계열 · 연구 영역별 발전 흐름",
         bg="#1D3557", fg="white", ch="5장"),
]

N = {i: d for i, d in enumerate(NODES)}

def center(i):
    d = N[i]
    return d["cx"], d["cy"]

def top(i):
    d = N[i]
    return d["cx"], d["cy"] + d["h"] / 2

def bottom(i):
    d = N[i]
    return d["cx"], d["cy"] - d["h"] / 2

# ── 그리기 ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("#F4F6F9")
ax.set_facecolor("#F4F6F9")

ARROW = dict(arrowstyle="-|>", color="#8899AA", lw=1.8, mutation_scale=18)

def draw_node(ax, d):
    cx, cy, w, h = d["cx"], d["cy"], d["w"], d["h"]
    # 그림자
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2 + 0.004, cy - h/2 - 0.004), w, h,
        boxstyle="round,pad=0.012", lw=0,
        facecolor="#BBBBBB", alpha=0.35, zorder=1))
    # 본체
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.012", lw=1.5,
        edgecolor="white", facecolor=d["bg"], zorder=2))
    # 챕터 태그
    ax.text(cx + w/2 - 0.012, cy + h/2 - 0.006,
            f"[{d['ch']}]", ha="right", va="top",
            fontsize=7.5, color="white", alpha=0.7,
            fontweight="bold", zorder=3)
    # 제목
    ax.text(cx, cy + h * 0.12, d["title"],
            ha="center", va="center",
            fontsize=10.5, fontweight="bold",
            color=d["fg"], zorder=3)
    # 설명
    ax.text(cx, cy - h * 0.22, d["desc"],
            ha="center", va="center",
            fontsize=8.5, color=d["fg"],
            alpha=0.90, linespacing=1.45, zorder=3)

for d in NODES:
    draw_node(ax, d)

def arrow(ax, x0, y0, x1, y1, **kw):
    props = {**ARROW, **kw}
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(**props))

def polyline(ax, pts, color="#8899AA", lw=1.6, ls="--", arrowhead=True):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=0)
    if arrowhead:
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, mutation_scale=16))

# ─ 직선 화살표 ─────────────────────────────────────────────────────
# ① → ②
arrow(ax, *bottom(0), *top(1))

# 왼쪽 체인
arrow(ax, *bottom(2), *top(3))
arrow(ax, *bottom(3), *top(4))
arrow(ax, *bottom(4), *top(5))

# 오른쪽 체인
arrow(ax, *bottom(6), *top(7))
arrow(ax, *bottom(7), *top(8))

# ─ 분기 (② → 왼쪽③A, 오른쪽③B) ──────────────────────────────────
bx, by = bottom(1)   # ②의 바닥
mid_y = 0.730        # 분기 수평선 y

ax.plot([bx, bx], [by, mid_y], color="#8899AA", lw=1.6, zorder=0)
# → 왼쪽
lx, ly = top(2)
ax.plot([bx, lx], [mid_y, mid_y], color="#8899AA", lw=1.6, ls="--", zorder=0)
ax.annotate("", xy=(lx, ly), xytext=(lx, mid_y),
            arrowprops=dict(arrowstyle="-|>", color="#8899AA", lw=1.6, mutation_scale=16))
# → 오른쪽
rx, ry = top(6)
ax.plot([bx, rx], [mid_y, mid_y], color="#8899AA", lw=1.6, ls="--", zorder=0)
ax.annotate("", xy=(rx, ry), xytext=(rx, mid_y),
            arrowprops=dict(arrowstyle="-|>", color="#8899AA", lw=1.6, mutation_scale=16))
ax.text(0.50, mid_y + 0.008, "두 갈래 분석",
        ha="center", va="bottom", fontsize=8, color="#8899AA", style="italic")

# ─ 통합 (⑥왼 + ⑤오른 → ⑦) ─────────────────────────────────────
merge_y = 0.155
fx, fy = top(9)     # ⑦의 top

# 왼쪽 기둥
lbx, lby = bottom(5)
ax.plot([lbx, lbx], [lby, merge_y], color="#8899AA", lw=1.6, ls="--", zorder=0)
# 오른쪽 기둥
rbx, rby = bottom(8)
ax.plot([rbx, rbx], [rby, merge_y], color="#8899AA", lw=1.6, ls="--", zorder=0)
# 수평 연결선
ax.plot([lbx, rbx], [merge_y, merge_y], color="#8899AA", lw=1.6, ls="--", zorder=0)
# 중앙 화살표
ax.plot([fx, fx], [merge_y, fy], color="#8899AA", lw=1.6, zorder=0)
ax.annotate("", xy=(fx, fy), xytext=(fx, merge_y + 0.004),
            arrowprops=dict(arrowstyle="-|>", color="#8899AA", lw=1.6, mutation_scale=18))
ax.text(0.50, merge_y + 0.006, "결과 통합",
        ha="center", va="bottom", fontsize=8, color="#8899AA", style="italic")

# ─ 제목 ─────────────────────────────────────────────────────────────
ax.text(0.50, 0.990, "KCI 법학 AI 논문 분석 파이프라인",
        ha="center", va="top", fontsize=17, fontweight="bold", color="#1D3557")
ax.text(0.50, 0.968, "Korean Legal AI Paper Analysis Pipeline (2016–2025)",
        ha="center", va="top", fontsize=10, color="#666666")

# ─ 범례 ─────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor="#1D3557", label="1장: 데이터 수집"),
    mpatches.Patch(facecolor="#2A9D8F", label="2장: 기술 통계"),
    mpatches.Patch(facecolor="#E63946", label="3–4장: 키워드 분석"),
    mpatches.Patch(facecolor="#6A4C93", label="3장: 인용 네트워크"),
    mpatches.Patch(facecolor="#E07B39", label="4장: 공출현 네트워크"),
    mpatches.Patch(facecolor="#C4882B", label="4장: Leiden 클러스터링"),
    mpatches.Patch(facecolor="#1D3557", label="5장: 시계열·종합"),
]
ax.legend(handles=legend_items, loc="lower right",
          fontsize=8, framealpha=0.88, title="챕터 구분",
          title_fontsize=8.5, bbox_to_anchor=(0.99, 0.01))

plt.tight_layout(rect=[0, 0, 1, 0.975])
out_path = OUT / "pipeline_flowchart.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"[저장] {out_path}")
