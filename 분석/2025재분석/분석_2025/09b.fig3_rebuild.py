"""
09b.fig3_rebuild.py
===================
그림 3(주제군 순위 변동) 재작성 — 동점 처리와 초기 미출현 주제군 처리를 고침

09.figures_v13.py의 fig3()는 순위를 `rank(method="first")`로 매긴다. 동점일 때
데이터프레임 행 순서가 순위를 가르므로, 실제로는 같은 편수인 주제군에 서로 다른
순위가 붙는다. V13 기준으로 네 군데에서 발생한다.

    초기  0편  5개 (EU AI Act, 딥페이크, 발명, 부정경쟁방지법, 중국) → 10~14위
    중기  8편  3개 (딥페이크, 리걸테크, 자율운항선박)                 →  9~11위
    후기  7편  2개 (발명, 자율운항선박)                             → 11~12위
    후기 34편  3개 (법인격, 자율주행자동차, 제조물책임)               →  7~9위

이 스크립트는 세 가지를 바꾼다.

    1. 동점은 `method="min"`으로 공동 순위를 부여한다.
    2. 해당 기간 논문이 0편인 주제군은 순위를 부여하지 않는다(빈칸).
       존재하지 않던 주제군에 순위를 매기면 "10위에서 3위로 상승"처럼 읽히지만
       실제로는 "없다가 생김"이다.
    3. 공동 순위는 선이 완전히 겹치므로 jitter로 살짝 어긋나게 그린다.
       jitter는 표시용일 뿐이며 라벨에 찍히는 순위 숫자는 공동 순위 그대로다.

입력:
    results/05_keyword_community_trend/kw_community_trend_data.csv
출력:
    results/원고그림_V13/그림3_주제군_순위변동_v2.png  (+ .pdf)
    results/원고그림_V13/그림3_데이터_v2.csv
    results/원고그림_V13/그림3_동점점검_v2.txt

사용법:
    cd 2025인용_재수집/분석_2025
    python 09b.fig3_rebuild.py

결과가 마음에 들면 09.figures_v13.py의 fig3()에 옮기고 파일명에서 _v2를 떼면 된다.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
TREND = RESULTS / "05_keyword_community_trend" / "kw_community_trend_data.csv"
OUT = RESULTS / "원고그림_V13"
OUT.mkdir(parents=True, exist_ok=True)

PERIODS = {"초기": (2016, 2018), "중기": (2019, 2022), "후기": (2023, 2025)}

# 공동 순위 표시용 세로 어긋남. 0으로 두면 선이 완전히 겹친다.
JITTER = 0.16

# 그림 안에 넣을 제목. 빈 문자열이면 제목 없이 그린다(캡션이 제목 역할).
TITLE = "연구 주제군 순위 변동"

_fonts = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_fp = next((p for p in _fonts if os.path.exists(p)), None)
if _fp:
    fm.fontManager.addfont(_fp)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"] = False


def period_rank_share(df: pd.DataFrame, s: int, e: int):
    """편수 0인 주제군은 순위 NaN. 동점은 공동 순위(method='min')."""
    sub = df[df.year.between(s, e)]
    agg = sub.groupby("community_label")["annual_count"].sum()
    share = agg / agg.sum() * 100
    present = agg > 0
    rank = share.where(present).rank(ascending=False, method="min")
    return rank, share, agg


def pack_labels(node_ys, min_gap, y_lo, y_hi):
    order = list(np.argsort(node_ys))
    y = np.array(node_ys, float)
    for _ in range(200):
        moved = False
        for i in range(1, len(order)):
            a, b = order[i - 1], order[i]
            if y[b] - y[a] < min_gap:
                push = (min_gap - (y[b] - y[a])) / 2
                y[a] -= push
                y[b] += push
                moved = True
        y = np.clip(y, y_lo, y_hi)
        if not moved:
            break
    return y


def apply_jitter(ranks: pd.Series) -> pd.Series:
    """같은 순위끼리 위아래로 고르게 벌린다. 중앙값이 원래 순위."""
    out = ranks.astype(float).copy()
    for r, grp in ranks.dropna().groupby(ranks.dropna()):
        n = len(grp)
        if n == 1:
            continue
        offs = (np.arange(n) - (n - 1) / 2) * JITTER
        out.loc[grp.index] = r + offs
    return out


def main() -> None:
    df = pd.read_csv(TREND, encoding="utf-8-sig")
    er, es, ec = period_rank_share(df, *PERIODS["초기"])
    mr, ms, mc = period_rank_share(df, *PERIODS["중기"])
    lr, ls, lc = period_rank_share(df, *PERIODS["후기"])

    summary = pd.DataFrame({
        "early_n": ec, "early_rank": er, "early_share": es,
        "mid_n": mc, "mid_rank": mr, "mid_share": ms,
        "late_n": lc, "late_rank": lr, "late_share": ls,
    }).sort_values("late_rank")
    summary.round(1).to_csv(OUT / "그림3_데이터_v2.csv", encoding="utf-8-sig")

    # ── 동점 점검표 ────────────────────────────────────────────────
    lines = ["그림3 동점·미출현 점검", "=" * 58, ""]
    for name, (n, r) in {"초기": (ec, er), "중기": (mc, mr), "후기": (lc, lr)}.items():
        zero = list(n[n == 0].index)
        if zero:
            lines.append(f"[{name}] 논문 0편 → 순위 미부여 {len(zero)}개")
            lines += [f"    - {z}" for z in zero]
        for val, g in n[n > 0].groupby(n[n > 0]):
            if len(g) > 1:
                rk = int(r[g.index[0]])
                lines.append(f"[{name}] {int(val)}편 동점 {len(g)}개 → 공동 {rk}위")
                lines += [f"    - {i}" for i in g.index]
        lines.append("")
    txt = "\n".join(lines)
    (OUT / "그림3_동점점검_v2.txt").write_text(txt, encoding="utf-8")
    print(txt)

    # ── 그리기 ────────────────────────────────────────────────────
    jr = {k: apply_jitter(v) for k, v in
          {"early": er, "mid": mr, "late": lr}.items()}

    cmap = plt.get_cmap("tab20")
    comm = list(summary.index)
    colors = {l: cmap(i / max(len(comm) - 1, 1)) for i, l in enumerate(comm)}

    x_pos = [0, 1, 2]
    x_labels = ["초기\n2016–2018", "중기\n2019–2022", "후기\n2023–2025"]
    N = len(summary)
    Y_LO, Y_HI = 0.6, N + 0.4
    MIN_GAP = 0.62

    fig, ax = plt.subplots(figsize=(12.5, 11))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for xp in x_pos:
        ax.axvline(xp, color="#DDDDDD", lw=1.0, zorder=0)

    node_left, node_right = {}, {}
    for lbl in summary.index:
        color = colors[lbl]
        ys_true = [er.get(lbl), mr.get(lbl), lr.get(lbl)]
        ys_draw = [jr["early"].get(lbl), jr["mid"].get(lbl), jr["late"].get(lbl)]
        xs = [x for x, y in zip(x_pos, ys_draw) if pd.notna(y)]
        yd = [y for y in ys_draw if pd.notna(y)]

        ax.plot(xs, yd, color=color, lw=2.6, alpha=0.9, marker="o", markersize=15,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=2.6,
                zorder=3, solid_capstyle="round")
        for xi, yv, tv in zip(x_pos, ys_draw, ys_true):
            if pd.isna(yv):
                continue
            ax.text(xi, yv, f"{int(tv)}", ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold", zorder=5)

        # 초기 미출현 표시
        if pd.isna(ys_draw[0]):
            ax.plot([x_pos[0]], [ys_draw[1]], marker="x", markersize=9,
                    color=color, alpha=0.35, zorder=2)
            ax.plot([x_pos[0], x_pos[1]], [ys_draw[1], ys_draw[1]],
                    color=color, lw=1.2, ls=":", alpha=0.35, zorder=2)

        node_left[lbl] = ys_draw[0] if pd.notna(ys_draw[0]) else ys_draw[1]
        node_right[lbl] = ys_draw[2]

    labels = list(summary.index)
    lyL = pack_labels([node_left[l] for l in labels], MIN_GAP, Y_LO, Y_HI)
    lyR = pack_labels([node_right[l] for l in labels], MIN_GAP, Y_LO, Y_HI)

    for l, yy in zip(labels, lyL):
        color = colors[l]
        short = str(l).split("/")[0].strip()
        mark = "" if summary.loc[l, "early_n"] > 0 else "  (미출현)"
        ax.plot([-0.045, -0.008], [yy, node_left[l]], color=color, lw=0.8,
                alpha=0.55, zorder=2)
        ax.text(-0.06, yy, short + mark, ha="right", va="center", fontsize=11,
                color=color, fontweight="bold",
                alpha=1.0 if summary.loc[l, "early_n"] > 0 else 0.55)
    for l, yy in zip(labels, lyR):
        color = colors[l]
        short = str(l).split("/")[0].strip()
        ax.plot([2.008, 2.045], [node_right[l], yy], color=color, lw=0.8,
                alpha=0.55, zorder=2)
        ax.text(2.06, yy, f"{short}  ({summary.loc[l, 'late_share']:.1f}%)",
                ha="left", va="center", fontsize=11, color=color, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=14, fontweight="bold")
    ax.set_xlim(-1.05, 3.05)
    ax.invert_yaxis()
    ax.set_ylim(N + 0.7, 0.3)
    ax.set_ylabel("점유율 순위  (1 = 최상위)", fontsize=13)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    # 논문에서는 캡션이 제목 역할을 하므로 TITLE = "" 로 두면 제목이 빠진다.
    if TITLE:
        ax.set_title(TITLE, fontsize=16, fontweight="bold", pad=18)
    plt.tight_layout()
    # 동점·미출현 처리에 관한 설명은 그림에 넣지 않는다. 원고의 그림 캡션 소관이다.
    for ext, dpi in [("png", 200), ("pdf", None)]:
        p = OUT / f"그림3_주제군_순위변동_v2.{ext}"
        plt.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"[저장] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
