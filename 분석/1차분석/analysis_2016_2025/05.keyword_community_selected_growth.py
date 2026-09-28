"""
05.keyword_community_selected_growth.py
=======================================
논문용 주요 키워드 커뮤니티 5개 누적 성장 그래프 생성

입력:
  results/05_keyword_community_trend/kw_community_trend_data.csv

출력:
  results/05_keyword_community_trend/kw_community_selected_growth.png
  results/05_keyword_community_trend/kw_community_selected_growth.csv

실행:
  python 05.keyword_community_selected_growth.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).parent
IN = BASE / "results" / "05_keyword_community_trend" / "kw_community_trend_data.csv"
OUT = BASE / "results" / "05_keyword_community_trend"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2016, 2026))

FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

for fp in FONT_CANDIDATES:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break

plt.rcParams["axes.unicode_minus"] = False


SELECTED = [
    {
        "source": "생성형 인공지능 / 저작권 / 공정이용",
        "label": "생성형 AI·저작권",
        "note": "ChatGPT 이후 급성장",
        "color": "#E63946",
    },
    {
        "source": "빅데이터 / 개인정보 / 개인정보보호",
        "label": "빅데이터·개인정보",
        "note": "GDPR·데이터법제 축적",
        "color": "#2A9D8F",
    },
    {
        "source": "4차 산업혁명 / 블록체인 / 지능정보사회",
        "label": "4차 산업혁명·거버넌스",
        "note": "초기 포괄 담론",
        "color": "#1D3557",
    },
    {
        "source": "알고리즘 / 머신러닝 / 딥러닝",
        "label": "알고리즘·머신러닝",
        "note": "투명성·공정성 논의",
        "color": "#F4A261",
    },
    {
        "source": "EU AI Act / 인공지능기본법 / 인공지능법",
        "label": "EU AI Act·인공지능기본법",
        "note": "최근 규제법제 성장",
        "color": "#6A4C93",
    },
]


def load_selected() -> pd.DataFrame:
    df = pd.read_csv(IN, encoding="utf-8-sig")
    selected_sources = {item["source"] for item in SELECTED}
    missing = selected_sources - set(df["community_label"].unique())
    if missing:
        raise ValueError(f"입력 데이터에서 찾지 못한 커뮤니티: {sorted(missing)}")

    label_map = {item["source"]: item["label"] for item in SELECTED}
    note_map = {item["source"]: item["note"] for item in SELECTED}

    selected = df[df["community_label"].isin(selected_sources)].copy()
    selected["display_label"] = selected["community_label"].map(label_map)
    selected["note"] = selected["community_label"].map(note_map)
    selected = selected.sort_values(["display_label", "year"])
    selected.to_csv(
        OUT / "kw_community_selected_growth.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return selected


def plot(selected: pd.DataFrame) -> None:
    color_map = {item["label"]: item["color"] for item in SELECTED}
    label_order = [item["label"] for item in SELECTED]

    pivot = (
        selected.pivot_table(
            index="year",
            columns="display_label",
            values="cumul_count",
            aggfunc="sum",
        )
        .reindex(index=YEARS, columns=label_order)
        .ffill()
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor="#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for label in label_order:
        values = pivot[label].astype(int)
        ax.plot(
            YEARS,
            values,
            marker="o",
            markersize=5,
            linewidth=2.6,
            color=color_map[label],
            label=label,
        )
        ax.text(
            YEARS[-1] + 0.12,
            values.iloc[-1],
            f"{values.iloc[-1]}편",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=color_map[label],
        )

    events = {
        2018: "GDPR",
        2021: "EU AI Act\n초안",
        2022: "ChatGPT",
        2024: "EU AI Act\n채택",
        2025: "인공지능\n기본법",
    }
    ymax = max(pivot.max()) * 1.14
    for year, label in events.items():
        ax.axvline(year, color="#777777", linewidth=0.8, linestyle=":", alpha=0.55)
        ax.text(
            year,
            ymax * 0.98,
            label,
            ha="center",
            va="top",
            fontsize=8,
            color="#555555",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="#DDDDDD",
                alpha=0.86,
            ),
        )

    ax.set_title(
        "주요 키워드 커뮤니티의 누적 논문 수 변화(2016-2025)",
        fontsize=15,
        fontweight="bold",
        color="#1D3557",
        pad=16,
    )
    ax.set_xlabel("발행연도", fontsize=11)
    ax.set_ylabel("누적 논문 수", fontsize=11)
    ax.set_xticks(YEARS)
    ax.set_xlim(YEARS[0] - 0.25, YEARS[-1] + 1.0)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.36)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=9.5,
    )

    fig.text(
        0.5,
        0.02,
        "주: 논문-커뮤니티 매핑 결과를 기준으로 집계. 주요 5개 커뮤니티만 표시.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = OUT / "kw_community_selected_growth.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="#F8F9FA")
    plt.close()
    print(f"[저장] {out_path}")


def main() -> None:
    selected = load_selected()
    plot(selected)


if __name__ == "__main__":
    main()
