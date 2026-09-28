"""
02.category_pie_charts.py
=========================
논문용 법 분야 분포 파이/도넛 차트 생성

출력:
  results/02_descriptive/category_pie_major.png
      - 전체 2,014편 기준: 융복합 + 주요 단일 분야
  results/02_descriptive/category_pie_fusion_detail.png
      - 융복합 604편 기준: 주요 융복합 조합

실행:
  python 02.category_pie_charts.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).parent
OUT = BASE / "results" / "02_descriptive"
OUT.mkdir(parents=True, exist_ok=True)

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


def load_final_categories() -> pd.DataFrame:
    confirmed = pd.read_csv(
        BASE / "KCI_AI_논문_카테고리_확정.csv",
        encoding="utf-8-sig",
    )
    manual = pd.read_csv(
        BASE / "KCI_AI_논문_카테고리_미확정_수동검토완료.csv",
        encoding="utf-8-sig",
    )
    df = pd.concat(
        [
            confirmed[["논문ID", "최종_카테고리"]],
            manual[["논문ID", "최종_카테고리"]],
        ],
        ignore_index=True,
    ).drop_duplicates("논문ID")
    df["최종_카테고리"] = df["최종_카테고리"].fillna("").astype(str).str.strip()
    return df


def save_donut(
    counts: pd.Series,
    total: int,
    title: str,
    out_path: Path,
    colors: list[str],
    note: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 7.2), facecolor="#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    wedges, _ = ax.pie(
        counts.values,
        startangle=90,
        counterclock=False,
        colors=colors[: len(counts)],
        wedgeprops=dict(width=0.48, edgecolor="white", linewidth=1.8),
    )

    ax.text(
        0,
        0,
        f"총\n{total:,}편",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#1D3557",
    )
    ax.set_title(title, fontsize=15, fontweight="bold", color="#1D3557", pad=18)

    legend_labels = [
        f"{idx}  {val:,}편 ({val / total * 100:.1f}%)"
        for idx, val in counts.items()
    ]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=10.3,
        labelspacing=0.9,
    )

    fig.text(0.5, 0.035, note, ha="center", fontsize=9, color="#555555")
    plt.tight_layout(rect=[0, 0.06, 0.84, 1])
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="#F8F9FA")
    plt.close()
    print(f"[저장] {out_path}")


def plot_major_distribution(df: pd.DataFrame) -> None:
    categories = df["최종_카테고리"]
    grouped = categories.where(~categories.str.startswith("융복합"), "융복합")
    counts = grouped.value_counts()

    major = counts[counts >= 50].copy()
    other = counts[counts < 50].sum()
    if other:
        major.loc["기타 단일 분야"] = other

    if "융복합" in major.index:
        ordered = pd.concat(
            [major.loc[["융복합"]], major.drop(index="융복합").sort_values(ascending=False)]
        )
    else:
        ordered = major.sort_values(ascending=False)

    colors = [
        "#2A9D8F",
        "#1D3557",
        "#457B9D",
        "#E07B39",
        "#6A4C93",
        "#C4882B",
        "#E63946",
        "#5C7A3E",
        "#888888",
    ]

    save_donut(
        ordered,
        len(df),
        "법 분야별 분포: 융복합 및 주요 단일 분야",
        OUT / "category_pie_major.png",
        colors,
        "주: 융복합 세부 조합은 하나의 범주로 통합하고, 50편 미만 단일 분야는 기타 단일 분야로 묶음.",
    )


def plot_fusion_distribution(df: pd.DataFrame) -> None:
    fusion = df[df["최종_카테고리"].str.startswith("융복합")].copy()
    counts = fusion["최종_카테고리"].value_counts()

    top_n = 10
    top = counts.head(top_n).copy()
    other = counts.iloc[top_n:].sum()
    if other:
        top.loc["기타 융복합"] = other

    labels = [
        label.replace("융복합(", "").replace(")", "")
        for label in top.index
    ]
    ordered = pd.Series(top.values, index=labels)

    colors = [
        "#6A4C93",
        "#7E57C2",
        "#9575CD",
        "#B39DDB",
        "#2A9D8F",
        "#457B9D",
        "#E07B39",
        "#C4882B",
        "#E63946",
        "#5C7A3E",
        "#888888",
    ]

    save_donut(
        ordered,
        len(fusion),
        "융복합 논문의 세부 법 분야 조합",
        OUT / "category_pie_fusion_detail.png",
        colors,
        f"주: 융복합 논문 {len(fusion):,}편 중 상위 {top_n}개 조합을 개별 표시하고 나머지는 기타 융복합으로 묶음.",
    )


def main() -> None:
    df = load_final_categories()
    plot_major_distribution(df)
    plot_fusion_distribution(df)


if __name__ == "__main__":
    main()
