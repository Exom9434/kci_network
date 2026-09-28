"""
04.community_features_figure.py
===============================
논문용 커뮤니티별 특징 표 이미지 생성

입력:
  results/paper_tables/table_05_community_features.csv

출력:
  results/paper_tables/table_05_community_features_figure.png

실행:
  python 04.community_features_figure.py
"""

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).parent
IN = BASE / "results" / "paper_tables" / "table_05_community_features.csv"
OUT = BASE / "results" / "paper_tables" / "table_05_community_features_figure.png"

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


def shorten_community(label: str) -> str:
    mapping = {
        "생성형 인공지능 / 저작권 / 공정이용": "생성형 AI·저작권",
        "빅데이터 / 개인정보 / 개인정보보호": "빅데이터·개인정보",
        "4차 산업혁명 / 블록체인 / 지능정보사회": "4차 산업혁명·거버넌스",
        "알고리즘 / 머신러닝 / 딥러닝": "알고리즘·머신러닝",
        "제조물책임 / 위험책임 / 소프트웨어": "제조물책임·불법행위",
        "자율주행자동차 / 규제 / 책임": "자율주행자동차",
        "EU AI Act / 인공지능기본법 / 인공지능법": "AI 기본법",
        "법인격 / 로봇 / 형사책임": "법인격·형사책임",
        "딥페이크 / 표현의 자유 / 인격권": "딥페이크·표현의 자유",
        "발명 / 발명자 / 특허법": "AI 발명·특허",
        "자율운항선박 / Maritime Autonomous Surface Ships / 원격운항자": "자율운항선박",
        "부정경쟁방지법 / Unfair Competition Prevention Act": "부정경쟁방지",
        "리걸테크 / 변호사법": "리걸테크",
        "중국 / China": "중국법",
    }
    return mapping.get(label, label.replace(" / ", "·"))


def wrap_text(value: str, width: int) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def main() -> None:
    df = pd.read_csv(IN, encoding="utf-8-sig")
    df = df.sort_values("매핑 논문 수", ascending=False).reset_index(drop=True)

    fig_df = pd.DataFrame(
        {
            "커뮤니티": df["커뮤니티"],
            "연구 군집": df["커뮤니티명"].map(shorten_community),
            "논문 수": df["매핑 논문 수"].astype(int),
            "대표 키워드": df["대표 키워드"].map(lambda x: wrap_text(x, 30)),
            "주요 쟁점": df["법학적 해석"].map(lambda x: wrap_text(x, 28)),
        }
    )

    fig_height = 0.66 * len(fig_df) + 1.9
    fig, ax = plt.subplots(figsize=(14.5, fig_height), facecolor="white")
    ax.axis("off")

    table = ax.table(
        cellText=fig_df.values,
        colLabels=fig_df.columns,
        cellLoc="left",
        colLoc="center",
        loc="center",
        colWidths=[0.07, 0.17, 0.08, 0.35, 0.33],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1, 2.15)

    header_color = "#1D3557"
    stripe_color = "#F3F6F8"
    edge_color = "#D7DEE5"

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("center")
        else:
            cell.set_facecolor(stripe_color if row % 2 == 0 else "white")
            if col in (0, 2):
                cell.get_text().set_ha("center")
            else:
                cell.get_text().set_ha("left")

    ax.set_title(
        "키워드 공출현 네트워크의 주요 연구 군집",
        fontsize=15,
        fontweight="bold",
        color="#1D3557",
        pad=18,
    )
    fig.text(
        0.5,
        0.015,
        "주: Leiden 커뮤니티 탐지 결과. 논문 수는 각 논문을 최다 키워드 소속 커뮤니티에 배정한 값.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.035, 1, 0.965])
    plt.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[저장] {OUT}")


if __name__ == "__main__":
    main()
