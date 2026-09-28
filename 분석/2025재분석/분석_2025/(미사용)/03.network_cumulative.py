"""
03.network_cumulative.py
========================
누적식 공시적 분석 — 내부 네트워크(03) + 확장 네트워크(03-2) 동시 처리

[분석 방식]
  전체 네트워크로 감지한 커뮤니티 배정을 그대로 사용하고,
  각 연도까지 누적된 논문 수를 집계하여 커뮤니티 성장 과정을 추적.
  → 커뮤니티 매칭 문제 없음, 방법론적으로 안정적.

[출력 — 내부 네트워크]  results/03_network/
  cumulative_growth.png        커뮤니티별 누적 성장 곡선
  cumulative_stacked.png       누적 스택 영역 차트 (비중 변화)
  cumulative_emergence.png     커뮤니티 등장 시점 타임라인
  cumulative_snapshot_*.png    주요 연도 네트워크 스냅샷

[출력 — 확장 네트워크]  results/03-2_network/
  동일한 파일들

사용법:
    python 03.network_cumulative.py
"""

import os, warnings
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── 한글 폰트 ─────────────────────────────────────────────────────
_font_candidates = [
    # macOS
    '/System/Library/Fonts/AppleSDGothicNeo.ttc',
    '/Library/Fonts/NanumGothic.ttf',
    '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
    # Linux
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
]
_font_path = next((p for p in _font_candidates if os.path.exists(p)), None)
if _font_path:
    fm.fontManager.addfont(_font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=_font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

BASE       = Path(__file__).parent
YEARS      = list(range(2016, 2026))
SNAP_YEARS = [2017, 2019, 2021, 2023, 2025]   # 스냅샷 연도
EMERGE_MIN = 5    # 커뮤니티 "등장"으로 인정할 최소 누적 논문 수


# ════════════════════════════════════════════════════════════════════
# 공통 분석 함수
# ════════════════════════════════════════════════════════════════════

def run_analysis(tag: str, graphml_path: Path, node_csv_path: Path,
                 summary_csv_path: Path, out_dir: Path):
    """
    tag             : '내부' or '확장'
    graphml_path    : network_giant(_with_community).graphml
    node_csv_path   : node_community.csv
    summary_csv_path: community_summary.csv
    out_dir         : 출력 폴더
    """
    print(f'\n{"="*60}')
    print(f'  [{tag} 네트워크] 누적 분석')
    print(f'{"="*60}')

    # ── 데이터 로드 ───────────────────────────────────────────────
    G          = nx.read_graphml(str(graphml_path))
    df_nodes   = pd.read_csv(node_csv_path,    encoding='utf-8-sig')
    df_summary = pd.read_csv(summary_csv_path, encoding='utf-8-sig')

    # community_summary: final_label 우선, 없으면 auto_label
    def resolve_label(row):
        fl = str(row.get('final_label', '')).strip()
        if fl and fl not in ('', 'nan'):
            return fl
        al = str(row.get('auto_label', '')).strip()
        return al if al and al != 'nan' else '기타'

    id2label = {int(r['community_id']): resolve_label(r)
                for _, r in df_summary.iterrows()}

    # node_community.csv community_id → 레이블
    df_nodes['community_id'] = pd.to_numeric(df_nodes['community_id'], errors='coerce').fillna(-1).astype(int)
    df_nodes['label'] = df_nodes['community_id'].map(id2label).fillna('기타')

    # 확장 네트워크면 AI 논문만 집계 (비AI KCI는 연도 정보 불완전)
    if 'node_type' in df_nodes.columns:
        df_plot = df_nodes[df_nodes['node_type'] == 'ai'].copy()
    else:
        df_plot = df_nodes.copy()

    df_plot = df_plot[df_plot['year'].between(2016, 2025)]
    df_plot['year'] = df_plot['year'].astype(int)

    # 기타 커뮤니티 제외
    df_plot = df_plot[df_plot['label'] != '기타']

    # 레이블 목록 (커뮤니티 총 크기 내림차순)
    label_order = (df_plot.groupby('label')['node_id']
                   .count().sort_values(ascending=False).index.tolist())

    # 색상 팔레트
    palette = (list(plt.cm.Set2.colors) + list(plt.cm.Set1.colors) +
               list(plt.cm.tab20.colors))
    lbl_color = {lbl: palette[i % len(palette)]
                 for i, lbl in enumerate(label_order)}

    # ── 연도별 누적 집계 ──────────────────────────────────────────
    # cumul_cnt[label][year] = 해당 연도까지 누적 논문 수
    annual = defaultdict(lambda: defaultdict(int))
    for _, r in df_plot.iterrows():
        annual[r['label']][r['year']] += 1

    cumul = {}
    for lbl in label_order:
        cumul[lbl] = []
        total = 0
        for yr in YEARS:
            total += annual[lbl].get(yr, 0)
            cumul[lbl].append(total)

    # ── ① 누적 성장 곡선 ─────────────────────────────────────────
    n_col  = 2
    n_row  = (len(label_order) + 1) // 2
    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(14, max(6, n_row * 3)),
                             sharex=True)
    axes = np.array(axes).flatten()

    for i, lbl in enumerate(label_order):
        ax = axes[i]
        ax.fill_between(YEARS, cumul[lbl], alpha=0.3, color=lbl_color[lbl])
        ax.plot(YEARS, cumul[lbl], marker='o', markersize=4,
                color=lbl_color[lbl], linewidth=2)
        ax.set_title(lbl, fontsize=10, pad=4)
        ax.set_xticks(YEARS)
        ax.set_xticklabels(YEARS, rotation=45, fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        ax.yaxis.set_major_locator(
            plt.MaxNLocator(integer=True))

    # 남는 axes 숨기기
    for j in range(len(label_order), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'[{tag} 네트워크] 커뮤니티별 누적 논문 수 (2016–2025)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / 'cumulative_growth.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [저장] cumulative_growth.png')

    # ── ② 누적 스택 영역 차트 ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    cumul_matrix = np.array([cumul[lbl] for lbl in label_order])
    ax.stackplot(YEARS, cumul_matrix,
                 labels=label_order,
                 colors=[lbl_color[lbl] for lbl in label_order],
                 alpha=0.85)
    ax.set_title(f'[{tag} 네트워크] 커뮤니티별 누적 논문 수 변화 (스택)',
                 fontsize=13)
    ax.set_xlabel('발행연도')
    ax.set_ylabel('누적 논문 수')
    ax.set_xticks(YEARS)
    ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'cumulative_stacked.png', dpi=150)
    plt.close(fig)
    print('  [저장] cumulative_stacked.png')

    # ── ③ 커뮤니티 등장 시점 타임라인 ───────────────────────────
    emerge_year = {}
    for lbl in label_order:
        total = 0
        for yr in YEARS:
            total += annual[lbl].get(yr, 0)
            if total >= EMERGE_MIN:
                emerge_year[lbl] = yr
                break

    fig, ax = plt.subplots(figsize=(13, max(5, len(label_order) * 0.45 + 2)))
    for i, lbl in enumerate(reversed(label_order)):
        yr = emerge_year.get(lbl)
        if yr is None:
            continue
        # 등장 연도부터 2025까지 선
        ax.barh(i, 2025 - yr + 1, left=yr, height=0.6,
                color=lbl_color[lbl], alpha=0.8)
        # 등장 연도 마커
        ax.plot(yr, i, 'o', color=lbl_color[lbl],
                markersize=8, markeredgecolor='white', markeredgewidth=1.2)
        # 최종 누적 수
        final_cnt = cumul[lbl][-1]
        ax.text(2025.3, i, f'{final_cnt}편', va='center', fontsize=8)

    ax.set_yticks(range(len(label_order)))
    ax.set_yticklabels(list(reversed(label_order)), fontsize=9)
    ax.set_xlim(2015.5, 2026.5)
    ax.set_xticks(YEARS)
    ax.set_xticklabels(YEARS, rotation=45)
    ax.set_title(f'[{tag} 네트워크] 커뮤니티 등장 시점 타임라인\n'
                 f'(● 누적 {EMERGE_MIN}편 도달 시점, 막대 끝 숫자: 최종 누적 편수)',
                 fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlabel('연도')
    fig.tight_layout()
    fig.savefig(out_dir / 'cumulative_emergence.png', dpi=150)
    plt.close(fig)
    print('  [저장] cumulative_emergence.png')

    # 등장 시점 요약 출력
    print(f'\n  [커뮤니티 등장 시점 (누적 {EMERGE_MIN}편 기준)]')
    print(f'  {"커뮤니티":22s}  {"등장연도":>6s}  {"최종편수":>6s}')
    print('  ' + '-' * 40)
    for lbl in label_order:
        yr  = emerge_year.get(lbl, '?')
        cnt = cumul[lbl][-1]
        print(f'  {lbl:22s}  {str(yr):>6s}  {cnt:>6d}')

    # ── ④ 주요 연도 네트워크 스냅샷 ─────────────────────────────
    print(f'\n  [스냅샷] 주요 연도별 네트워크 ({", ".join(map(str, SNAP_YEARS))})')

    # 전체 레이아웃 한 번만 계산 (AI 논문 노드 기준)
    ai_nodes_all = [n for n in G.nodes()
                    if G.nodes[n].get('node_type', 'ai') == 'ai'
                    or 'node_type' not in G.nodes[n]]
    G_ai = G.subgraph(ai_nodes_all)
    pos  = nx.spring_layout(G_ai, k=2.0, seed=42, weight='weight', iterations=80)

    # 커뮤니티 ID → label 매핑 (노드 속성에서)
    node2label = {}
    for n in G.nodes():
        raw = G.nodes[n].get('community_id', -1)
        try:
            cid = int(float(raw))
        except (ValueError, TypeError):
            cid = -1
        node2label[n] = id2label.get(cid, '기타')

    fig, axes = plt.subplots(1, len(SNAP_YEARS),
                             figsize=(5 * len(SNAP_YEARS), 5))
    if len(SNAP_YEARS) == 1:
        axes = [axes]

    for ax, snap_yr in zip(axes, SNAP_YEARS):
        # 해당 연도까지 발행된 AI 노드만
        visible = [n for n in ai_nodes_all
                   if int(G.nodes[n].get('year', 0)) <= snap_yr]
        if not visible:
            ax.set_title(f'{snap_yr}년\n(데이터 없음)')
            ax.axis('off')
            continue

        G_snap = G.subgraph(visible)
        pos_snap = {n: pos[n] for n in visible if n in pos}

        # 엣지
        nx.draw_networkx_edges(G_snap, pos_snap, ax=ax,
                               alpha=0.12, width=0.4, edge_color='#888888')

        # 노드 (커뮤니티 색상)
        for lbl in label_order:
            nodes_lbl = [n for n in visible
                         if node2label.get(n) == lbl and n in pos_snap]
            if not nodes_lbl:
                continue
            nx.draw_networkx_nodes(G_snap, pos_snap, ax=ax,
                                   nodelist=nodes_lbl,
                                   node_size=20,
                                   node_color=lbl_color[lbl],
                                   alpha=0.85,
                                   linewidths=0)

        n_vis = len(visible)
        n_edge = G_snap.number_of_edges()
        ax.set_title(f'{snap_yr}년 이전\n{n_vis}편 / {n_edge}건',
                     fontsize=10)
        ax.axis('off')

    # 공통 범례
    patches = [mpatches.Patch(color=lbl_color[lbl], label=lbl)
               for lbl in label_order]
    fig.legend(handles=patches, loc='lower center',
               fontsize=7, ncol=min(6, len(label_order)),
               framealpha=0.8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f'[{tag} 네트워크] 주요 연도 누적 스냅샷',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'cumulative_snapshot.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [저장] cumulative_snapshot.png')


# ════════════════════════════════════════════════════════════════════
# 내부 네트워크 (03)
# ════════════════════════════════════════════════════════════════════
out_03 = BASE / 'results' / '03_network'
run_analysis(
    tag             = '내부',
    graphml_path    = out_03 / 'network_giant_with_community.graphml',
    node_csv_path   = out_03 / 'node_community.csv',
    summary_csv_path= out_03 / 'community_summary.csv',
    out_dir         = out_03,
)

# ════════════════════════════════════════════════════════════════════
# 확장 네트워크 (03-2)
# ════════════════════════════════════════════════════════════════════
out_032 = BASE / 'results' / '03-2_network'
run_analysis(
    tag             = '확장',
    graphml_path    = out_032 / 'network_extended_giant_with_community.graphml',
    node_csv_path   = out_032 / 'node_community.csv',
    summary_csv_path= out_032 / 'community_summary.csv',
    out_dir         = out_032,
)

print('\n' + '=' * 60)
print('  모든 누적 분석 완료!')
print('=' * 60)
