#!/usr/bin/env bash
# =====================================================================
# STEP 2 — (final_label 채운 뒤) 시각화 + 누적 + 인용통계/흐름
#   ※ 05는 step1에서 이미 1회 실행(중복 실행 시 keyword_label_x/_y 중복컬럼 생김)
# 파이썬: uv 환경(.venv). 다른 환경이면 PY="uv run python" bash run_step2_viz.sh
# =====================================================================
set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
echo "[python] $PY"

"$PY" - <<'PY'
import pandas as pd
paths=["results/03_network/community_summary.csv","results/03-2_network/community_summary.csv"]
allempty=True
for p in paths:
    try:
        d=pd.read_csv(p)
        filled=d["final_label"].notna().sum() if "final_label" in d.columns else 0
        print(f"[점검] {p}: final_label {filled}/{len(d)} 채움")
        if filled>0: allempty=False
    except FileNotFoundError:
        raise SystemExit(f"!! {p} 없음 — 먼저 run_step1_build.sh 실행.")
if allempty:
    raise SystemExit("!! final_label 두 파일 모두 비어있음. 채우고 다시 실행.\n"
                     "   (자동라벨로 강행하려면 이 점검 블록을 주석 처리)")
PY

run(){ echo; echo "==================== $1 ===================="; "$PY" "$1"; }
run "03.network_viz.py"
run "03-2.network_viz(extened).py"
run "03.network_cumulative.py"
run "06.citation_network.py"
run "06.citation_ratio_trend.py"
run "07.citation_flow_analysis.py"
run "08.topic_citation_flow.py"

echo; echo "✅ STEP 2 완료 — 결과는 $(pwd)/results/"
