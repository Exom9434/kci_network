#!/usr/bin/env bash
# =====================================================================
# STEP 1 — 네트워크 빌드 + 라이덴 + 키워드라벨(05) → 여기서 멈춤
#   05를 먼저 돌려 community_summary.csv 에 keyword_label(서술형)을 넣어둠.
#   그걸 보며 final_label 을 채운 뒤 run_step2_viz.sh 실행.
# 파이썬: uv 환경(.venv). 다른 환경이면 PY="uv run python" bash run_step1_build.sh
# =====================================================================
set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
echo "[python] $PY"

if [ ! -e "00.KCI_AI_논문_상세_및_인용데이터.csv" ]; then
  echo "!! 인용데이터 없음. 먼저: (상위폴더) python run_recollect.py --year 2025"; exit 1
fi
"$PY" - <<'PY'
import pandas as pd
df=pd.read_csv("00.KCI_AI_논문_상세_및_인용데이터.csv",encoding="utf-8-sig",dtype=str,low_memory=False)
y=pd.to_numeric(df["pub_year"],errors="coerce"); t=df["target_arti_id"].astype(str).str.strip()
n=((y==2025)&(t!="")&(t!="nan")).sum()
print(f"[점검] 2025 참조행: {n:,}")
if n==0: raise SystemExit("!! 2025 참조 0 — recollect 미반영. 중단.")
PY

run(){ echo; echo "==================== $1 ===================="; "$PY" "$1"; }
run "03.network_build.py"
run "03.network_leiden.py"
run "03-2.network_build(extended).py"
run "03-2.network_leiden(extened).py"
run "05.citation_keyword_label.py"     # keyword_label 을 community_summary.csv 에 추가(라벨링 보조)
run "compare_communities.py"            # 예전 커뮤니티와 겹침 매칭표(라벨 이어붙이기 판단)

echo
echo "======================================================================"
echo " STEP 1 완료. 아래 두 파일의 final_label 을 채우세요(참고 컬럼 활용):"
echo "   - results/03_network/community_summary.csv     (내부)"
echo "   - results/03-2_network/community_summary.csv    (확장)"
echo "   보조자료: 같은 파일의 keyword_label(서술형) + auto_label,"
echo "            results/label_compare/*_community_match.csv(예전 라벨 매칭)"
echo " 채운 뒤 →  bash run_step2_viz.sh"
echo "======================================================================"
