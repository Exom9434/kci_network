#!/usr/bin/env bash
# 이 파이프라인은 중간에 '수동 라벨(final_label)' 단계가 있어 한 번에 돌리지 않습니다.
echo "이 분석은 2단계로 실행합니다:"
echo "  1) bash run_step1_build.sh   # 빌드+라이덴 → community_summary.csv 생성 후 멈춤"
echo "  2) community_summary.csv 의 final_label 을 손으로 채우기 (내부/확장 2개)"
echo "  3) bash run_step2_viz.sh     # 시각화+누적+라벨+인용통계/흐름"
