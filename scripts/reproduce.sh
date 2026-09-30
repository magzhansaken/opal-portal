#!/bin/bash
# Reproduce every experiment of the paper on a CPU (about 15-20 CPU-hours in total).
# Every step skips finished seeds, and neural training resumes from per-epoch checkpoints,
# so the script can be interrupted and restarted at any time.
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
BASE='"epochs": 60, "patience": 8, "deep_x": true'
for DS in oulad act-mooc; do
  # act-mooc is one large cohort: recompute step activations in the backward pass to save memory (same results)
  CK=""; [ "$DS" = act-mooc ] && CK=', "grad_ckpt": true'
  python scripts/run_tabular.py --dataset $DS --models LR,RF,XGBoost,LightGBM,HGB+Platt,MLP
  python scripts/run_neural.py --dataset $DS --kind opal --name opal --seeds 0,1,2,3,4 --threads 1 --cfg "{$BASE$CK}"
  python scripts/run_neural.py --dataset $DS --kind gru --name GRU --seeds 0,1,2,3,4 --threads 1 --cfg '{"epochs": 60, "patience": 8}'
  python scripts/run_neural.py --dataset $DS --kind transformer --name Transformer --seeds 0,1,2,3,4 --threads 1 --cfg '{"epochs": 60, "patience": 8}'
  python scripts/run_graph.py --dataset $DS --models R-GCN --seeds 0,1,2,3,4 --threads 1 --lr 0.001 --wd 0.0001 --drop 0.1 --patience 8
  python scripts/run_graph.py --dataset $DS --models HGT --seeds 0,1,2,3,4 --threads 1 --lr 0.001 --wd 0.0001 --drop 0.3 --patience 8
  python scripts/run_rec.py --dataset $DS --models heur,BPR-MF,LightGCN,GRU4Rec,SASRec --seeds 0,1,2 --threads 1
done
# ablations (OULAD)
for ABL in "abl_no_graph:\"use_graph\": false" "abl_no_coevo:\"use_coevo\": false" "abl_no_rec:\"use_rec\": false" \
           "abl_no_peer:\"peer\": false" "abl_no_assess:\"use_assess\": false"; do
  NAME=${ABL%%:*}; OPT=${ABL#*:}
  python scripts/run_neural.py --dataset oulad --kind opal --name $NAME --seeds 0,1,2 --threads 1 --cfg "{$BASE, $OPT}"
done
python scripts/run_neural.py --dataset oulad --kind opal --name abl_no_featpath --seeds 0,1,2 --threads 1 --cfg '{"epochs": 60, "patience": 8, "deep_x": false, "use_wide": false}'
# validation-only variant of the recommendation head (Supplementary Section S3; not part of the reported model)
for DS in oulad act-mooc; do
  CK=""; [ "$DS" = act-mooc ] && CK=', "grad_ckpt": true'
  python scripts/run_neural.py --dataset $DS --kind opal --name val_hybrid --seeds 0 --threads 1 --cfg "{$BASE$CK, \"rec_hybrid\": true}"
done
# evaluation protocol and marking-delay sensitivity
grep -q within_presentation results/oulad/protocol_inflation.json 2>/dev/null || python scripts/protocol_inflation.py
python scripts/run_tabular.py --dataset oulad_lag14 --models LightGBM
python scripts/run_neural.py --dataset oulad_lag14 --kind opal --name opal --seeds 0,1,2 --threads 1 --cfg "{$BASE}"
# analysis
python scripts/analyze.py oulad
python scripts/analyze.py act-mooc
python scripts/dataset_stats.py
python scripts/val_variant.py
