#!/usr/bin/env bash
# accept_v109.sh — the v10.9 serving ACCEPTANCE SUITE (spine #2-#5 as one re-runnable PASS/FAIL gate).
# The durable capstone: after any index rebuild / encoder swap (v11) do  rebuild -> hot-swap -> bash accept_v109.sh.
# Each step is isolated (a death in one does not abort the chain); the scorecard at the end shows what passed.
cd /home/rohit/build_pipeline
PY=/home/rohit/biosphere-atlas-venv/bin/python
export PYTHONPATH=/home/rohit:/home/rohit/sentrybio/scripts CUDA_VISIBLE_DEVICES=""
L=/tmp/accept_v109.log; : > "$L"
step(){ echo; echo "======== $* ========" | tee -a "$L"; }
run(){ ( "$@" ) 2>&1 | tee -a "$L"; }   # subshell so a die doesn't kill the chain

echo "===== ACCEPT v10.9 START $(date) =====" | tee -a "$L"

step "GATE 0 — service health + settled config + calibration provenance"
run curl -s http://127.0.0.1:8093/health -o /tmp/h.json
$PY - <<'PYEOF' 2>&1 | tee -a "$L"
import json
d=json.load(open('/tmp/h.json'))
bands=[b['max_bp'] for b in d['routing']]
gates_ok=all(v['verify']['passed'] for v in d['tiers'].values())
ids_ok=len({v['encoder_id'] for v in d['tiers'].values()})==1
cfg_ok=d['mode']=='adaptive' and bands==[3000,None]
a=json.load(open('/zfs_raid/SentryBio/serve_index_v109_5kb/calibration.json'))
b=json.load(open('/zfs_raid/SentryBio/serve_index_v109_20kb/calibration.json'))
print(f"  mode/bands ok={cfg_ok}  gates ok={gates_ok}  encoder-ids match={ids_ok}  calib 5kb==20kb={a==b}")
print("GATE0:", "PASS" if (cfg_ok and gates_ok and ids_ok and a==b) else "FAIL")
PYEOF

step "#2 — settled 4-axis benchmark (numbers-of-record)"
NG=40 NG2=120 NLAT=40 OUTJSON=/zfs_raid/SentryBio/radial_training/benchmark_settled.json run $PY benchmark_adaptive.py all

step "#3 — CEILING GAP (accuracy justice)"
NG=80 run $PY win_ceiling.py

step "Win2 — 5kb+ fusion cell (close the ghost)"
NG_W2=100 W2_LENGTHS=4000,5000,6000 run $PY benchmark_adaptive.py win2

step "#4 — HONESTY gates (graded novelty + degenerate)"
NGN=60 run $PY accept_honesty.py

step "#5 — HOT-SWAP DRILL"
run bash hotswap_drill.sh

echo | tee -a "$L"
echo "############## SCORECARD ##############" | tee -a "$L"
grep -hE "^GATE0:|PRE-REG.*(JUSTICE|HEADROOM)|SPINE #4|TEST 1 |TEST 2 |HOTSWAP DRILL:|PRE-REG @|best=|GAP (family|genus)" "$L" | sed 's/^/  /' | tee -a "$L"
echo "===== ACCEPT v10.9 DONE $(date) =====" | tee -a "$L"
