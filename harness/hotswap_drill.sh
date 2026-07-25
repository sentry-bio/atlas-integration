#!/usr/bin/env bash
# Spine #5 hot-swap drill — proves /admin/reload REFUSES a broken artifact and leaves the old service LIVE.
# Runs on a SCRATCH instance (port 8094) over symlinks; NEVER mutates the real index dirs. Broken artifact =
# a dir that symlinks the real 968MB index/gids/calib but carries a corrupt encoder_id in meta.json, so the
# encoder-match gate rejects it without copying anything.
set -u
R5=/zfs_raid/SentryBio/serve_index_v109_5kb
R20=/zfs_raid/SentryBio/serve_index_v109_20kb
W=/zfs_raid/SentryBio/hotswap_drill
PY=/home/rohit/biosphere-atlas-venv/bin/python
UVI=/home/rohit/biosphere-atlas-venv/bin/uvicorn
rm -rf "$W"; mkdir -p "$W/cur5" "$W/cur20" "$W/broken20"
# valid staging symlinks
for f in index.faiss gids.npy meta.json calibration.json; do ln -sf "$R5/$f" "$W/cur5/$f"; ln -sf "$R20/$f" "$W/cur20/$f"; done
# broken 20kb: real data symlinked, meta.json corrupted (bad encoder_id)
ln -sf "$R20/index.faiss" "$W/broken20/index.faiss"; ln -sf "$R20/gids.npy" "$W/broken20/gids.npy"; ln -sf "$R20/calibration.json" "$W/broken20/calibration.json"
$PY - "$R20/meta.json" "$W/broken20/meta.json" <<'PYEOF'
import json,sys
m=json.load(open(sys.argv[1])); m["encoder_id"]="sha256:DELIBERATELY_BROKEN_FOR_DRILL"
json.dump(m,open(sys.argv[2],"w"))
PYEOF
# launch scratch service on 8094 over the CURRENT (valid) symlinks
pkill -f "port 8094" 2>/dev/null; sleep 1
PYTHONPATH=/home/rohit:/home/rohit/sentrybio/scripts CUDA_VISIBLE_DEVICES="" \
 SERVE_INDEX_5KB="$W/cur5" SERVE_INDEX_20KB="$W/cur20" SERVE_DEVICE=cpu STRICT_ENCODER=1 \
 nohup $UVI atlas_serve:app --host 127.0.0.1 --port 8094 > /tmp/hotswap_8094.log 2>&1 &
cd /home/rohit/build_pipeline
for i in $(seq 1 40); do ss -ltn|grep -q :8094 && break; sleep 3; done
echo "=== scratch 8094 health (before drill) ==="
curl -s http://127.0.0.1:8094/health | $PY -c "import sys,json;d=json.load(sys.stdin);print('up, gates',{k:v['verify']['passed'] for k,v in d['tiers'].items()})"
GOOD_ID=$(curl -s http://127.0.0.1:8094/health | $PY -c "import sys,json;print(json.load(sys.stdin)['encoder_id'][:20])")
# STEP A: point 20kb symlink dir at the BROKEN artifact, reload -> expect REJECT (503) + old still live
rm -rf "$W/cur20"; cp -r "$W/broken20" "$W/cur20" 2>/dev/null || { rm -rf "$W/cur20"; mkdir "$W/cur20"; for f in index.faiss gids.npy calibration.json; do ln -sf "$W/broken20/$f" "$W/cur20/$f"; done; cp "$W/broken20/meta.json" "$W/cur20/meta.json"; }
CODE=$(curl -s -o /tmp/reload_a.json -w "%{http_code}" -X POST http://127.0.0.1:8094/admin/reload)
echo "=== STEP A: reload broken artifact -> HTTP $CODE (expect 503) ==="
STILL=$(curl -s http://127.0.0.1:8094/health | $PY -c "import sys,json;print('alive' if json.load(sys.stdin).get('status')=='ok' else 'DOWN')" 2>/dev/null || echo DOWN)
NOW_ID=$(curl -s http://127.0.0.1:8094/health | $PY -c "import sys,json;print(json.load(sys.stdin)['encoder_id'][:20])" 2>/dev/null)
echo "  old service still: $STILL ; encoder still: $NOW_ID (expect == $GOOD_ID)"
# STEP B: restore valid symlinks, reload -> expect 200
rm -rf "$W/cur20"; mkdir "$W/cur20"; for f in index.faiss gids.npy meta.json calibration.json; do ln -sf "$R20/$f" "$W/cur20/$f"; done
CODE2=$(curl -s -o /tmp/reload_b.json -w "%{http_code}" -X POST http://127.0.0.1:8094/admin/reload)
echo "=== STEP B: reload valid artifact -> HTTP $CODE2 (expect 200) ==="
# verdict
A_OK=$([ "$CODE" = "503" ] && [ "$STILL" = "alive" ] && [ "$NOW_ID" = "$GOOD_ID" ] && echo yes || echo no)
B_OK=$([ "$CODE2" = "200" ] && echo yes || echo no)
echo "=== DRILL VERDICT: broken-rejected+old-live=$A_OK  valid-swapped=$B_OK ==="
[ "$A_OK" = "yes" ] && [ "$B_OK" = "yes" ] && echo "HOTSWAP DRILL: PASS" || echo "HOTSWAP DRILL: REVIEW"
# teardown scratch
pkill -f "port 8094" 2>/dev/null; rm -rf "$W"
echo "scratch 8094 + drill artifacts torn down"
