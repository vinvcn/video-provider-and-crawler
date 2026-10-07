#!/usr/bin/env bash
# Batch fan-out for Pexels video metadata (browser-driven; metadata only, no media).
#
# Each batch: sample fresh random terms -> fetch-search (20 pages/term) -> ingest
# -> report coverage and that batch's marginal novelty. Stops when the coverage
# target is reached or marginal novelty falls below the floor. Resume-safe: state
# lives in storage/state/search.json, so the loop can be interrupted any time.
#
# Tunables: TARGET_PCT (90) BATCH_TERMS (250) MAX_BATCHES (20) NOVELTY_FLOOR (50)
#
# Usage:  bash scripts/fanout_loop.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TARGET_PCT="${TARGET_PCT:-90}"
BATCH_TERMS="${BATCH_TERMS:-250}"
MAX_BATCHES="${MAX_BATCHES:-20}"
NOVELTY_FLOOR="${NOVELTY_FLOOR:-50}"
PROGRESS="${PROGRESS:-/tmp/opencode/fanout-progress.log}"
mkdir -p "$(dirname "$PROGRESS")"

db() { docker exec dev-middleware-stock-db psql -U postgres -d stock -tAc "$1"; }
log() { echo "$(date '+%F %T') $*" | tee -a "$PROGRESS"; }

pick_terms() {
python3 - "$BATCH_TERMS" <<'PY'
import json, pathlib, random, re, subprocess, sys

batch = int(sys.argv[1])
state = json.loads(pathlib.Path("storage/state/search.json").read_text())


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "term"


done = {
    key[len("search-"):].rsplit("-p", 1)[0]
    for key in state.get("done_keys", [])
    if key.startswith("search-")
}
sql = (
    "SELECT term FROM catalog_queries WHERE term ~ '^[a-z][a-z ]{2,40}$' "
    "ORDER BY random() LIMIT 8000"
)
out = subprocess.run(
    ["docker", "exec", "dev-middleware-stock-db", "psql", "-U", "postgres",
     "-d", "stock", "-tAc", sql],
    capture_output=True, text=True,
).stdout
picked, seen = [], set()
for term in out.split("\n"):
    key = slug(term)
    if term and key not in done and key not in seen:
        seen.add(key)
        picked.append(term)
    if len(picked) >= batch:
        break
random.shuffle(picked)
print(",".join(picked))
PY
}

for i in $(seq 1 "$MAX_BATCHES"); do
  before=$(db "SELECT count(*) FROM stock_videos")
  cov=$(db "SELECT round((SELECT count(*) FROM stock_videos)*100.0/(SELECT count(*) FROM catalog_videos),2)")
  log "batch $i start: stock=$before coverage=${cov}%"

  terms=$(pick_terms)
  n_terms=$(awk -F, '{print NF}' <<<"$terms")
  if [ -z "$terms" ] || [ "$n_terms" -eq 0 ]; then
    log "no fresh terms left; stopping"
    break
  fi

  batch_log="/tmp/opencode/fanout-batch-$i.log"
  uv run vpc fetch-search --terms "$terms" --pages-per-term 20 --pool 4 --retries 60 \
    >>"$batch_log" 2>&1
  uv run vpc ingest --skip-thumbnails >>"$batch_log" 2>&1
  fails=$(grep -c "ITEM-FAIL" "$batch_log" 2>/dev/null || true)
  ingest_line=$(grep -o '{"records".*}' "$batch_log" | tail -1)

  after=$(db "SELECT count(*) FROM stock_videos")
  new=$((after - before))
  per_term=$((new / n_terms))
  cov=$(db "SELECT round((SELECT count(*) FROM stock_videos)*100.0/(SELECT count(*) FROM catalog_videos),2)")
  log "batch $i end: +$new rows, terms=$n_terms, new/term=$per_term, fails=$fails, coverage=${cov}%"
  [ -n "$ingest_line" ] && log "  ingest: $ingest_line"

  if awk -v c="$cov" -v t="$TARGET_PCT" 'BEGIN{exit !(c+0>=t+0)}'; then
    log "coverage target ${TARGET_PCT}% reached; stopping"
    break
  fi
  if awk -v n="$per_term" -v f="$NOVELTY_FLOOR" 'BEGIN{exit !(n+0<f+0)}'; then
    log "novelty floor ${NOVELTY_FLOOR}/term hit; search channel is saturating; stopping"
    break
  fi
done
log "LOOP-DONE"
