#!/usr/bin/env bash
# Batch gap fill: fetch metadata per missing catalog id through the Next data
# route (vpc fetch-ids), ingest, report. Runs until the catalog gap is empty or
# a batch stops producing rows. Resume-safe (state in storage/state/ids.json).
#
# Tunables: BATCH_IDS (2000) MAX_BATCHES (500) ORDER (lastmod|id|random) MAX_IDS (0 = no cap)
# Usage:  bash scripts/gapfill_loop.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BATCH_IDS="${BATCH_IDS:-2000}"
MAX_BATCHES="${MAX_BATCHES:-500}"
ORDER="${ORDER:-lastmod}"
MAX_IDS="${MAX_IDS:-0}"
LOG_DIR="${LOG_DIR:-$REPO/storage/logs}"
mkdir -p "$LOG_DIR"
PROGRESS="${PROGRESS:-$LOG_DIR/gapfill-progress.log}"
mkdir -p "$(dirname "$PROGRESS")"

db() { docker exec dev-middleware-stock-db psql -U postgres -d stock -tAc "$1"; }
log() { echo "$(date '+%F %T') $*" | tee -a "$PROGRESS"; }

attempted=0
zero_streak=0
for i in $(seq 1 "$MAX_BATCHES"); do
  before=$(db "SELECT count(*) FROM stock_videos")
  gap=$(db "SELECT count(*) FROM catalog_videos c WHERE c.missing_since IS NULL AND NOT EXISTS (SELECT 1 FROM stock_videos s WHERE s.pexels_id = c.pexels_id)")
  cov=$(db "SELECT round((SELECT count(*) FROM stock_videos)*100.0/(SELECT count(*) FROM catalog_videos),2)")
  log "batch $i start: stock=$before gap=$gap coverage=${cov}% attempted=$attempted"
  if [ "$gap" -eq 0 ]; then
    log "catalog gap is empty; stopping"
    break
  fi

  limit="$BATCH_IDS"
  if [ "$MAX_IDS" -gt 0 ]; then
    remaining=$((MAX_IDS - attempted))
    if [ "$remaining" -le 0 ]; then
      log "id cap ($MAX_IDS) reached; stopping"
      break
    fi
    [ "$remaining" -lt "$BATCH_IDS" ] && limit="$remaining"
  fi

  batch_log="$LOG_DIR/gapfill-batch-$i.log"
  uv run vpc fetch-ids --limit "$limit" --order "$ORDER" --pool 4 --retries 60 \
    >>"$batch_log" 2>&1
  attempted=$((attempted + limit))
  uv run vpc ingest --kind ids --skip-thumbnails >>"$batch_log" 2>&1

  after=$(db "SELECT count(*) FROM stock_videos")
  new=$((after - before))
  missing=$(grep -cE 'ITEM-MISS|status=404' "$batch_log" 2>/dev/null || true)
  cov=$(db "SELECT round((SELECT count(*) FROM stock_videos)*100.0/(SELECT count(*) FROM catalog_videos),2)")
  log "batch $i end: +$new rows, missing=$missing, attempted=$attempted/${MAX_IDS:-∞}, coverage=${cov}%"

  if [ "$new" -eq 0 ]; then
    zero_streak=$((zero_streak + 1))
    if [ "$zero_streak" -ge 3 ]; then
      log "no new rows for $zero_streak consecutive batches; stopping"
      break
    fi
    log "no new rows this batch (#$zero_streak); continuing (deleted ids get marked)"
  else
    zero_streak=0
  fi
done
log "LOOP-DONE"
