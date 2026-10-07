#!/usr/bin/env bash
# Daily incremental crawl (metadata only): feed head delta -> ingest -> refresh
# the sitemap oracle -> coverage report. Safe to run repeatedly.
#
# Semantics: the first run has no head marker, so it only records one (HEAD-BOOTSTRAP,
# a single page). Every later run walks from the live feed head down to the marker
# recorded by the previous completed run and then advances the marker. A crashed
# run never advances the marker, so nothing is skipped.
#
# Suggested schedule (systemd user timer or cron), e.g. daily 04:30:
#   30 4 * * *  cd /home/pc/projects/video-provider-and-crawler && bash scripts/daily_incremental.sh >> /tmp/opencode/daily-incremental.log 2>&1
#
# Usage:  bash scripts/daily_incremental.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# --pages caps a runaway walk if the marker is lost (a healthy delta is a few pages).
MAX_PAGES="${MAX_PAGES:-400}"

log() { echo "$(date '+%F %T') $*"; }

log "head-delta fetch (cap ${MAX_PAGES} pages)"
uv run vpc fetch-seed --head --pages "$MAX_PAGES" --retries 20

log "ingest (metadata only)"
uv run vpc ingest --skip-thumbnails

log "refresh sitemap oracle (new ids / lastmod)"
uv run vpc fetch-sitemaps --kind videos

log "coverage"
docker exec dev-middleware-stock-db psql -U postgres -d stock -tAc "
SELECT 'stock=' || (SELECT count(*) FROM stock_videos)
    || ' catalog=' || (SELECT count(*) FROM catalog_videos)
    || ' coverage=' || round((SELECT count(*) FROM stock_videos) * 100.0
                             / (SELECT count(*) FROM catalog_videos), 2) || '%'
    || ' thumb_pending=' || (SELECT count(*) FROM stock_videos WHERE thumbnail_path IS NULL);"

log "done"
