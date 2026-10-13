#!/usr/bin/env bash
# build_indexes.sh — production ANN indexes (docs/phase2-index-design §4.4).
#
# Not a migration: CREATE INDEX CONCURRENTLY cannot run inside a transaction,
# and store/migrate.py executes each file on one connection. Memory guardrails:
# maintenance_work_mem covers one shard-sized build (~1.8GB peak at 222k dense
# rows); parallel workers off because the container shm is 64MB.
#
# The sparse arm needs the cast expression: the column is bare `sparsevec`
# (no declared dimension), which HNSW cannot index directly —
# "column does not have dimensions". Queries must ORDER BY the same
# expression (crawler/cli.py cmd_search).
set -uo pipefail

docker exec dev-middleware-stock-db psql -U postgres -d stock -v ON_ERROR_STOP=1 \
  -c "SET maintenance_work_mem = '1280MB'" \
  -c "SET max_parallel_maintenance_workers = 0" \
  -c "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stock_videos_embedding_hnsw
        ON stock_videos USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)" \
  -c "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stock_videos_sparse_hnsw
        ON stock_videos USING hnsw ((sparse_embedding::sparsevec(1000000000)) sparsevec_ip_ops)" \
  -c "ANALYZE stock_videos"

docker exec dev-middleware-stock-db psql -U postgres -d stock -tAc \
  "SELECT indexname || ' = ' || pg_size_pretty(pg_relation_size(indexname::regclass))
   FROM pg_indexes WHERE tablename = 'stock_videos' AND indexname LIKE '%hnsw%';"
