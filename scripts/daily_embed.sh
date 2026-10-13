#!/usr/bin/env bash
# daily_embed.sh —— 每日增量嵌入：分批灌到 0 为止（5000 行/批，async 写路径）。
# 用法：daily_embed.sh <bm25|bge-m3>
# 60 批上限防止死循环；任何一批失败由 set -e 中止，交给 Airflow 重试。
set -euo pipefail

model="${1:?usage: daily_embed.sh <bm25|bge-m3>}"

for _ in $(seq 1 60); do
  out=$(python -m crawler.cli embed --model "$model" --limit 5000 --write async)
  echo "$out"
  n=$(printf '%s' "$out" | python3 -c 'import json,sys; print(json.load(sys.stdin)["embedded"])')
  if [ "$n" = "0" ]; then
    echo "daily_embed: $model drained"
    exit 0
  fi
done

echo "daily_embed: $model hit the 60-batch cap" >&2
exit 1
