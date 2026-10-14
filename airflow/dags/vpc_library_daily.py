"""vpc_library_daily —— Pexels 素材库每日增量（Airflow DAG）。

链路：目录刷新（sitemaps）→ 活水头条（fetch-seed --head，顺带刷新 buildId）
→ 补齐新 id 元数据（fetch-ids）→ 入库 + 缩略图（ingest）→ 稀疏/稠密增量嵌入
（HNSW 自动增量插入）。

执行位置分两类：

- httpx 步骤（sitemaps / ingest / embed）直接跑在 airflow 容器内：仓库
  bind-mount 到 /opt/airflow/projects/vpc；DSN 覆盖为共享网络内的
  stock-db:5432；出网走宿主代理 host.docker.internal:7897。
- 浏览器步骤（fetch-seed / fetch-ids）经 SSH 回宿主机执行：browser-use 只装在
  宿主机，且 pexels 的 _next/data 路由只对浏览器页内 fetch 豁免 Cloudflare。
  密钥见 middleware/dev/ssh/airflow_vpc（不入库）。

buildId 提示：页面 __NEXT_DATA__.buildId 随站点发版轮换；stale 时浏览器抓取
会自动重读，无需人工干预（state: storage/state/ids.json）。

调度：每天 04:30 Asia/Shanghai（低峰）；max_active_runs=1，嵌入批次不并发。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from airflow.decorators import dag
from airflow.operators.bash import BashOperator

DAG_ID = "vpc_library_daily"

_REPO = "/opt/airflow/projects/vpc"
_PRELUD = (
    "set -euo pipefail; "
    f"set -a; . {_REPO}/.env; set +a; "
    "export VPC_DB_DSN=postgresql://postgres:postgres@stock-db:5432/stock; "
    "export VPC_USE_PROXY=1; "
    "export HTTP_PROXY=http://host.docker.internal:7897; "
    "export HTTPS_PROXY=http://host.docker.internal:7897; "
    "export NO_PROXY=stock-db,localhost,127.0.0.1,.internal; "
    f"export PYTHONPATH={_REPO}; "
    f"cd {_REPO}; "
)

_SSH_HOST = (
    # 密钥对容器内 airflow 用户可读（本地单用户环境，专用单用途密钥）；
    # OpenSSH 拒绝使用组/他人可读的私钥，因此先复制到 /tmp 收紧到 600。
    "install -m 600 /opt/airflow/ssh/airflow_vpc /tmp/airflow_ssh_key && "
    "ssh -i /tmp/airflow_ssh_key "
    "-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/airflow_known_hosts "
    "-o BatchMode=yes -o ConnectTimeout=10 pc@host.docker.internal "
)
_REMOTE_ENV = (
    "export PATH=/home/pc/.local/bin:$PATH; "
    "export DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority; "
    "export HTTP_PROXY=http://localhost:7897 HTTPS_PROXY=http://localhost:7897; "
    "cd /home/pc/projects/video-provider-and-crawler && "
)

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


def _vpc(args: str) -> str:
    """容器内执行 vpc 命令。"""
    return _PRELUD + "python -m crawler.cli " + args


def _remote(args: str) -> str:
    """宿主机上执行 vpc 命令（浏览器步骤）。"""
    return _SSH_HOST + "'" + _REMOTE_ENV + "uv run vpc " + args + "'"


@dag(
    dag_id=DAG_ID,
    description="Pexels 素材库每日增量：sitemaps → head → ids → ingest → 稀疏/稠密嵌入",
    default_args=default_args,
    schedule="30 4 * * *",
    start_date=datetime(2026, 10, 13, tzinfo=ZoneInfo("Asia/Shanghai")),
    catchup=False,
    max_active_runs=1,
    tags=["vpc", "pexels", "incremental"],
)
def vpc_library_daily():
    """定义素材库每日增量任务链（sitemaps → head → ids → ingest → 嵌入）。"""
    fetch_sitemaps = BashOperator(
        task_id="fetch_sitemaps",
        bash_command=_vpc("fetch-sitemaps --kind videos"),
    )
    fetch_head = BashOperator(
        task_id="fetch_seed_head",
        bash_command=_remote("fetch-seed --head --pages 5 --retries 2"),
    )
    fetch_ids = BashOperator(
        task_id="fetch_ids",
        bash_command=_remote("fetch-ids --order lastmod --limit 2000 --retries 3"),
    )
    ingest = BashOperator(
        task_id="ingest",
        bash_command=_vpc("ingest"),
    )
    embed_sparse = BashOperator(
        task_id="embed_sparse",
        bash_command=_PRELUD + "bash scripts/daily_embed.sh bm25",
    )
    embed_dense = BashOperator(
        task_id="embed_dense",
        bash_command=_PRELUD + "bash scripts/daily_embed.sh bge-m3",
    )
    fetch_sitemaps >> fetch_head >> fetch_ids >> ingest >> embed_sparse >> embed_dense


vpc_library_daily()
