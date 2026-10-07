"""Forward-only SQL migration runner (applied versions tracked in schema_migrations)."""

from __future__ import annotations

from pathlib import Path

from store.db import connect

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def main(dsn: str | None = None) -> None:
    """Apply every pending migrations/*.sql in lexical order."""
    conn = connect(dsn)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.commit()
        applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        pending = sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.stem not in applied)
        if not pending:
            print("no pending migrations")
            return
        for path in pending:
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
            conn.commit()
            print(f"applied {path.name}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
