-- Phase 1: canonical store for crawled Pexels video metadata.
-- One row per Pexels video (pexels_id is the natural, idempotent upsert key).
-- Vector columns arrive in Phase 2 (separate migration).

CREATE TABLE IF NOT EXISTS stock_videos (
    pexels_id       BIGINT PRIMARY KEY,
    slug            TEXT,
    title           TEXT,
    description     TEXT,
    tags            TEXT[] NOT NULL DEFAULT '{}',
    duration        INTEGER,
    width           INTEGER,
    height          INTEGER,
    fps             NUMERIC,
    aspect_ratio    NUMERIC,
    orientation     TEXT,
    license         TEXT,
    user_name       TEXT,
    created_at      TIMESTAMPTZ,
    video_files     JSONB NOT NULL DEFAULT '[]',
    thumbnail_url   TEXT,
    thumbnail_path  TEXT,
    download_link   TEXT,
    source          TEXT NOT NULL DEFAULT 'pexels',
    raw             JSONB,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stock_videos_tags ON stock_videos USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_stock_videos_duration ON stock_videos (duration);
CREATE INDEX IF NOT EXISTS idx_stock_videos_orientation ON stock_videos (orientation);
CREATE INDEX IF NOT EXISTS idx_stock_videos_thumbnail_pending
    ON stock_videos (pexels_id) WHERE thumbnail_path IS NULL;
