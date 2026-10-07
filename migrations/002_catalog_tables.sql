-- Phase 1 support: sitemap-derived catalog oracle + search-query universe.
-- catalog_videos mirrors the public video sitemap (id / slug / lastmod) and is
-- the coverage oracle for the crawl (how many of the library we hold).
-- catalog_queries mirrors the search-query sitemap (the fan-out term source).

CREATE TABLE IF NOT EXISTS catalog_videos (
    pexels_id       BIGINT PRIMARY KEY,
    slug            TEXT,
    lastmod         TIMESTAMPTZ,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_catalog_videos_lastmod ON catalog_videos (lastmod);

CREATE TABLE IF NOT EXISTS catalog_queries (
    term            TEXT PRIMARY KEY,
    lastmod         TIMESTAMPTZ,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_catalog_queries_lastmod ON catalog_queries (lastmod);
