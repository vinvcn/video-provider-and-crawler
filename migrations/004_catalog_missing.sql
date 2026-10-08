-- Gap-fill hardening: remember catalog rows whose video is gone.
-- The per-ID data route answers with a `?missing_medium` redirect for deleted or
-- unlisted videos (HTTP 200, no `pageProps.medium`). The crawler flags those in
-- the spool, ingest marks them here, and gap queries skip them so a batch cannot
-- loop forever on dead IDs (2026-10-09 head-of-line stall).

ALTER TABLE catalog_videos
    ADD COLUMN IF NOT EXISTS missing_since TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_catalog_videos_missing_since
    ON catalog_videos (pexels_id) WHERE missing_since IS NOT NULL;

COMMENT ON COLUMN catalog_videos.missing_since IS
    'video gone (deleted/unlisted) as of this time; excluded from gap-fill worklists';
