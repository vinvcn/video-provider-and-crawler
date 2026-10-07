-- Phase 2 groundwork: embedding columns on stock_videos.
-- Dimension is fixed at 768 for the documented starter model
-- (tongyi-embedding-vision-flash, docs/handoff-2026-10-07.md §5); a model switch
-- with a different dimension is a new migration (ALTER TYPE / new column) while
-- embedding_model lets two models coexist during the transition.
-- Indexes (HNSW / GIN) are deliberately deferred to a later migration: building
-- them while the crawl is still ingesting would tax every write.

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE stock_videos
    ADD COLUMN IF NOT EXISTS embed_text       TEXT,
    ADD COLUMN IF NOT EXISTS embedding        vector(768),
    ADD COLUMN IF NOT EXISTS embedding_model  TEXT,
    ADD COLUMN IF NOT EXISTS embedded_at      TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS sparse_embedding sparsevec,
    ADD COLUMN IF NOT EXISTS sparse_model     TEXT,
    ADD COLUMN IF NOT EXISTS tsv              tsvector;

COMMENT ON COLUMN stock_videos.embed_text IS 'canonical text used for embedding (title + description + tags)';
COMMENT ON COLUMN stock_videos.embedding IS 'dense vector (768) from embedding_model';
COMMENT ON COLUMN stock_videos.sparse_embedding IS 'sparse vector (lexical model, e.g. BM25) from sparse_model';
