-- Production dense model moves to BAAI/bge-m3 (SiliconFlow), which emits
-- 1024-dim vectors and has no Matryoshka truncation (docs/research/
-- dense-embedding-models-2026-10-08.md). The column is still empty, so the
-- type change is an instantaneous metadata update.

ALTER TABLE stock_videos ALTER COLUMN embedding TYPE vector(1024);

COMMENT ON COLUMN stock_videos.embedding IS
    'dense vector (1024) from embedding_model (production: BAAI/bge-m3)';
