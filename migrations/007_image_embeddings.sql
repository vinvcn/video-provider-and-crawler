-- Lazy persisted image vectors for the MPT dedup gate (library-first matching
-- plan, ADR-0019). The gate used to embed every candidate thumbnail through
-- DashScope inside MPT; the retrieval service now owns that vector: cache-first
-- lookup here, one embed per new image on a miss.
--
-- Key scheme: `pexels:<id>` — the flat thumbnail dir shares the id, so a hit
-- is a pure DB lookup with no file read and no hashing. (key, model) is the
-- identity: vectors from different models never mix, and a model switch with
-- a different dimension is a new migration (see 006 for the precedent).
--
-- Dimension is fixed at 768 for the starter model tongyi-embedding-vision-flash;
-- the response reports len(vector) rather than persisting a redundant column.

CREATE TABLE IF NOT EXISTS image_embeddings (
    key         TEXT NOT NULL,
    model       TEXT NOT NULL,
    embedding   vector(768) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (key, model)
);

COMMENT ON TABLE image_embeddings IS
    'lazy image vectors; pexels:<id> keys, 768 dims, tongyi-embedding-vision-flash via DashScope';
COMMENT ON COLUMN image_embeddings.key IS 'image identity, e.g. pexels:8775004';
