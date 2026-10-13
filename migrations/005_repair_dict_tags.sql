-- Repair tags stored as stringified tag objects (2026-10-11 discovery).
--
-- Some v3 payloads return tags as {"name", "search_term"} objects; the old
-- normalizer ran str() on them, producing "{'name': ...}" strings that pollute
-- BM25 and the dense embed text (188,942 rows affected). The raw payload kept
-- the original array, so this rebuilds the column without a re-crawl.
-- Order-preserving and idempotent: re-running matches no rows once clean.

UPDATE stock_videos
SET tags = COALESCE(
        (
            SELECT array_agg(part.tag ORDER BY part.ord)
            FROM (
                SELECT CASE
                           WHEN jsonb_typeof(elem) = 'object' THEN NULLIF(elem->>'name', '')
                           WHEN jsonb_typeof(elem) = 'string' THEN NULLIF(elem#>>'{}', '')
                           ELSE NULL
                       END AS tag,
                       ord
                FROM jsonb_array_elements(raw->'tags') WITH ORDINALITY AS e(elem, ord)
            ) AS part
            WHERE part.tag IS NOT NULL
        ),
        '{}'
    )
WHERE jsonb_typeof(raw->'tags') = 'array'
  AND EXISTS (SELECT 1 FROM unnest(tags) AS t(tag) WHERE t.tag LIKE '{%');
