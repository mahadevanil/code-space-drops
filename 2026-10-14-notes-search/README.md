# Semantic search over your own notes

Companion code for the October 14, 2026 Code Space BUILD post
(Instagram/LinkedIn: @code_space_official).

Turns a list of notes into embeddings (vectors that capture meaning,
not exact words), then searches them by cosine similarity instead of
keyword matching.

## Run it

```
pip install sentence-transformers numpy
python notes_search.py
```

Runs fully local and free via `sentence-transformers`. An OpenAI-backed
version (`text-embedding-3-small`) is included commented out in the file
if you'd rather call a hosted API.
