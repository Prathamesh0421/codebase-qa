"""Single-shot dense retrieval: embed the query, rank by cosine similarity,
take the top k. No lexical search, no symbol lookup, no call-graph expansion.

This is the eval baseline naive RAG applied to code actually looks like, and
it is deliberately not made any smarter than that -- the entire point is to
have something honest to measure hybrid and hybrid+graph against later.
"""

import psycopg

from codeqa.indexing.embeddings import EmbeddingProvider
from codeqa.retrieval.strategy import RetrievedChunk

# score = 1 - cosine_distance, i.e. pgvector's <=> operator subtracted from 1.
# <=> returns 0 for identical vectors and up to 2 for opposite ones, so this
# isn't a proper similarity bounded to [0, 1] in the general case -- but for
# normalized embeddings (bge-small-en-v1.5 and most sentence-transformers
# models are) it lands in [-1, 1] with 1 meaning identical, which is the
# conventional cosine-similarity reading.
_QUERY = """
    SELECT c.id, f.path, c.kind, c.symbol_name, c.qualified_name,
           c.start_line, c.end_line, c.content,
           1 - (c.embedding <=> %(qvec)s::vector) AS score
      FROM chunks c
      JOIN files f ON f.id = c.file_id AND f.repo_id = c.repo_id
     WHERE c.repo_id = %(repo_id)s
     ORDER BY c.embedding <=> %(qvec)s::vector
     LIMIT %(top_k)s
"""


class NaiveStrategy:
    def retrieve(
        self,
        conn: psycopg.Connection,
        repo_id: int,
        query_text: str,
        embedder: EmbeddingProvider,
        top_k: int,
    ) -> list[RetrievedChunk]:
        query_vector = embedder.embed([query_text])[0]

        with conn.cursor() as cur:
            cur.execute(_QUERY, {"qvec": query_vector, "repo_id": repo_id, "top_k": top_k})
            rows = cur.fetchall()

        return [
            RetrievedChunk(
                chunk_id=row[0],
                file_path=row[1],
                kind=row[2],
                symbol_name=row[3],
                qualified_name=row[4],
                start_line=row[5],
                end_line=row[6],
                content=row[7],
                score=row[8],
            )
            for row in rows
        ]
