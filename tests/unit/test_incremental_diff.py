"""_diff_file_chunks and _map_chunks_to_ids: pure, no database. The
duplicate-content_sha case is the one worth pinning directly here rather
than trusting the integration suite to happen to exercise it -- two
distinct definitions with byte-identical text are rare but real (e.g. two
trivially identical stub methods), and content_sha alone can't tell them
apart.
"""

from codeqa.indexing.chunker import Chunk
from codeqa.indexing.incremental import _diff_file_chunks, _map_chunks_to_ids


def make_chunk(symbol_name: str, content_sha: str) -> Chunk:
    return Chunk(
        kind="function",
        symbol_name=symbol_name,
        qualified_name=None,
        start_byte=0,
        end_byte=1,
        start_line=1,
        end_line=1,
        content=f"def {symbol_name}(): ...",
        content_sha=content_sha,
    )


class TestDiffFileChunks:
    def test_a_chunk_present_in_both_is_neither_removed_nor_added(self):
        old = {"sha-a": [1]}
        removed, added = _diff_file_chunks(old, [make_chunk("f", "sha-a")])
        assert removed == []
        assert added == []

    def test_a_new_content_sha_is_added(self):
        removed, added = _diff_file_chunks({}, [make_chunk("f", "sha-new")])
        assert removed == []
        assert [c.symbol_name for c in added] == ["f"]

    def test_a_content_sha_no_longer_present_is_removed(self):
        old = {"sha-gone": [7]}
        removed, added = _diff_file_chunks(old, [])
        assert removed == [7]
        assert added == []

    def test_duplicate_content_sha_pairs_up_positionally_not_all_to_one_id(self):
        # Two old chunks shared a hash (byte-identical definitions); two new
        # chunks also share it. Both old ids should be consumed, not one id
        # reused for both and the other left dangling as "removed".
        old = {"sha-dup": [10, 11]}
        new_chunks = [make_chunk("stub_a", "sha-dup"), make_chunk("stub_b", "sha-dup")]
        removed, added = _diff_file_chunks(old, new_chunks)
        assert removed == []
        assert added == []

    def test_more_new_duplicates_than_old_only_the_excess_is_added(self):
        old = {"sha-dup": [10]}
        new_chunks = [make_chunk("stub_a", "sha-dup"), make_chunk("stub_b", "sha-dup")]
        removed, added = _diff_file_chunks(old, new_chunks)
        assert removed == []
        assert [c.symbol_name for c in added] == ["stub_b"]

    def test_more_old_duplicates_than_new_only_the_excess_is_removed(self):
        old = {"sha-dup": [10, 11]}
        new_chunks = [make_chunk("stub_a", "sha-dup")]
        removed, added = _diff_file_chunks(old, new_chunks)
        assert removed == [11]
        assert added == []


class TestMapChunksToIds:
    def test_maps_each_chunk_to_its_existing_id(self):
        c = make_chunk("f", "sha-a")
        result = _map_chunks_to_ids([c], {"sha-a": [42]})
        assert result == {id(c): 42}

    def test_a_chunk_with_no_matching_id_is_omitted(self):
        c = make_chunk("f", "sha-missing")
        assert _map_chunks_to_ids([c], {}) == {}

    def test_duplicate_content_sha_maps_each_chunk_to_a_distinct_id(self):
        c1 = make_chunk("stub_a", "sha-dup")
        c2 = make_chunk("stub_b", "sha-dup")
        result = _map_chunks_to_ids([c1, c2], {"sha-dup": [10, 11]})
        assert result == {id(c1): 10, id(c2): 11}
