"""bfs_over_graph: pure bounded reachability over an already-built
networkx graph. No database -- tested directly against constructed
nx.DiGraph fixtures, including the deliberately cyclic one that proves
termination.
"""

import networkx as nx
import pytest

from codeqa.graph.traversal import bfs_over_graph


def linear_chain():
    g = nx.DiGraph()
    g.add_edges_from([(1, 2), (2, 3), (3, 4)])
    return g


def cyclic_with_tail():
    # 1 -> 2 -> 3 -> 1 (cycle), plus 3 -> 4 (a real node reachable through
    # the cycle before it repeats).
    g = nx.DiGraph()
    g.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4)])
    return g


class TestBasicReachability:
    def test_finds_all_nodes_within_depth(self):
        nodes = bfs_over_graph(linear_chain(), 1, "callees", max_depth=5)
        assert {n.chunk_id: n.depth for n in nodes} == {2: 1, 3: 2, 4: 3}

    def test_start_node_itself_excluded(self):
        nodes = bfs_over_graph(linear_chain(), 1, "callees", max_depth=5)
        assert all(n.chunk_id != 1 for n in nodes)

    def test_depth_bound_is_respected(self):
        nodes = bfs_over_graph(linear_chain(), 1, "callees", max_depth=2)
        assert {n.chunk_id for n in nodes} == {2, 3}

    def test_unreachable_node_never_appears(self):
        g = nx.DiGraph()
        g.add_edges_from([(1, 2)])
        g.add_node(99)  # present in the graph, but not reachable from 1
        nodes = bfs_over_graph(g, 1, "callees", max_depth=5)
        assert all(n.chunk_id != 99 for n in nodes)


class TestDirection:
    def test_callees_walks_forward(self):
        nodes = bfs_over_graph(linear_chain(), 1, "callees", max_depth=5)
        assert {n.chunk_id for n in nodes} == {2, 3, 4}

    def test_callers_walks_backward(self):
        nodes = bfs_over_graph(linear_chain(), 4, "callers", max_depth=5)
        assert {n.chunk_id for n in nodes} == {1, 2, 3}

    def test_callers_depths_are_correct(self):
        nodes = bfs_over_graph(linear_chain(), 4, "callers", max_depth=5)
        assert {n.chunk_id: n.depth for n in nodes} == {3: 1, 2: 2, 1: 3}


class TestCycleTermination:
    """The literal Phase 7 'done when' bar."""

    def test_terminates_on_a_cyclic_graph(self):
        # No timeout mechanism here -- if this doesn't terminate, the test
        # process hangs, which is itself the strongest possible failure
        # signal for a cycle-safety bug.
        nodes = bfs_over_graph(cyclic_with_tail(), 1, "callees", max_depth=10)
        assert nodes  # got here at all means it terminated

    def test_cycle_members_found_at_their_first_occurrence(self):
        nodes = bfs_over_graph(cyclic_with_tail(), 1, "callees", max_depth=10)
        by_id = {n.chunk_id: n.depth for n in nodes}
        assert by_id == {2: 1, 3: 2, 4: 3}
        # Node 1 is the start (excluded) even though the cycle revisits it
        # at depth 3 -- it must not appear a second time under a new depth.
        assert 1 not in by_id

    def test_large_depth_bound_does_not_cause_infinite_loop(self):
        # A depth bound far exceeding the cycle length is the real stress
        # case: naive "keep going until max_depth" logic without a visited
        # set would loop around the cycle repeatedly.
        nodes = bfs_over_graph(cyclic_with_tail(), 1, "callees", max_depth=1000)
        assert {n.chunk_id for n in nodes} == {2, 3, 4}


class TestEdgeCases:
    def test_unknown_start_node_returns_empty(self):
        assert bfs_over_graph(linear_chain(), 999, "callees", max_depth=5) == []

    def test_node_with_no_outgoing_edges_returns_empty(self):
        g = nx.DiGraph()
        g.add_edges_from([(1, 2)])
        assert bfs_over_graph(g, 2, "callees", max_depth=5) == []

    def test_zero_max_depth_returns_empty(self):
        nodes = bfs_over_graph(linear_chain(), 1, "callees", max_depth=0)
        assert nodes == []

    @pytest.mark.parametrize("direction", ["callees", "callers"])
    def test_self_loop_terminates(self, direction):
        g = nx.DiGraph()
        g.add_edge(1, 1)
        nodes = bfs_over_graph(g, 1, direction, max_depth=10)
        assert nodes == []  # the only "reachable" node is the start itself, excluded
