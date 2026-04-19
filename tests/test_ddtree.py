"""Tests for DDTree builder (Algorithm 1 of Ringel & Romano 2026)."""

from __future__ import annotations

import itertools
import math

import pytest

from mio.dflash.ddtree import build_ddtree, linear_trajectory_layout, enumerate_paths


def _enumerate_all_prefixes(top_k_tokens, top_k_logprobs, K):
    """Yield (score, ranks_tuple, prefix_token_ids) for all prefixes up to depth L using top-K per depth."""
    L = len(top_k_tokens)
    for d in range(1, L + 1):
        per_depth_ranges = [range(1, min(K, len(top_k_tokens[i])) + 1) for i in range(d)]
        for ranks in itertools.product(*per_depth_ranges):
            score = sum(top_k_logprobs[i][r - 1] for i, r in enumerate(ranks))
            tokens = tuple(top_k_tokens[i][r - 1] for i, r in enumerate(ranks))
            yield score, ranks, tokens


def _brute_force_topB(top_k_tokens, top_k_logprobs, B, K=None):
    """Reference: enumerate all prefixes and return top-B by score (prefix-closed)."""
    L = len(top_k_tokens)
    if K is None:
        K = min(B, len(top_k_tokens[0]))
    all_prefixes = list(_enumerate_all_prefixes(top_k_tokens, top_k_logprobs, K))
    all_prefixes.sort(key=lambda x: -x[0])

    # Enforce prefix closure: take highest-score prefixes, but include ancestors.
    # Proposition 2 says top-B by score is already prefix-closed under DDTree's
    # convention (log q < 0 means ancestor's score >= child's). Verify that.
    selected_ranks: set[tuple[int, ...]] = set()
    selected_tokens: list[tuple[int, ...]] = []
    for score, ranks, tokens in all_prefixes:
        if len(selected_ranks) >= B:
            break
        # Add ancestors if not already present
        for d in range(1, len(ranks) + 1):
            if ranks[:d] not in selected_ranks:
                if len(selected_ranks) >= B:
                    return selected_tokens
                selected_ranks.add(ranks[:d])
                # Corresponding tokens
                selected_tokens.append(tuple(tokens[:d]))
    return selected_tokens


def test_empty_budget():
    layout = build_ddtree([[1, 2, 3]], [[-0.1, -0.5, -1.0]], budget=0)
    assert layout.size == 0


def test_empty_distributions():
    layout = build_ddtree([], [], budget=100)
    assert layout.size == 0


def test_budget_one_returns_top_rank_position_one():
    top_tokens = [[42, 7, 13], [99, 1, 5]]
    top_logps = [[math.log(0.6), math.log(0.3), math.log(0.1)],
                 [math.log(0.7), math.log(0.2), math.log(0.1)]]
    layout = build_ddtree(top_tokens, top_logps, budget=1)
    assert layout.size == 1
    assert layout.token_ids == [42]
    assert layout.depth == [1]
    assert layout.parent_idx == [-1]


def test_linear_trajectory_matches_budget_equal_to_L():
    """With budget=L and a sharply peaked distribution, DDTree should produce the linear argmax path."""
    top_tokens = [[10, 20], [30, 40], [50, 60]]
    # Very peaked: rank-1 has p=0.99, rank-2 has p=0.01
    peaked = [math.log(0.99), math.log(0.01)]
    top_logps = [peaked, peaked, peaked]

    layout = build_ddtree(top_tokens, top_logps, budget=3)
    # Should spend all 3 nodes on the depth-1,2,3 rank-1 chain
    assert layout.size == 3
    assert layout.token_ids == [10, 30, 50]
    assert layout.depth == [1, 2, 3]
    assert layout.parent_idx == [-1, 0, 1]


def test_prefix_closure_invariant():
    """For any output, every node's parent must also be in the tree (or -1)."""
    import random
    random.seed(0)
    top_tokens = [[i * 10 + j for j in range(5)] for i in range(4)]
    # Randomish logprobs to stress-test
    top_logps = [[math.log(0.5 ** (k + 1)) for k in range(5)] for _ in range(4)]

    for B in [1, 3, 7, 15, 31, 63, 200]:
        layout = build_ddtree(top_tokens, top_logps, budget=B)
        in_tree_indices = set(range(layout.size))
        for i, p in enumerate(layout.parent_idx):
            assert p == -1 or p in in_tree_indices
            assert p == -1 or p < i, f"parent {p} must come before child {i}"


def test_top_B_by_prefix_score():
    """DDTree output tokens should match brute-force top-B enumeration."""
    top_tokens = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
    top_logps = [
        [math.log(0.5), math.log(0.3), math.log(0.2)],
        [math.log(0.6), math.log(0.3), math.log(0.1)],
        [math.log(0.4), math.log(0.35), math.log(0.25)],
    ]

    for B in [1, 3, 5, 10, 27]:
        layout = build_ddtree(top_tokens, top_logps, budget=B)
        # Reconstruct token-path tuples from layout
        paths: set[tuple[int, ...]] = set()
        for i in range(layout.size):
            path = []
            j = i
            while j != -1:
                path.append(layout.token_ids[j])
                j = layout.parent_idx[j]
            path.reverse()
            paths.add(tuple(path))

        brute = set(_brute_force_topB(top_tokens, top_logps, B, K=3))
        assert paths == brute, (
            f"B={B} mismatch:\n  ddtree={sorted(paths)}\n  brute={sorted(brute)}"
        )


def test_heavy_first_level_expansion_matches_brute_force():
    """DDTree should match brute-force even when depth-1 is near-uniform."""
    top_tokens = [[1, 2, 3, 4], [10, 20], [100, 200]]
    top_logps = [
        [math.log(0.26), math.log(0.25), math.log(0.25), math.log(0.24)],
        [math.log(0.99), math.log(0.01)],
        [math.log(0.99), math.log(0.01)],
    ]
    for B in [3, 5, 8, 12]:
        layout = build_ddtree(top_tokens, top_logps, budget=B)
        paths: set[tuple[int, ...]] = set()
        for i in range(layout.size):
            path = []
            j = i
            while j != -1:
                path.append(layout.token_ids[j])
                j = layout.parent_idx[j]
            path.reverse()
            paths.add(tuple(path))
        brute = set(_brute_force_topB(top_tokens, top_logps, B, K=4))
        assert paths == brute, f"B={B}: ddtree={sorted(paths)} vs brute={sorted(brute)}"


def test_linear_trajectory_layout_helper():
    layout = linear_trajectory_layout([5, 10, 15])
    assert layout.token_ids == [5, 10, 15]
    assert layout.parent_idx == [-1, 0, 1]
    assert layout.depth == [1, 2, 3]
    # ancestor_matrix for node 2 (depth-3) should include {0, 1, 2}
    assert layout.ancestor_matrix[2] == [0, 1, 2]


def test_enumerate_paths_linear_tree():
    """For a linear tree (budget=L on peaked dist), enumerate_paths returns a single path."""
    top_tokens = [[10, 20], [30, 40], [50, 60]]
    peaked = [math.log(0.99), math.log(0.01)]
    top_logps = [peaked, peaked, peaked]
    layout = build_ddtree(top_tokens, top_logps, budget=3)
    paths = enumerate_paths(layout, top_tokens, top_logps, max_paths=5)
    assert paths == [[10, 30, 50]]


def test_enumerate_paths_multipath_order():
    """Paths should be sorted by score descending."""
    top_tokens = [[1, 2], [10, 20]]
    # p=0.6 and p=0.4 at depth 1
    top_logps = [[math.log(0.6), math.log(0.4)], [math.log(0.99), math.log(0.01)]]
    layout = build_ddtree(top_tokens, top_logps, budget=4)
    paths = enumerate_paths(layout, top_tokens, top_logps, max_paths=4)
    # Leaves only: expect [1, 10] (score log(0.6)+log(0.99)) and [2, 10] (log(0.4)+log(0.99))
    # plus [1, 20] and [2, 20] if within budget.
    assert len(paths) >= 2
    assert paths[0] == [1, 10]  # highest-score leaf first
    assert paths[0][0] == 1


def test_enumerate_paths_respects_max_paths():
    top_tokens = [[i for i in range(4)] for _ in range(3)]
    top_logps = [[math.log(0.4), math.log(0.3), math.log(0.2), math.log(0.1)] for _ in range(3)]
    layout = build_ddtree(top_tokens, top_logps, budget=30)
    paths = enumerate_paths(layout, top_tokens, top_logps, max_paths=3)
    assert len(paths) == 3


def test_enumerate_paths_empty():
    layout = build_ddtree([], [], budget=10)
    assert enumerate_paths(layout, [], [], max_paths=5) == []


def test_ancestor_matrix_consistent_with_parent_idx():
    top_tokens = [[i for i in range(5)] for _ in range(4)]
    top_logps = [[math.log(p) for p in (0.5, 0.25, 0.15, 0.07, 0.03)] for _ in range(4)]
    layout = build_ddtree(top_tokens, top_logps, budget=50)
    for i in range(layout.size):
        # Walk parent chain manually
        chain = []
        j = i
        while j != -1:
            chain.append(j)
            j = layout.parent_idx[j]
        chain.reverse()
        assert layout.ancestor_matrix[i] == chain
