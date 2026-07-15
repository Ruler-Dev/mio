"""DDTree (Diffusion Draft Tree) — Algorithm 1 of Ringel & Romano 2026.

Build the optimal draft tree of up to B prefix-closed tokens from one block
diffusion draft forward pass. Pure-Python; no MLX dependency.

Usage:
    layout = build_ddtree(top_k_token_ids, top_k_logprobs, budget=256)
    # layout.token_ids: list[int], length |T|
    # layout.parent_idx: list[int]  (parent index in layout.token_ids, -1 for root)
    # layout.depth:     list[int]   (1..L)
    # layout.ancestor_matrix: list[list[int]] — for each node, the list of indices in the tree
                              on its ancestor path (root excluded), including itself.

The root (bonus token) is NOT included in the layout; it is the fixed prefix
on which the tree hangs.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass


@dataclass
class TreeLayout:
    token_ids: list[int]
    parent_idx: list[int]        # -1 means "root is parent"
    depth: list[int]             # 1..L
    ancestor_matrix: list[list[int]]  # each entry: node indices on path (self-inclusive)
    children: list[list[int]]    # per-node: indices of children

    @property
    def size(self) -> int:
        return len(self.token_ids)


def build_ddtree(
    top_k_tokens: list[list[int]],
    top_k_logprobs: list[list[float]],
    budget: int,
) -> TreeLayout:
    """Best-first construction of the optimal draft tree.

    Args:
        top_k_tokens: per-depth list of the top-K token IDs, length L; each
            inner list is of length K (descending by prob).
        top_k_logprobs: same shape as top_k_tokens; log q_i^{(k)}.
        budget: node budget B (root not counted).

    Returns:
        TreeLayout whose token_ids list has length min(budget, sum of enumerable prefixes).
    """
    if budget <= 0:
        return TreeLayout([], [], [], [], [])
    if not top_k_tokens:
        return TreeLayout([], [], [], [], [])

    L = len(top_k_tokens)
    K = min(len(top_k_tokens[0]), budget) if top_k_tokens else 0
    # Paper's K = min(B, |V|); we bound it by the provided top-K width.

    # Heap entries: (neg_score, ranks_tuple, parent_index_in_T)
    # parent_index_in_T: index in the output list of the parent node (-1 if root).
    # Python's heapq is a min-heap — we push negated scores.

    # Initial rank tuple (1,) → first child of root, with score = log q_1^{(1)}.
    init_score = top_k_logprobs[0][0]
    heap: list[tuple[float, tuple[int, ...], int]] = []
    heapq.heappush(heap, (-init_score, (1,), -1))

    token_ids: list[int] = []
    parent_idx: list[int] = []
    depth: list[int] = []
    # Map from rank-tuple -> index in T once added.
    index_by_rank: dict[tuple[int, ...], int] = {}

    while heap and len(token_ids) < budget:
        neg_score, ranks, parent = heapq.heappop(heap)
        score = -neg_score
        d = len(ranks)
        if d > L:
            continue

        # Resolve parent index from the rank tuple's prefix.
        if d == 1:
            parent_in_T = -1
        else:
            parent_ranks = ranks[:-1]
            parent_in_T = index_by_rank.get(parent_ranks)
            if parent_in_T is None:
                # Parent not yet added — Algorithm 1's best-first order guarantees
                # each prefix is popped before its children, so this should only happen
                # if the parent was itself filtered for budget reasons.
                continue

        idx_in_T = len(token_ids)
        last_rank = ranks[-1]  # 1-based rank
        token_id = top_k_tokens[d - 1][last_rank - 1]

        token_ids.append(token_id)
        parent_idx.append(parent_in_T)
        depth.append(d)
        index_by_rank[ranks] = idx_in_T

        # Push sibling: (ranks[:-1] + (last_rank + 1,))
        if last_rank < K and last_rank < len(top_k_logprobs[d - 1]):
            sibling_score = (
                score
                - top_k_logprobs[d - 1][last_rank - 1]
                + top_k_logprobs[d - 1][last_rank]
            )
            sibling_ranks = ranks[:-1] + (last_rank + 1,)
            heapq.heappush(heap, (-sibling_score, sibling_ranks, parent_in_T))

        # Push first child: (ranks + (1,))
        if d < L and len(top_k_logprobs[d]) > 0:
            child_score = score + top_k_logprobs[d][0]
            child_ranks = ranks + (1,)
            heapq.heappush(heap, (-child_score, child_ranks, idx_in_T))

    # Build ancestor_matrix + children adjacency.
    children: list[list[int]] = [[] for _ in token_ids]
    ancestor_matrix: list[list[int]] = []
    for i, p in enumerate(parent_idx):
        # ancestors list: self + parent chain until root
        chain: list[int] = []
        j = i
        while j != -1:
            chain.append(j)
            j = parent_idx[j]
        chain.reverse()  # root-nearest first
        ancestor_matrix.append(chain)
        if p != -1:
            children[p].append(i)

    return TreeLayout(
        token_ids=token_ids,
        parent_idx=parent_idx,
        depth=depth,
        ancestor_matrix=ancestor_matrix,
        children=children,
    )


def enumerate_paths(
    layout: TreeLayout,
    top_k_tokens: list[list[int]],
    top_k_logprobs: list[list[float]],
    max_paths: int,
) -> list[list[int]]:
    """Return root-to-leaf token-id paths, sorted descending by sum-of-logprobs.

    A node is a "leaf" if it has no children in the layout. The path is the
    chain from root (excluded — the root is the bonus token handled by caller)
    down to that leaf.

    Args:
        layout:        Output of `build_ddtree`.
        top_k_tokens:  Same per-depth top-K tokens passed to build_ddtree.
        top_k_logprobs:Same per-depth logprobs passed to build_ddtree.
        max_paths:     Return at most this many paths.

    Returns:
        list of token-id sequences (variable length, one per leaf), length ≤ max_paths.
    """
    if layout.size == 0 or max_paths <= 0:
        return []

    # Pre-invert top_k_tokens to token -> rank per depth for O(1) lookup.
    per_depth_rank: list[dict[int, int]] = [
        {tok: r for r, tok in enumerate(depth_tokens)} for depth_tokens in top_k_tokens
    ]

    scored: list[tuple[float, list[int]]] = []
    for i in range(layout.size):
        if layout.children[i]:
            continue
        chain_indices = layout.ancestor_matrix[i]
        tokens = [layout.token_ids[j] for j in chain_indices]
        score = 0.0
        for depth_idx, tok in enumerate(tokens):
            d = depth_idx  # 0-based into top_k_*
            if d >= len(top_k_logprobs):
                break
            rank = per_depth_rank[d].get(tok, 0)
            score += top_k_logprobs[d][rank]
        scored.append((score, tokens))

    scored.sort(key=lambda x: -x[0])
    return [tokens for _, tokens in scored[:max_paths]]


def linear_trajectory_layout(argmax_tokens: list[int]) -> TreeLayout:
    """Degenerate tree = the single argmax trajectory (equivalent to vanilla DFlash)."""
    n = len(argmax_tokens)
    parent_idx = [i - 1 for i in range(n)]  # chain; first has -1
    parent_idx[0] = -1
    depth = list(range(1, n + 1))
    ancestor_matrix = [[j for j in range(0, i + 1)] for i in range(n)]
    children = [[i + 1] if i + 1 < n else [] for i in range(n)]
    return TreeLayout(
        token_ids=list(argmax_tokens),
        parent_idx=parent_idx,
        depth=depth,
        ancestor_matrix=ancestor_matrix,
        children=children,
    )
