"""Cherry extraction from a phylogenetic tree.

Algorithm: smallest-combined-branch-length first, iteratively prune the cherry,
turning the parent into a leaf. Returns (leaf_name_a, leaf_name_b, tau) tuples
where tau = bl_a + bl_b. Adapted from
~/tkf-mixdom/python/build_tkf92_cherry_counts.py.
"""

from __future__ import annotations

from .bio import Node


def extract_cherries(root: Node) -> list[tuple[str, str, float]]:
    """Iteratively pick cherries (two-leaf siblings) by smallest combined
    branch length, prune both leaves (parent becomes a leaf), and repeat.

    Returns list of (leaf_a, leaf_b, tau) where tau = bl_a + bl_b.
    """
    cherries: list[tuple[str, str, float]] = []

    while True:
        candidates: list[tuple[float, Node]] = []  # (combined_bl, parent)
        stack = [root]
        while stack:
            node = stack.pop()
            if node.children:
                leaf_kids = [c for c in node.children if c.is_leaf()]
                if len(node.children) == 2 and len(leaf_kids) == 2:
                    bl = leaf_kids[0].branch_length + leaf_kids[1].branch_length
                    candidates.append((bl, node))
                stack.extend(node.children)

        if not candidates:
            # Multifurcating root with all-leaf children: pick the closest
            # pair as a cherry, remove them.
            if (root.children and all(c.is_leaf() for c in root.children)
                    and len(root.children) >= 2):
                pairs = []
                ch = root.children
                for i in range(len(ch)):
                    for j in range(i + 1, len(ch)):
                        pairs.append(
                            (ch[i].branch_length + ch[j].branch_length, i, j)
                        )
                if pairs:
                    pairs.sort(key=lambda x: x[0])
                    bl, i, j = pairs[0]
                    cherries.append((ch[i].name, ch[j].name, bl))
                    new_children = [c for k, c in enumerate(ch) if k != i and k != j]
                    root.children = new_children
                    if len(root.children) <= 1:
                        break
                    continue
            break

        candidates.sort(key=lambda x: x[0])
        bl, parent = candidates[0]
        a, b = parent.children[0], parent.children[1]
        cherries.append((a.name, b.name, bl))
        parent.children = []
        parent.name = parent.name or f"_internal_{id(parent)}"
        if parent is root:
            break

    return cherries
