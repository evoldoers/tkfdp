#!/usr/bin/env python3
"""
Uber-figure variant of tkf_trajectory.py.

Combines three components into ONE coordinated matplotlib figure:

    TOP    : SCFG parse tree (matplotlib re-render of tkf_trajectory.tex,
             with each L(t) terminal anchored at its trajectory column)
    MIDDLE : Galton-Watson indel trajectory (identical to tkf_trajectory.pdf
             minus the lottery row, since that ends up at the bottom)
    BOTTOM : lottery-ticket row with MATCH curve

All three panels share the same x-axis (column index), so the eye
can trace each non-terminal in the parse tree down to its trajectory
column and from there down to its lottery ticket.

Imports parsing/layout/drawing helpers from `tkf_trajectory.py` to
guarantee that the trajectory and lottery rendering is bit-identical
to the original figure.

Same canonical invocation:

    python tkf_trajectory_uber.py "M(G(G))ID(GG(I)G)" \
        --seed 99 --lottery-seed 99 \
        --output tkf_trajectory_uber.pdf
"""

import argparse
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import tkf_trajectory as tkftraj
from tkf_trajectory import (
    parse_input,
    set_parents,
    assign_columns,
    assign_times,
    flatten,
    draw_trajectory_on_ax,
    t_obs_expr,
    tau_expr,
)


# ---------------------------------------------------------------------------
# Build a parse-tree LAYOUT structure that mirrors tkf_trajectory.tex
# ---------------------------------------------------------------------------
# Each layout node is a dict:
#   {'label': str,           # math source (no $...$ wrappers)
#    'children': [layout],
#    'col': int or None}     # trajectory column if this node corresponds
#                            # to a single column (an L(...) or one of its
#                            # terminal children); None otherwise.
#
# The structure replicates link_forest / d_forest / s_forest in
# tkf_trajectory.py so that the parse tree is the EXACT same one
# rendered in tkf_trajectory.tex, just expressed as Python data.

def _link_layout(node):
    b = node['birth_idx']
    t_exp = t_obs_expr(b)
    L_label = f'L({t_exp})'
    if node['type'] in 'MI':
        terminal = 'A'
        T_loc = t_exp
    else:
        d = node['death_idx']
        terminal = 'G_{' + tau_expr(b, d) + '}'
        T_loc = tau_expr(b, d)
    terminal_leaf = {'label': terminal, 'children': [], 'col': node['col']}
    d_subtree = _d_layout(node['children'], t_exp, T_loc)
    return {'label': L_label,
            'children': [terminal_leaf, d_subtree],
            'col': node['col']}


def _d_layout(children, t_obs, T_loc):
    D_label = f'D({t_obs},\\, {T_loc})'
    if not children:
        eps = {'label': '\\varepsilon', 'children': [], 'col': None}
        return {'label': D_label, 'children': [eps], 'col': None}
    first, rest = children[0], children[1:]
    return {'label': D_label,
            'children': [_link_layout(first),
                         _d_layout(rest, t_obs, T_loc)],
            'col': None}


def _s_layout(forest):
    S_label = 'S'
    if not forest:
        eps = {'label': '\\varepsilon', 'children': [], 'col': None}
        return {'label': S_label, 'children': [eps], 'col': None}
    first, rest = forest[0], forest[1:]
    return {'label': S_label,
            'children': [_link_layout(first),
                         _s_layout(rest)],
            'col': None}


def build_parse_tree_layout(forest):
    """Top-level: return the root layout node."""
    if len(forest) == 1:
        return _link_layout(forest[0])
    return _s_layout(forest)


# ---------------------------------------------------------------------------
# Tree layout: assign (x, y) coordinates
# ---------------------------------------------------------------------------
# Strategy ("natural layout + guide lines"):
#
#   * Each leaf is given a unique x-slot in pre-order DFS, separated by
#     leaf_dx wide enough to fit the longest label without collision
#     (typical: 1.6-2.0 trajectory-column widths per leaf slot).
#   * Internal nodes are placed at the mean of their children's x.
#   * y is the depth (root at 0, deepest leaf at depth d), drawn
#     downward (we invert the y-axis on the Axes).
#   * Terminal leaves whose `col` is set (i.e. the A / G_tau atoms that
#     produce a trajectory column) additionally remember their
#     trajectory column; the draw_uber_figure function uses this to
#     draw a thin guide line from each such leaf down to the matching
#     trajectory column.
#
# This sacrifices "the leaves sit directly above the trajectory
# columns" in favour of (a) zero label-collision in the tree, and (b)
# guide lines do the eye-tracing job. With 9 trajectory columns and
# 18 parse-tree leaves, perfect column-leaf alignment would have
# forced unreadable label crowding.

def _depth(node):
    if not node['children']:
        return 0
    return 1 + max(_depth(c) for c in node['children'])


def _collect_leaves(node, lst=None):
    if lst is None: lst = []
    if not node['children']:
        lst.append(node)
    else:
        for c in node['children']:
            _collect_leaves(c, lst)
    return lst


def _assign_leaf_x_anchored(node, eps_step=0.45):
    """DFS leaf-x assignment that ANCHORS leaves with a `col` at
    their trajectory column x (i.e. lf.x = lf.col). Epsilon leaves
    are filled in between by uniform interpolation, with trailing
    ε's marching out at fixed eps_step.

    This is the layout used by `draw_uber_figure`: it guarantees
    that the eye can drop a vertical line from each terminal
    A / G_tau in the tree straight down to its trajectory column."""
    leaves = _collect_leaves(node)
    anchored_idx = [i for i, lf in enumerate(leaves)
                    if lf.get('col') is not None]
    if not anchored_idx:
        for i, lf in enumerate(leaves):
            lf['x'] = float(i + 1)
        return leaves
    # 1) Anchored leaves -> col.
    for i in anchored_idx:
        leaves[i]['x'] = float(leaves[i]['col'])
    # 2) Epsilons before first anchored.
    first_anc = anchored_idx[0]
    for i in range(first_anc):
        leaves[i]['x'] = leaves[first_anc]['x'] - eps_step * (first_anc - i)
    # 3) Epsilons between anchored neighbors: uniform interpolation.
    for a, b in zip(anchored_idx, anchored_idx[1:]):
        x_a, x_b = leaves[a]['x'], leaves[b]['x']
        gap = b - a
        for k, j in enumerate(range(a + 1, b), start=1):
            t = k / gap
            leaves[j]['x'] = x_a + t * (x_b - x_a)
    # 4) Trailing epsilons (after last anchored): march right at eps_step.
    last_anc = anchored_idx[-1]
    for k, j in enumerate(range(last_anc + 1, len(leaves)), start=1):
        leaves[j]['x'] = leaves[last_anc]['x'] + eps_step * k
    return leaves


def _max_col_in_subtree(node):
    """Return the largest 'col' among the descendants of node (or
    node itself). None if no descendant has a col."""
    if node.get('col') is not None:
        best = node['col']
    else:
        best = None
    for c in node['children']:
        sub = _max_col_in_subtree(c)
        if sub is not None and (best is None or sub > best):
            best = sub
    return best


def _layout_d_chain(d_node, owner_L_x, last_x):
    """Walk a D-chain (D → [L_sub, D_rest]  |  [ε]). owner_L_x = x of
    the L that owns this whole chain; last_x = rightmost x assigned
    so far in this chain (starts at owner_L_x).

    Layout rule (per the user's spec):
      - Spawning D (children[0] is an L-subtree): D.x = L_sub.x
        (i.e. the column of the L it spawns).
      - Terminating D (children[0] is ε): D.x = last_x + 0.5,
        i.e. midway between the rightmost descendant already placed
        and the next column over.
    """
    first = d_node['children'][0]
    if first['label'] == '\\varepsilon':
        # Terminating D: sits midway between rightmost-so-far and next col.
        d_node['x'] = last_x + 0.5
        first['x'] = d_node['x']
        return
    # Spawning D: first child is an L-subtree.
    L_sub = first
    _layout_L(L_sub)
    d_node['x'] = L_sub['x']
    sub_max = _max_col_in_subtree(L_sub)
    new_last = max(last_x, sub_max if sub_max is not None else last_x)
    _layout_d_chain(d_node['children'][1], owner_L_x, new_last)


def _layout_L(L_node):
    """L = [terminal_leaf, D_subtree]. L.x = terminal_leaf.col."""
    terminal = L_node['children'][0]
    if terminal.get('col') is not None:
        L_node['x'] = float(terminal['col'])
        terminal['x'] = float(terminal['col'])
    else:
        # Defensive: shouldn't happen, but fall back to the leaf's own x.
        L_node['x'] = terminal.get('x', 0.0)
    _layout_d_chain(L_node['children'][1], L_node['x'], L_node['x'])


def _layout_s_chain(s_node, last_x):
    """Walk an S-chain (S → [L_sub, S_rest]  |  [ε]) at the top level.
    Same rule as D-chain: spawning S sits over the spawned L's column;
    terminating S sits midway past the rightmost descendant."""
    first = s_node['children'][0]
    if first['label'] == '\\varepsilon':
        s_node['x'] = last_x + 0.5
        first['x'] = s_node['x']
        return
    L_sub = first
    _layout_L(L_sub)
    s_node['x'] = L_sub['x']
    sub_max = _max_col_in_subtree(L_sub)
    new_last = max(last_x, sub_max if sub_max is not None else last_x)
    _layout_s_chain(s_node['children'][1], new_last)


def _assign_internal_x(root):
    """Top-level x-assignment using the user's L / D rules.

    The root is either a single L (1 top-level lineage) or an S-chain
    (multiple top-level lineages). Either way, terminal leaves with
    a col already have their x; this function walks the tree and
    fills in internal-node x's so that each L sits directly above
    its lineage column, each spawning D sits over the spawned L,
    and each terminating D sits one half-column past the rightmost
    descendant placed so far."""
    if root['label'].startswith('L'):
        _layout_L(root)
    elif root['label'] == 'S':
        # Top-level S-chain. Initial last_x = the first spawned L's col
        # (or 0 if the very first S terminates in ε, which shouldn't
        # happen for a non-empty forest).
        first = root['children'][0]
        if first['label'] == '\\varepsilon':
            root['x'] = 0.5
            first['x'] = 0.5
        else:
            _layout_L(first)
            root['x'] = first['x']
            _layout_s_chain(root['children'][1], first['x'])
    else:
        # Defensive: fall back to mean-of-children.
        for c in root['children']:
            _assign_internal_x(c)
        if root['children']:
            root['x'] = sum(c['x'] for c in root['children']) / len(root['children'])
    return root['x']


def _assign_y(node, depth=0):
    node['y'] = depth
    for c in node['children']:
        _assign_y(c, depth + 1)


def _spread_terminating_ds(root, max_spread=0.4):
    """Spread terminating D / S nodes horizontally within their
    integer-column slot to avoid collisions.

    All terminating Ds (and the terminating S) in the same column gap
    initially land at the same `last_x + 0.5`. This pass collects them,
    groups by that gap, and spreads each group across [c + 0.5 - h,
    c + 0.5 + h] with h = max_spread/2. Ordering: oldest (smallest
    depth = closest to root) goes furthest RIGHT inside the gap;
    deepest (most recent) goes furthest LEFT. This preserves the
    "children right of parents" convention.

    Epsilon children of these Ds move along with their parents, so
    they remain immediately under the D.
    """
    terms = []

    def walk(node):
        ch = node.get('children') or []
        if ch and ch[0].get('label') == '\\varepsilon':
            terms.append(node)
        for c in ch:
            walk(c)

    walk(root)

    from collections import defaultdict
    by_col = defaultdict(list)
    for t in terms:
        col_key = int(round(t['x'] - 0.5))
        by_col[col_key].append(t)

    for col, group in by_col.items():
        if len(group) <= 1:
            continue
        group.sort(key=lambda n: n['y'])  # shallow (oldest) first
        K = len(group)
        x_right = col + 0.5 + max_spread / 2.0
        x_left = col + 0.5 - max_spread / 2.0
        for k, node in enumerate(group):
            new_x = x_right - k * (max_spread / (K - 1))
            node['x'] = new_x
            for child in node.get('children', []):
                if child.get('label') == '\\varepsilon':
                    child['x'] = new_x


def layout_parse_tree(root):
    """Anchor parse-tree leaves at their trajectory column. Returns
    the list of leaves in DFS pre-order. Internal-node x is set by
    _assign_internal_x, which places L over its lineage column,
    spawning D over its spawned L, and terminating D one half-column
    past the rightmost descendant. _spread_terminating_ds then
    spreads colliding terminating Ds horizontally within each column
    gap (older further right, newer further left)."""
    leaves = _assign_leaf_x_anchored(root)
    _assign_internal_x(root)
    _assign_y(root)
    _spread_terminating_ds(root)
    return leaves


def _is_terminal_leaf(node):
    """True if node is a leaf (no children)."""
    return not node['children']


# ---------------------------------------------------------------------------
# Label formatting for the parse tree
# ---------------------------------------------------------------------------
# Internal nodes carry long expressions like "D(T - t_{1},\, t_{2} - t_{1})"
# that don't fit at a column-wide x-spacing. We break them onto two lines
# at the comma inside D(...). L(...) labels stay on one line.

def _format_node_label(raw_label):
    """Return a math-mode string for the tree node label. Long D(...)
    labels are split at the comma to a two-line stack."""
    if raw_label.startswith('D(') and ',\\,' in raw_label:
        # 'D(t1,\\, t2)' -> 'D(t1,' on top, '\\, t2)' on bottom
        head, tail = raw_label.split(',\\,', 1)
        # Use \\substack to stack two lines in mathmode.
        return r'$\substack{' + head + r', \\ ' + tail.strip() + '}$'
    return rf'${raw_label}$'


# ---------------------------------------------------------------------------
# Render the parse tree onto an Axes
# ---------------------------------------------------------------------------

def draw_parse_tree_on_ax(ax, root, *, font_size=7.5, arrow_color='0.25'):
    """Render the laid-out parse tree onto ax. Caller sets xlim/ylim."""
    def walk(node):
        for c in node['children']:
            ax.annotate(
                '', xy=(c['x'], c['y']), xytext=(node['x'], node['y']),
                arrowprops=dict(arrowstyle='-|>', mutation_scale=6,
                                color=arrow_color, lw=0.6,
                                shrinkA=8, shrinkB=8),
                zorder=2,
            )
            walk(c)
        ax.text(node['x'], node['y'],
                _format_node_label(node['label']),
                ha='center', va='center',
                fontsize=font_size,
                zorder=3)

    walk(root)


# ---------------------------------------------------------------------------
# Master draw
# ---------------------------------------------------------------------------

def draw_uber_figure(forest, n_events, output_path, lottery_seed=None):
    """Compose parse tree (top) + trajectory (middle) + lottery (bottom)
    into ONE matplotlib figure with a single shared x-axis. Each terminal
    A / G_tau leaf in the parse tree sits directly above its trajectory
    column; epsilon leaves are placed mid-way between anchored neighbors
    (or marched right past the rightmost anchored leaf) so they don't
    distort the column-aligned spine."""
    nodes = flatten(forest)
    max_col = max(n['col'] for n in nodes)

    # 1) Build + lay out the parse tree.
    root = build_parse_tree_layout(forest)
    leaves = layout_parse_tree(root)
    tree_depth = _depth(root)
    tree_x_max = max(lf['x'] for lf in leaves)

    # 2) Sizing.
    #    1 inch per trajectory column.  The tree shares this scale;
    #    its rightmost leaf sits at tree_x_max ~ max_col + a small
    #    overflow from trailing ε's. We accommodate the overflow in
    #    fig width.
    inch_per_col = 1.55   # wider columns: trees labels (D(...), L(...))
                          # at deep levels need more horizontal room
                          # than the trajectory's MIDG markers. ~15%
                          # bump tidies remaining label collisions
                          # without forcing a noticeably bigger fig.
    margin_left = 1.4
    margin_right = 0.5
    panel_w = max(max_col, tree_x_max) * inch_per_col + 1.5
    fig_w = margin_left + panel_w + margin_right

    # Heights: tuned so the figure fills a portrait letter [p] float
    # at `width=\textwidth,keepaspectratio`. Letter ink area is 6.5"
    # x 9.5", aspect ~0.68. With panel_w ~ 13.5" (set above), we need
    # fig_h ~ 13.5/0.68 = ~19.8" to match. Capping at 19" keeps the
    # parse tree LEGIBLE: each tree depth-step is generous (0.55"+),
    # the trajectory is comfortable, and the figure naturally
    # occupies a full page once LaTeX scales it down to fit width.
    h_tree = max(8.0, 0.55 * (tree_depth + 1) + 1.0)
    h_traj = max(5.0, 1.4 + 0.50 * (n_events + 1))
    h_lottery = 3.0 if lottery_seed is not None else 0.0
    fig_h = h_tree + h_traj + h_lottery

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        2, 1,
        height_ratios=[h_tree, h_traj + h_lottery],
        hspace=0.06,
        figure=fig,
    )
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_traj = fig.add_subplot(gs[1, 0])

    # 3) Draw parse tree. Shared x-axis with trajectory. xlim starts at
    # 0.5 (one half-column left of col 1) since the immortal-link gutter
    # at col 0 has been reclaimed.
    draw_parse_tree_on_ax(ax_tree, root)
    ax_tree.set_xlim(0.5, max(max_col + 0.5, tree_x_max + 0.25))
    ax_tree.set_ylim(-0.6, tree_depth + 0.6)
    ax_tree.invert_yaxis()
    ax_tree.set_xticks([]); ax_tree.set_yticks([])
    for side in ('left', 'right', 'top', 'bottom'):
        ax_tree.spines[side].set_visible(False)

    # 4) Draw trajectory + lottery on shared x-scale. Suppress the
    # immortal-link dashed line at col 0 so the freed horizontal space
    # can be reclaimed by the parse-tree panel.
    max_col2, Y_TOP, _Y_BOT, lottery_bottom = draw_trajectory_on_ax(
        ax_traj, forest, n_events,
        lottery_seed=lottery_seed,
        column_xticks_top=True,
        show_column_labels=True,
        show_immortal_link=False,
    )
    assert max_col2 == max_col
    ax_traj.set_xlim(0.5, max(max_col + 0.5, tree_x_max + 0.25))
    ax_traj.set_ylim(Y_TOP - 0.4, lottery_bottom)
    ax_traj.invert_yaxis()

    # 5) Optional: faint vertical column guides through both panels.
    # These help the eye trace each tree leaf down to the trajectory
    # column directly below it. Only the trajectory columns get a
    # guide line; epsilon-only x-positions don't.
    for col in range(1, max_col + 1):
        for ax_ in (ax_tree, ax_traj):
            ax_.axvline(x=col, color='0.85', linewidth=0.5,
                        linestyle=(0, (2, 2)), zorder=0)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Uber-figure variant of tkf_trajectory.py: stacks '
                    'parse tree (top), Galton-Watson trajectory (middle), '
                    'and lottery-ticket row (bottom) into a single PDF.')
    p.add_argument('sequence',
                   help='Tree-decorated sequence string '
                        '(e.g. "M(G(G))ID(GG(I)G)").')
    p.add_argument('-o', '--output', default='tkf_trajectory_uber.pdf',
                   help='Output figure path (default: '
                        'tkf_trajectory_uber.pdf). A .png mirror is also '
                        'saved alongside.')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed for time ordering (default: '
                        'nondeterministic).')
    p.add_argument('--lottery-seed', type=int, default=None,
                   help='RNG seed for lottery-ticket IDs and MATCH pair. '
                        'Default: no lottery row.')
    p.add_argument('--label-order', choices=['spatial', 'temporal'],
                   default='spatial',
                   help="t_k subscript convention (see tkf_trajectory.py).")
    args = p.parse_args()

    forest = parse_input(args.sequence)
    if not forest:
        print('Empty or invalid input.', file=sys.stderr)
        sys.exit(1)
    set_parents(forest)
    assign_columns(forest)
    rng = random.Random(args.seed)
    n_events = assign_times(forest, rng, label_order=args.label_order)

    # Save the user-requested file (PDF or PNG; both stored).
    out_path = args.output
    if out_path.lower().endswith('.pdf'):
        png_path = out_path[:-4] + '.png'
    elif out_path.lower().endswith('.png'):
        png_path = None  # the requested file is already PNG
    else:
        png_path = out_path + '.png'

    draw_uber_figure(forest, n_events, out_path,
                     lottery_seed=args.lottery_seed)
    print(f'Figure written to {out_path}', file=sys.stderr)

    if png_path is not None:
        draw_uber_figure(forest, n_events, png_path,
                         lottery_seed=args.lottery_seed)
        print(f'PNG mirror written to {png_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
