#!/usr/bin/env python3
"""
TKF91 / TKF92 Galton-Watson trajectory visualizer.

Parses a gravestone-augmented, tree-decorated output string such as

    M(G(G))ID(GG(I)G)

and produces

  (1) a matplotlib figure depicting the indel trajectory in the style of
      Holmes' Fig. 3 (a Galton-Watson branching process correspondence
      between a pairwise alignment and an indel history), and

  (2) a LaTeX `forest`-package snippet of the corresponding SCFG parse
      tree, with each non-terminal annotated by its time arguments
      expressed in the symbolic event times t_1, t_2, ... that label
      the vertical axis of the figure.

Symbol legend (per Ian's spec):
    M = surviving ancestral marker        (alive at T, born at 0)
    I = surviving insertion marker        (alive at T, born at t_b > 0)
    D = gravestone for ancestral site     (born at 0, died at t_d < T)
    G = gravestone for transient insert   (born at t_b > 0, died t_d > t_b)

Children inside parentheses are taken to be listed in left-to-right
alignment order, which (in this implementation) is also their order in
the temporal sequence of events.  Each child of a node has its birth
event registered at a t_i strictly later than the parent's birth, and
all events inside a node's subtree occur strictly before that node's
death event (if any).  Surviving children are allowed to outlive their
parents simply by not having a death event of their own (their lineage
extends all the way to T).
"""

import argparse
import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_input(s):
    """Parse a tree-decorated sequence string into a forest of nodes."""
    pos = [0]

    def node():
        t = s[pos[0]]
        pos[0] += 1
        children = []
        if pos[0] < len(s) and s[pos[0]] == '(':
            pos[0] += 1  # consume '('
            while pos[0] < len(s) and s[pos[0]] != ')':
                if s[pos[0]] in 'MIDG':
                    children.append(node())
                else:
                    pos[0] += 1  # skip whitespace etc.
            if pos[0] < len(s) and s[pos[0]] == ')':
                pos[0] += 1  # consume ')'
        return {'type': t, 'children': children}

    forest = []
    while pos[0] < len(s):
        if s[pos[0]] in 'MIDG':
            forest.append(node())
        else:
            pos[0] += 1
    return forest


# ---------------------------------------------------------------------------
# Tree decoration: parents, columns, event-time indices
# ---------------------------------------------------------------------------

def set_parents(forest):
    """
    Set two parent pointers on every node:

      `parent`        the tree-structural parent (the L whose children
                      list contains this node, or None at top level)

      `arrow_parent`  the column from which the horizontal birth arrow
                      should be drawn:
                        * for a nested I/G: its tree parent
                        * for a top-level I/G: the nearest spatial-left
                          top-level ancestor (M or D); None means the
                          immortal link (column 0)

    The arrow_parent distinction is what prevents the birth arrow of a
    top-level insertion like the `I` in `M(...)I D(...)` from crossing
    the M lineage. Because we issue times in reverse-sibling order,
    M's own descendants are guaranteed to be born *after* such a
    top-level I, so the M->I arrow doesn't cross anything either.
    """
    for tree in forest:
        tree['parent'] = None
        _set_tree_parent(tree)

    last_top_ancestor = None
    for n in forest:
        if n['type'] in 'IG':
            n['arrow_parent'] = last_top_ancestor
        else:
            n['arrow_parent'] = None
            last_top_ancestor = n
        for c in n['children']:
            _set_arrow_parent(c, n)


def _set_tree_parent(node):
    for c in node['children']:
        c['parent'] = node
        _set_tree_parent(c)


def _set_arrow_parent(node, ap):
    node['arrow_parent'] = ap
    for c in node['children']:
        _set_arrow_parent(c, node)


def assign_columns(forest):
    """Pre-order DFS column assignment (1-indexed, left to right)."""
    c = [0]
    def go(n):
        c[0] += 1
        n['col'] = c[0]
        for ch in n['children']:
            go(ch)
    for t in forest:
        go(t)


def _effective_children(forest, flat):
    """
    Map each 'logical parent' to its 'effective children' in column order.

    A logical parent is either a node in `flat` (keyed by id) or the
    immortal link (keyed by None). Its effective children are its
    tree-structural children plus any top-level I/G whose arrow_parent
    points at it. The leftmost-most-recent rule applies to this whole
    list, not just to tree children, which is what makes branching
    arrows for top-level insertions like `M(...)I(...)D(...)` honour
    the no-crossing property.
    """
    eff = defaultdict(list)
    for n in flat:
        for c in n['children']:
            eff[id(n)].append(c)
    for n in forest:
        if n['type'] in 'IG':
            ap = n.get('arrow_parent')
            eff[id(ap) if ap is not None else None].append(n)
    for k in eff:
        eff[k].sort(key=lambda x: x['col'])
    return eff


def assign_times(forest, rng=None, label_order='spatial'):
    """
    Assign event-time POSITIONS and LABELS.

    Two distinct concepts per event:

      `birth_pos` / `death_pos`
          The integer position in [1, n] giving where the event sits in
          temporal order (1 = earliest, just below time 0; n = latest,
          just above time T). Always sampled from a uniform-random
          topological sort of the constraint DAG.

      `birth_idx` / `death_idx`
          The integer that appears as the t_? subscript both on the
          figure's vertical axis and in the parse-tree expressions.

    With label_order='temporal' the labels equal the positions, so the
    y-axis reads 0, t_1, t_2, ..., t_n, T monotonically and the parse
    tree's subscripts change with the seed.

    With label_order='spatial' (the default) the labels are assigned by
    walking the forest in preorder DFS, emitting birth-then-death for
    each node. The parse tree is then INVARIANT under --seed, and only
    the y-axis tick labels in the figure get permuted to follow the
    sampled temporal order.

    Constraints captured by the DAG:

      (P1) tree parent's birth < child's birth, when tree parent is I/G
      (P2) child's birth < tree parent's death, when tree parent is D/G
      (P3) for top-level I/G whose arrow_parent is a D:
           child's birth < arrow parent's death
      (Self) own birth < own death, for every G
      (Sib) effective siblings born in REVERSE column order
            (leftmost = most recent).
    """
    if rng is None:
        rng = random.Random()
    flat = flatten(forest)
    by_id = {id(n): n for n in flat}

    # Enumerate events as (node_id, kind) so they're hashable.
    events = []
    for n in flat:
        if n['type'] in 'IG':
            events.append((id(n), 'birth'))
        if n['type'] in 'DG':
            events.append((id(n), 'death'))

    eff = _effective_children(forest, flat)

    # Build the constraint DAG.
    edges = []
    for n in flat:
        if n['type'] not in 'IG':
            continue
        tp = n.get('parent')
        if tp is not None:
            if tp['type'] in 'IG':
                edges.append(((id(tp), 'birth'), (id(n), 'birth')))
            if tp['type'] in 'DG':
                edges.append(((id(n), 'birth'), (id(tp), 'death')))
    for n in forest:
        if n['type'] not in 'IG':
            continue
        ap = n.get('arrow_parent')
        if ap is not None and ap['type'] in 'DG':
            edges.append(((id(n), 'birth'), (id(ap), 'death')))
    for n in flat:
        if n['type'] == 'G':
            edges.append(((id(n), 'birth'), (id(n), 'death')))
    for _key, kids in eff.items():
        for i in range(len(kids) - 1):
            left, right = kids[i], kids[i + 1]
            edges.append(((id(right), 'birth'), (id(left), 'birth')))

    # Random topological sort -> POSITIONS.
    in_deg = defaultdict(int)
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        in_deg[b] += 1
    for e in events:
        _ = in_deg[e]

    available = [e for e in events if in_deg[e] == 0]
    order = []
    while available:
        i = rng.randrange(len(available))
        e = available.pop(i)
        order.append(e)
        for b in adj[e]:
            in_deg[b] -= 1
            if in_deg[b] == 0:
                available.append(b)

    if len(order) != len(events):
        raise ValueError(
            f'Constraint cycle detected (sorted {len(order)} of '
            f'{len(events)} events). Check input tree for inconsistencies.')

    # Boundary defaults (time 0 / time T).
    for n in flat:
        if n['type'] in 'MD':
            n['birth_pos'] = 0
            n['birth_idx'] = 0
        if n['type'] in 'MI':
            n['death_pos'] = None
            n['death_idx'] = None

    # Assign POSITIONS in temporal order (the sampled topological order).
    for i, (nid, kind) in enumerate(order):
        node = by_id[nid]
        if kind == 'birth':
            node['birth_pos'] = i + 1
        else:
            node['death_pos'] = i + 1

    # Assign LABELS.
    if label_order == 'temporal':
        for n in flat:
            if n['type'] in 'IG':
                n['birth_idx'] = n['birth_pos']
            if n['type'] in 'DG':
                n['death_idx'] = n['death_pos']
    elif label_order == 'spatial':
        c = [0]
        def visit(node):
            if node['type'] in 'IG':
                c[0] += 1
                node['birth_idx'] = c[0]
            if node['type'] in 'DG':
                c[0] += 1
                node['death_idx'] = c[0]
            for ch in node['children']:
                visit(ch)
        for t in forest:
            visit(t)
    else:
        raise ValueError(f"label_order must be 'spatial' or 'temporal', got {label_order!r}")

    return len(order)


def flatten(forest):
    out = []
    def go(n):
        out.append(n)
        for c in n['children']:
            go(c)
    for t in forest:
        go(t)
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _draw_lottery_row(ax, forest, n_events, lottery_seed):
    """Lottery-ticket cartoon row beneath the trajectory: each alive
    column draws an ID from a CRP-style pool; a single matched pair
    (the speed-dating MATCH) shares an ID, signposting Potts-coupled
    coevolution between distant alive lineages.
    """
    nodes = flatten(forest)
    nodes_by_col = sorted(nodes, key=lambda n: n['col'])
    max_col = max(n['col'] for n in nodes)

    rng = random.Random(lottery_seed)
    pool = list(range(0, 1000))
    rng.shuffle(pool)
    ids = pool[:max_col]

    alive_cols = [n['col'] for n in nodes_by_col if n['type'] in 'MI']
    if len(alive_cols) >= 2:
        best = (None, None, -1)
        for _ in range(50):
            i, j = sorted(rng.sample(alive_cols, 2))
            if j - i > best[2]:
                best = (i, j, j - i)
        match_i, match_j, _ = best
    elif len(alive_cols) == 1:
        match_i, match_j = alive_cols[0], alive_cols[0]
    else:
        match_i, match_j = 1, max_col
    ids[match_j - 1] = ids[match_i - 1]

    PALE_RED = mcolors.hsv_to_rgb([0.00, 0.18, 1.00])
    BRIGHT_RED = mcolors.hsv_to_rgb([0.00, 0.55, 1.00])

    def ticket_color(c):
        return BRIGHT_RED if c in (match_i, match_j) else PALE_RED

    y_top = n_events + 1
    y_t0 = y_top + 0.5
    y_t1 = y_top + 1.6
    tw = 0.78
    th = y_t1 - y_t0
    for col in range(1, max_col + 1):
        x_l = col - tw / 2
        ax.add_patch(FancyBboxPatch(
            (x_l, y_t0), tw, th,
            boxstyle='round,pad=0,rounding_size=0.10',
            facecolor=ticket_color(col), edgecolor='black', linewidth=1.2,
            zorder=3))
        ax.add_patch(FancyBboxPatch(
            (x_l + 0.07, y_t0 + 0.10), tw - 0.14, th - 0.20,
            boxstyle='round,pad=0,rounding_size=0.07',
            facecolor='none', edgecolor='black', linewidth=0.6,
            linestyle=(0, (3, 2)), zorder=4))
        ax.text(col, (y_t0 + y_t1) / 2, f'{ids[col - 1]:03d}',
                ha='center', va='center',
                fontweight='bold', fontsize=9, family='monospace',
                zorder=5)

    if match_i != match_j:
        y_arc_top = y_t1 + 0.15
        y_arc_peak = y_arc_top + 1.0
        verts = [(match_i, y_arc_top),
                 (match_i, y_arc_peak),
                 (match_j, y_arc_peak),
                 (match_j, y_arc_top)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor='none',
                                edgecolor='#C40233', linewidth=1.6, zorder=4))
        x_mid = 0.5 * (match_i + match_j)
        y_mid = y_arc_peak - 0.05
        ax.text(x_mid, y_mid, '♥ MATCH ♥',
                ha='center', va='center', color='white',
                fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.30,rounding_size=0.30',
                          facecolor='#C40233', edgecolor='#C40233',
                          linewidth=0),
                zorder=6)
        return y_arc_peak + 0.5
    return y_t1 + 0.3


def draw_trajectory_on_ax(ax, forest, n_events, lottery_seed=None,
                          column_xticks_top=True, show_column_labels=True,
                          show_immortal_link=True):
    """Render the trajectory (Galton-Watson tree + optional lottery row)
    onto an existing matplotlib Axes. Returns (max_col, Y_TOP, Y_BOT,
    lottery_bottom) for the caller to coordinate axis limits.

    This is the workhorse used both by `draw_figure` (which makes its
    own figure) and by `tkf_trajectory_uber.py` (which stacks this on
    top of a parse-tree panel).

    show_immortal_link: when False, omit the dashed vertical line at
    column 0 AND skip the horizontal spawning arrow for top-level
    insertions whose arrow_parent is None (those would be drawn from
    column 0 to the insertion's column). Used by the uber-figure to
    reclaim horizontal space.
    """
    nodes = flatten(forest)
    max_col = max(n['col'] for n in nodes)

    Y_TOP, Y_BOT = 0, n_events + 1

    def y_of(pos):
        return Y_BOT if pos is None else pos

    # Faint top (time 0) and bottom (time T) reference lines.
    ax.axhline(y=Y_TOP, color='0.75', linewidth=0.6, zorder=1)
    ax.axhline(y=Y_BOT, color='0.75', linewidth=0.6, zorder=1)

    # Immortal link at column 0 (dashed). Skipped when caller passes
    # show_immortal_link=False (e.g. the uber-figure caller, which uses
    # the freed horizontal space for the parse-tree panel).
    if show_immortal_link:
        ax.plot([0, 0], [Y_TOP, Y_BOT], linestyle=(0, (4, 3)),
                color='0.4', linewidth=1.0, zorder=1)

    LW = 1.7
    MS = 9

    for n in nodes:
        col = n['col']
        y_b = y_of(n.get('birth_pos'))
        y_d = y_of(n.get('death_pos'))

        # Vertical lineage line.
        ax.plot([col, col], [y_b, y_d], 'k-',
                linewidth=LW, solid_capstyle='butt', zorder=2)

        # Top marker (birth event for insertions; nothing for ancestors).
        if n['type'] in 'IG':
            ax.plot([col], [y_b], 'o',
                    markerfacecolor='white', markeredgecolor='black',
                    markersize=MS, markeredgewidth=1.4, zorder=4)

        # Bottom marker (filled circle if alive at T, X if dead).
        if n['type'] in 'MI':
            ax.plot([col], [y_d], 'o',
                    markerfacecolor='black', markeredgecolor='black',
                    markersize=MS, zorder=4)
        else:
            ax.plot([col], [y_d], 'x', color='black',
                    markersize=MS + 1, markeredgewidth=2.0, zorder=4)

        # Horizontal branching arrow for insertions. When show_immortal_link
        # is False and the arrow_parent is None (top-level insertion spawned
        # from the implicit immortal link), skip the arrow entirely instead
        # of drawing it from a column 0 that no longer exists.
        if n['type'] in 'IG':
            ap = n.get('arrow_parent')
            if ap is None and not show_immortal_link:
                pass  # no arrow when the immortal link is hidden
            else:
                p_col = ap['col'] if ap else 0  # 0 = immortal link
                ax.plot([p_col, col], [y_b, y_b], 'k-',
                        linewidth=LW, zorder=2)

    if lottery_seed is not None:
        lottery_bottom = _draw_lottery_row(ax, forest, n_events, lottery_seed)
    else:
        lottery_bottom = Y_BOT + 0.4

    # Build position -> label map for y-axis tick labels.
    pos_to_label = {}
    for n in nodes:
        if n['type'] in 'IG':
            bp = n.get('birth_pos')
            if bp is not None and bp > 0:
                pos_to_label[bp] = n['birth_idx']
        if n['type'] in 'DG':
            dp = n.get('death_pos')
            if dp is not None and dp > 0:
                pos_to_label[dp] = n['death_idx']

    yticks = [0] + list(range(1, n_events + 1)) + [Y_BOT]
    ylabels = [r'$0$']
    for i in range(1, n_events + 1):
        lab = pos_to_label.get(i, i)
        ylabels.append(rf'$t_{{{lab}}}$')
    ylabels.append(r'$T$')
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    if show_column_labels:
        # x-tick labels: column types (M/I/D/G).
        type_by_col = {n['col']: n['type'] for n in nodes}
        xticks = list(range(1, max_col + 1))
        xlabels = [type_by_col[c] for c in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontweight='bold', fontsize=11)
        if column_xticks_top:
            ax.xaxis.tick_top()
        ax.tick_params(axis='x', length=0, pad=6)
    else:
        ax.set_xticks([])

    ax.tick_params(axis='y', length=4)

    for side in ('right', 'top', 'bottom'):
        ax.spines[side].set_visible(False)

    ax.set_ylabel('time', labelpad=10)

    return max_col, Y_TOP, Y_BOT, lottery_bottom


def draw_figure(forest, n_events, output_path, lottery_seed=None):
    nodes = flatten(forest)
    max_col = max(n['col'] for n in nodes)

    fig_w = max(6.0, 1.6 + 0.95 * max_col)
    extra_h = 2.5 if lottery_seed is not None else 0.0
    fig_h = max(4.2, 1.2 + 0.42 * (n_events + 1)) + extra_h
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    max_col, Y_TOP, _Y_BOT, lottery_bottom = draw_trajectory_on_ax(
        ax, forest, n_events, lottery_seed=lottery_seed,
        column_xticks_top=True, show_column_labels=True)

    ax.set_xlim(-0.7, max_col + 0.5)
    ax.set_ylim(Y_TOP - 0.4, lottery_bottom)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# LaTeX parse tree (forest package)
# ---------------------------------------------------------------------------

def t_obs_expr(b_idx):
    """Observation interval of a node born at event-index b_idx."""
    return 'T' if b_idx == 0 else f'T - t_{{{b_idx}}}'


def tau_expr(b_idx, d_idx):
    """Lifetime tau of a node: t_d - t_b (or just t_d if b_idx == 0)."""
    return f't_{{{d_idx}}}' if b_idx == 0 else f't_{{{d_idx}}} - t_{{{b_idx}}}'


def _N(content):
    """Wrap a forest node content in braces so commas inside it are safe."""
    return '{' + content + '}'


def link_forest(node, indent=2):
    """Forest LaTeX for the L(...) subtree rooted at `node`."""
    b = node['birth_idx']
    t_exp = t_obs_expr(b)
    if node['type'] in 'MI':
        terminal = 'A'
        T_loc = t_exp
    else:
        d = node['death_idx']
        terminal = f'G_{{{tau_expr(b, d)}}}'
        T_loc = tau_expr(b, d)
    ind = ' ' * indent
    out = [f'{ind}[{_N(f"$L({t_exp})$")}']
    out.append(f'{ind}  [{_N(f"${terminal}$")}]')
    out.append(d_forest(node['children'], t_exp, T_loc, indent + 2))
    out.append(f'{ind}]')
    return '\n'.join(out)


def d_forest(children, t_obs, T_loc, indent):
    """
    Forest LaTeX for a D(t_obs, T_loc) subtree producing a list of children.

    The grammar expands D iteratively, so each child adds one level of
    left-deep nesting of D non-terminals, terminating with epsilon.
    """
    ind = ' ' * indent
    d_label = _N(f'$D({t_obs},\\, {T_loc})$')
    if not children:
        return (f'{ind}[{d_label}\n'
                f'{ind}  [{_N("$\\varepsilon$")}]\n'
                f'{ind}]')
    first, rest = children[0], children[1:]
    out = [f'{ind}[{d_label}']
    out.append(link_forest(first, indent + 2))
    out.append(d_forest(rest, t_obs, T_loc, indent + 2))
    out.append(f'{ind}]')
    return '\n'.join(out)


def s_forest(forest, indent=2):
    """Sequence wrapper for multi-root case: S -> L S | epsilon."""
    ind = ' ' * indent
    s_label = _N('$S$')
    if not forest:
        return (f'{ind}[{s_label}\n'
                f'{ind}  [{_N("$\\varepsilon$")}]\n'
                f'{ind}]')
    first, rest = forest[0], forest[1:]
    out = [f'{ind}[{s_label}']
    out.append(link_forest(first, indent + 2))
    out.append(s_forest(rest, indent + 2))
    out.append(f'{ind}]')
    return '\n'.join(out)


def latex_parse_tree(forest):
    out = [
        '% Requires: \\usepackage{forest}',
        '%           \\usetikzlibrary{arrows.meta}',
        '',
        r'\begin{forest}',
        r'  for tree={font=\small,',
        r'            l sep=12pt, s sep=6pt,',
        r'            parent anchor=south, child anchor=north,',
        r'            edge={-{Stealth[length=4pt]}}, inner sep=2pt,',
        r'            anchor=center}',
    ]
    if len(forest) == 1:
        out.append(link_forest(forest[0]))
    else:
        out.append(s_forest(forest))
    out.append(r'\end{forest}')
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Visualize a TKF91 indel trajectory from a tree-decorated '
                    'sequence such as M(G(G))ID(GG(I)G).')
    p.add_argument('sequence',
                   help='Tree-decorated sequence string')
    p.add_argument('-o', '--output', default='tkf_trajectory.png',
                   help='Output figure path (default: tkf_trajectory.png)')
    p.add_argument('--latex', default=None,
                   help='Write parse-tree LaTeX snippet to this file '
                        '(default: print to stdout)')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed for time ordering of otherwise '
                        'unconstrained events (default: nondeterministic)')
    p.add_argument('--lottery-seed', type=int, default=None,
                   help='If given, draw a lottery-ticket row beneath the '
                        'trajectory using this RNG seed for the IDs and the '
                        'speed-dating MATCH pair. Default: no lottery row.')
    p.add_argument('--label-order', choices=['spatial', 'temporal'],
                   default='spatial',
                   help="t_k subscript convention: 'spatial' (default) "
                        'assigns k in fixed left-to-right DFS order so the '
                        'parse tree is invariant under --seed, with the '
                        "figure's y-axis labels permuted to match the "
                        "random temporal order; 'temporal' makes labels "
                        'match temporal order (the y-axis reads t_1..t_n '
                        'monotonically, but the parse tree changes with --seed).')
    args = p.parse_args()

    forest = parse_input(args.sequence)
    if not forest:
        print('Empty or invalid input.', file=sys.stderr)
        sys.exit(1)
    set_parents(forest)
    assign_columns(forest)
    rng = random.Random(args.seed)
    n_events = assign_times(forest, rng, label_order=args.label_order)

    draw_figure(forest, n_events, args.output,
                lottery_seed=args.lottery_seed)

    latex = latex_parse_tree(forest)
    if args.latex:
        with open(args.latex, 'w') as f:
            f.write(latex + '\n')
        print(f'LaTeX snippet written to {args.latex}', file=sys.stderr)
    else:
        print(latex)

    print(f'Figure written to {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
