#!/usr/bin/env python3
"""Regenerate neurips_workshop_paper2.tex from the PSB source.

Reads   ../tkf-dp/psb-paper/paper2-coupled.tex   (ws-procs11x85 / PSB)
Writes  ./neurips_workshop_paper2.tex            (NeurIPS 2026 workshop)

Every edit asserts its expected match count and aborts on a mismatch, so if
the PSB source changes upstream this fails loudly instead of writing a
half-converted file.  Run it only to re-port from an updated PSB source --
edits made directly to ./neurips_workshop_paper2.tex will be overwritten.

Nothing in ../tkf-dp is read except paper2-coupled.tex, and nothing there is
written.
"""
import pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / 'tkf-dp' / 'psb-paper' / 'paper2-coupled.tex'
DST = HERE / 'neurips_workshop_paper2.tex'

if not SRC.exists():
    sys.exit("PSB source not found: %s\n(This script is the only thing that "
             "needs it; building this directory does not.)" % SRC)

s = SRC.read_text()

def sub1(old, new, label):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit("FAIL [%s]: expected 1 occurrence, found %d" % (label, n))
    s = s.replace(old, new)
    print("ok   %s" % label)

def subN(old, new, expect, label):
    global s
    n = s.count(old)
    if n != expect:
        sys.exit("FAIL [%s]: expected %d, found %d" % (label, expect, n))
    s = s.replace(old, new)
    print("ok   %s (x%d)" % (label, expect))

# ---------------------------------------------------------------- preamble --
head = s[:s.index('\\begin{document}')]
macros = "\n".join(l for l in head.splitlines() if l.startswith('\\newcommand{\\'))
if macros.count('\\newcommand') != 18:
    sys.exit("FAIL: expected 18 \\newcommand macros, found %d"
             % macros.count('\\newcommand'))

PREAMBLE = r"""%% Variational inference in coupled models of amino acid substitution
%% NeurIPS 2026 workshop submission.
%%
%% Generated from ../tkf-dp/psb-paper/paper2-coupled.tex by port-from-psb.py.
%% This directory is self-contained: nothing outside it is needed to build.
%% neurips_2026.sty and checklist.tex are the official template files, copied
%% byte-for-byte and NOT modified.

\documentclass{article}

%% The bibliography is numeric (plainnat), so natbib is loaded in numbers
%% mode.  This is the mechanism the template documents for passing natbib
%% options; without it the build fails outright with "Bibliography not
%% compatible with author-year citations".
\PassOptionsToPackage{numbers,compress}{natbib}

%% Track: workshop with double-blind reviewing.  Anonymises the author block,
%% adds review line numbers, and hides the \begin{ack} section.  For the
%% camera-ready copy change this to [dblblindworkshop, final].
\usepackage[dblblindworkshop]{neurips_2026}

%% ---------------------------------------------------------------------------
%% ATTENTION: required by the style file for workshop tracks; prints in the
%% first-page footer of the camera-ready copy.  Replace with the workshop name.
\workshoptitle{WORKSHOP TITLE}
%% ---------------------------------------------------------------------------

%% ---- packages loaded by the NeurIPS 2026 template, in template order ----
\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{hyperref}       % hyperlinks
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{xcolor}         % colors

%% ---- additional packages this paper needs ----
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\usepackage{cleveref}       % must be loaded after hyperref

\newtheorem{defn}{Definition}
\newtheorem{rem}{Remark}
\crefname{defn}{Definition}{Definitions}
\Crefname{defn}{Definition}{Definitions}
\crefname{rem}{Remark}{Remarks}
\Crefname{rem}{Remark}{Remarks}

\graphicspath{{./figures/}}

""" + macros + r"""

\title{Variational inference in coupled models of amino acid substitution}

%% Not typeset under [dblblindworkshop] without "final" -- the style file
%% substitutes "Anonymous Author(s)".  Kept here for the camera-ready copy.
\author{%
  Annabel Large \\
  Department of Bioengineering\\
  University of California, Berkeley\\
  Berkeley, CA 94720, USA \\
  \texttt{annabel\_large@berkeley.edu} \\
  \And
  Ian Holmes \\
  Department of Bioengineering\\
  University of California, Berkeley\\
  Berkeley, CA 94720, USA \\
  \texttt{ihh@berkeley.edu} \\
}

\begin{document}

\maketitle
"""

old_open = s[:s.index('\\raggedbottom\n') + len('\\raggedbottom\n')]
s = PREAMBLE + s[len(old_open):]
print("ok   preamble replaced (\\raggedbottom dropped; the style file sets "
      "\\flushbottom)")

sub1(r"""
\title{Variational inference in coupled models of amino acid substitution}

\author{Annabel Large and Ian Holmes$^\dag$}
\address{Department of Bioengineering, University of California, Berkeley,\\
Berkeley, CA 94720, USA\\
$^\dag$E-mail: \texttt{ihh@berkeley.edu}}
""", "", "WS title/author/address block removed (now in the preamble)")

sub1(r"""\keywords{coevolution; continuous-time Markov chain; reversibility;
exchangeability; lumpability; Potts model; Dirichlet process; TKF model.}""",
r"""%% The NeurIPS 2026 template has no keywords element, so these are
%% commented out rather than dropped:
%% Keywords: coevolution; continuous-time Markov chain; reversibility;
%% exchangeability; lumpability; Potts model; Dirichlet process; TKF model.""",
     "keywords commented out (no such element in the template)")

sub1(r"""\copyrightinfo{\copyright\ 2026 The Authors. Open Access chapter published by World Scientific Publishing Company and distributed under the terms of the Creative Commons Attribution Non-Commercial (CC BY-NC) 4.0 License.}

""", "", "PSB copyright footnote removed (the style file prints its own)")

# ------------------------------------------------------------------ tables --
# All four tables share one ws-procs11x85 shape:
#
#   \begin{table}[t] \centering \tbl{CAPTION}{\footnotesize
#     \begin{minipage}{\textwidth}\centering BODY \end{minipage}
#     \label{...}}
#   \end{table}
#
# The template wants \caption above the tabular, \centering, no minipage and
# no \tbl.  One regex converts all four; the count is asserted.
TABLE_RE = re.compile(
    r"\\begin\{table\}\[t\]\n"
    r"\\centering\n"
    r"\\tbl\{(?P<cap>.*?)\}\n"
    r"\{\\footnotesize\n"
    r"\\begin\{minipage\}\{\\textwidth\}\\centering\n"
    r"(?P<body>.*?)"
    r"\\end\{minipage\}\n"
    r"\\label\{(?P<label>[^}]*)\}\n"
    r"\}\n"
    r"\\end\{table\}", re.S)

def table_repl(m):
    return ("\\begin{table}[t]\n"
            "\\caption{%s}\n"
            "\\label{%s}\n"
            "\\centering\n"
            "\\footnotesize\n"
            "%s"
            "\\end{table}" % (m.group('cap'), m.group('label'), m.group('body')))

s, n = TABLE_RE.subn(table_repl, s)
if n != 4:
    sys.exit("FAIL [tables]: expected 4 ws-procs tables, converted %d" % n)
print("ok   4 tables: \\tbl -> \\caption above the tabular, label moved up, "
      "minipage wrapper dropped, \\centering (x4)")

# ------------------------------------------------------- Figure 1 (TikZ) ----
# Was a 0.58/0.40 minipage pair with the caption beside the diagram; the
# template wants the caption below the figure.
sub1(r"""\begin{figure}[t]
\begin{minipage}[c]{0.58\textwidth}
\centering
\begin{tikzpicture}""",
r"""\begin{figure}[t]
\centering
%% The diagram was drawn to fill a 0.58\textwidth minipage of PSB's 6.6in
%% block; freed of the side-by-side layout it occupies under half the 5.5in
%% width, so it is scaled up uniformly.  \resizebox scales the text with the
%% cells, leaving every internal proportion as drawn.
\resizebox{0.75\textwidth}{!}{%
\begin{tikzpicture}""",
     "fig:params: left minipage dropped, figure centred, diagram scaled to "
     "0.75\\textwidth")

sub1(r"""\end{tikzpicture}
\end{minipage}\hfill
\begin{minipage}[c]{0.40\textwidth}
\caption{\textbf{The pair parameterizations""",
r"""\end{tikzpicture}}
\caption{\textbf{The pair parameterizations""",
     "fig:params: caption moved below the diagram, per the template")

sub1(r"""}
\label{fig:params}
\end{minipage}
\end{figure}""",
r"""}
\label{fig:params}
\end{figure}""", "fig:params: closed")

# ------------------------------------------------ 5.5in text block fixes ---
# The TikZ diagram was drawn to fit a 3.83in minipage inside PSB's 6.6in
# block.  At 5.5in full width the three panels per row plus the row of state
# labels sit comfortably, but the legend row ran past the last panel.
sub1(r"""  \fill[black!30](11,0)rectangle++(0.7,0.7);\node[right]at(11.8,0.35){\tiny $\pi$ (diag.)};""",
     r"""  \fill[black!30](10.6,0)rectangle++(0.7,0.7);\node[right]at(11.4,0.35){\tiny $\pi$ (diag.)};""",
     "fig:params legend: last swatch pulled in to clear the 5.5in edge")

sub1(r"""\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrr}""",
     r"""\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrr}""",
     "tab:pairfit: \\tabcolsep 5pt -> 4pt (was 2.9pt overfull)")

sub1(r"""\begin{equation}\label{eq:logWR}
 \log\frac{W_{x,x^{s:c}}}{R_{x_sc}}=\tfrac12\Big[\log\tfrac{\pi_{x_s}}{\pi_c}-\Delta^c_sE(x)\Big],\;\;
 \Delta^c_sE(x)=-[h_s(c)-h_s(x_s)]-\!\!\sum_{s'\ne s}\![H_{ss'}(c,x_{s'})-H_{ss'}(x_s,x_{s'})],
\end{equation}""",
r"""\begin{equation}\label{eq:logWR}
\begin{gathered}
 \log\frac{W_{x,x^{s:c}}}{R_{x_sc}}=\tfrac12\Big[\log\tfrac{\pi_{x_s}}{\pi_c}-\Delta^c_sE(x)\Big],\\
 \Delta^c_sE(x)=-[h_s(c)-h_s(x_s)]-\!\!\sum_{s'\ne s}\![H_{ss'}(c,x_{s'})-H_{ss'}(x_s,x_{s'})],
\end{gathered}
\end{equation}""",
     "eq:logWR reflowed onto two lines (was 18.6pt overfull)")

# --------------------------------------------- acknowledgments + backmatter --
sub1(r"""\section*{Acknowledgments}
We thank Debora Marks""",
r"""%% The style file's ack environment hides this section in the anonymised
%% submission and titles it "Acknowledgments and Disclosure of Funding" in
%% the camera-ready copy.
\begin{ack}
We thank Debora Marks""", "acknowledgments -> ack environment")

sub1(r"""at \url{https://tkfdp.net/}.




\clearpage
\bibliographystyle{ws-procs11x85}
\bibliography{../math-paper/refs,paper2-refs}

\end{document}""",
r"""at \url{https://tkfdp.net/}.
\end{ack}

%% References follow the acknowledgments and do not count towards the page
%% limit.  natbib+plainnat sets the unnumbered "References" heading itself.
%% The .bib file lives in this directory.
\small
\bibliographystyle{plainnat}
\bibliography{refs}
\normalsize

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ATTENTION: every checklist answer below is still \answerTODO{} (red).
\newpage
\input{checklist.tex}

\end{document}""",
     "bibliography -> plainnat + the local refs.bib; checklist appended")

# ------------------------------------------------------------- assertions --
for m in (r"\botrule", r"\colrule", r"\tbl{", r"\tablecaptionfont",
          r"\refstepcounter", r"\address{", r"\keywords{", r"\copyrightinfo{",
          r"\raggedbottom", r"\begin{minipage}"):
    if m in s:
        sys.exit("FAIL: ws-procs macro %s still present" % m)
if r"\hline" in s:
    sys.exit("FAIL: \\hline present; the template asks for no rules of that kind")
print("ok   no ws-procs11x85 macros remain")

DST.write_text(s)
print("\nwrote %s (%d bytes)" % (DST.name, len(s)))
