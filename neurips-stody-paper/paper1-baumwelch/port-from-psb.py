#!/usr/bin/env python3
"""Regenerate neurips_workshop_paper1.tex from the PSB source.

Reads   ../../psb-paper/paper1-baumwelch.tex      (ws-procs11x85 / PSB)
Writes  ./neurips_workshop_paper1.tex              (NeurIPS 2026 workshop)

Every edit asserts its expected match count and aborts on a mismatch, so if
the PSB source changes upstream this fails loudly instead of writing a
half-converted file.  Run it only to re-port from an updated PSB source --
edits made directly to ./neurips_workshop_paper1.tex will be overwritten.
"""
import pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent.parent / 'psb-paper' / 'paper1-baumwelch.tex'
DST = HERE / 'neurips_workshop_paper1.tex'

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
if macros.count('\\newcommand') != 16:
    sys.exit("FAIL: expected 16 \\newcommand macros, found %d"
             % macros.count('\\newcommand'))

PREAMBLE = r"""%% Closed-form Baum-Welch for TKF-based evolutionary models
%% NeurIPS 2026 workshop submission.
%%
%% Generated from ../../psb-paper/paper1-baumwelch.tex by port-from-psb.py.
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
\usepackage{cleveref}       % must be loaded after hyperref

\newtheorem{defn}{Definition}
\newtheorem{rem}{Remark}
\crefname{defn}{Definition}{Definitions}
\Crefname{defn}{Definition}{Definitions}

\graphicspath{{./figures/}}

""" + macros + r"""

\title{Closed-form Baum--Welch for TKF-based evolutionary models}

%% Not typeset under [dblblindworkshop] without "final" -- the style file
%% substitutes "Anonymous Author(s)".  Kept here for the camera-ready copy.
\author{%
  Ian Holmes \\
  Department of Bioengineering\\
  University of California, Berkeley\\
  Berkeley, CA 94720, USA \\
  \texttt{ihh@berkeley.edu} \\
  \And
  Annabel Large \\
  Department of Bioengineering\\
  University of California, Berkeley\\
  Berkeley, CA 94720, USA \\
  \texttt{annabel\_large@berkeley.edu} \\
}

\begin{document}

\maketitle
"""

old_open = s[:s.index('\\raggedbottom\n') + len('\\raggedbottom\n')]
s = PREAMBLE + s[len(old_open):]
print("ok   preamble replaced (\\raggedbottom dropped; the style file sets "
      "\\flushbottom)")

sub1(r"""
\title{Closed-form Baum--Welch for TKF-based evolutionary models}

\author{Ian Holmes$^\dag$ and Annabel Large}
\address{Department of Bioengineering, University of California, Berkeley,\\
Berkeley, CA 94720, USA\\
$^\dag$E-mail: \texttt{ihh@berkeley.edu}}
""", "", "WS title/author/address block removed (now in the preamble)")

sub1(r"""\keywords{TKF model; indel evolution; Baum--Welch; expectation
maximisation; stochastic variational inference; fragment mixtures.}""",
r"""%% The NeurIPS 2026 template has no keywords element, so these are
%% commented out rather than dropped:
%% Keywords: TKF model; indel evolution; Baum--Welch; expectation
%% maximisation; stochastic variational inference; fragment mixtures.""",
     "keywords commented out (no such element in the template)")

sub1(r"""\copyrightinfo{\copyright\ 2026 The Authors. Open Access chapter published by World Scientific Publishing Company and distributed under the terms of the Creative Commons Attribution Non-Commercial (CC BY-NC) 4.0 License.}

""", "", "PSB copyright footnote removed (the style file prints its own)")

# unnumbered display math -> amsmath, so review line numbers are correct
subN("\\[\n", "\\begin{equation*}\n", 2, "\\[ -> equation*")
subN("\\]\n", "\\end{equation*}\n", 2, "\\] -> equation*")

# ------------------------------------------- Table 1: score derivatives ----
sub1(r"""\begin{table}[t]
\hrule\vskip7pt
\tbl{Score derivatives""",
r"""\begin{table}[t]
\caption{Score derivatives""", "T1: \\tbl -> \\caption, WS \\hrule dropped")

sub1(r"""[\cdot]^{\theta}_{\xi}$ abbreviates the $(\theta,\xi)$ entry.
}
{%
\footnotesize""",
r"""[\cdot]^{\theta}_{\xi}$ abbreviates the $(\theta,\xi)$ entry.}
\label{tab:score-derivatives}
\centering
\footnotesize""", "T1: caption closed, label moved up, centred")

# no vertical rules in tables, per the template; \hline -> booktabs
subN(r"""\begin{tabular}{c|cccccc}
& $\log\alpha$ & $\log(1{-}\alpha)$ & $\log\beta$
& $\log(1{-}\beta)$ & $\log\gamma$ & $\log(1{-}\gamma)$ \\
\hline""",
r"""\begin{tabular}{ccccccc}
\toprule
& $\log\alpha$ & $\log(1{-}\alpha)$ & $\log\beta$
& $\log(1{-}\beta)$ & $\log\gamma$ & $\log(1{-}\gamma)$ \\
\midrule""", 2, "T1 inner tabulars: vertical rule removed, booktabs rules")

# booktabs \bottomrule must follow a completed row, hence the added \\
sub1(r"""  & $1+[\cdot]^{\del}_{\beta}-\dfrac{s\alpha}{\Phi}$
\end{tabular}}""",
r"""  & $1+[\cdot]^{\del}_{\beta}-\dfrac{s\alpha}{\Phi}$
\\
\bottomrule
\end{tabular}}""", "T1 inner tabular 1: \\bottomrule")

sub1(r"""  & $\dfrac{s+2}{2(1+s)}-\dfrac{s\alpha}{\Phi}$
\end{tabular}}""",
r"""  & $\dfrac{s+2}{2(1+s)}-\dfrac{s\alpha}{\Phi}$
\\
\bottomrule
\end{tabular}}""", "T1 inner tabular 2: \\bottomrule")

sub1(r"""\end{tabular}%
}
\label{tab:score-derivatives}
\vskip7pt\hrule
\end{table}""",
r"""\end{tabular}
\end{table}""", "T1: trailing WS \\hrule and duplicate label removed")

# --------------------------------- Table 2: Pfam held-out log-likelihood ---
sub1(r"""\begin{table}[t]
\refstepcounter{table}\label{tab:pfam}%
\noindent
\begin{minipage}[c]{0.53\textwidth}
{\tablecaptionfont Table~\thetable. Held-out per-pair log-likelihood""",
r"""\begin{table}[t]
\caption{Held-out per-pair log-likelihood""",
     "T2: minipage/refstepcounter layout -> \\caption")

sub1(r"""latent fragment structure in \MixFrag/\MixDom; the order-summed column
marginalises it out.\par}
\end{minipage}\hfill
\begin{minipage}[c]{0.43\textwidth}
\centering
\footnotesize""",
r"""latent fragment structure in \MixFrag/\MixDom; the order-summed column
marginalises it out.}
\label{tab:pfam}
\centering
\footnotesize""", "T2: caption closed, label, centred")

sub1(r"""\botrule
\end{tabular}
\end{minipage}
\end{table}

\Cref{tab:pfam} reports""",
r"""\bottomrule
\end{tabular}
\end{table}

\Cref{tab:pfam} reports""", "T2: closed")

# ----------------------------------------- Table 3: BAliBASE accuracy -----
# [b] pushed this table below the References heading; [t] keeps it in content.
sub1(r"""\begin{table}[b]
\refstepcounter{table}\label{tab:balibase}%
\noindent
\begin{minipage}[c]{0.53\textwidth}
{\tablecaptionfont Table~\thetable. Full-length BAliBASE""",
r"""\begin{table}[t]
\caption{Full-length BAliBASE""",
     "T3: minipage layout -> \\caption, float [b] -> [t]")

sub1(r"""trained models; bottom block: MAFFT and MUSCLE~\cite{Katoh2013,Edgar2004}.\par}
\end{minipage}\hfill
\begin{minipage}[c]{0.43\textwidth}
\centering
\footnotesize""",
r"""trained models; bottom block: MAFFT and MUSCLE~\cite{Katoh2013,Edgar2004}.}
\label{tab:balibase}
\centering
\footnotesize""", "T3: caption closed, label, centred")

sub1(r"""\botrule
\end{tabular}
\end{minipage}
\end{table}""",
r"""\bottomrule
\end{tabular}
\end{table}""", "T3: closed")

subN(r"\colrule", r"\midrule", 3, "\\colrule -> \\midrule")

for m in (r"\botrule", r"\tbl{", r"\tablecaptionfont", r"\refstepcounter",
          r"\address{", r"\keywords{", r"\copyrightinfo{", r"\raggedbottom"):
    if m in s:
        sys.exit("FAIL: ws-procs macro %s still present" % m)
# The one surviving \hline is inside the pair-HMM transition matrix, an array
# in math mode where the rule is notation rather than table styling.
if s.count(r"\hline") != 1:
    sys.exit("FAIL: expected exactly 1 \\hline (the pair-HMM matrix), found %d"
             % s.count(r"\hline"))
print("ok   no ws-procs11x85 macros remain")

# ------------------------------------------------------------------ figure --
sub1(r"""\begin{figure}[tb]
\centering
\begin{minipage}[c]{0.6\textwidth}
\centering
\includegraphics[width=\linewidth]{tkf_trajectory_gravestone.pdf}
\end{minipage}\hfill
\begin{minipage}[c]{0.37\textwidth}
\caption{\textbf{TKF91""",
r"""\begin{figure}[tb]
\centering
\includegraphics[width=0.75\linewidth]{tkf_trajectory_gravestone.pdf}
\caption{\textbf{TKF91""",
     "figure: caption moved below the image, per the template")

sub1(r"""above its trajectory column.}
\label{fig:tkf-trajectory}
\end{minipage}
\end{figure}""",
r"""above its trajectory column.}
\label{fig:tkf-trajectory}
\end{figure}""", "figure: closed")

# ------------------------------------------------ 5.5in text block fixes ---
sub1(r"    \setlength{\arraycolsep}{6pt}",
     r"    \setlength{\arraycolsep}{3pt}",
     "pair-HMM matrix: \\arraycolsep 6pt -> 3pt (was 13.1pt overfull)")

sub1(r"""F_a \;=\; \hat n''_{aa}\,\frac{\ext}{\ext+(1-\ext)\,\tau'_{aa}},\quad
\hat n_{ab} = \hat n''_{ab}-\delta_{ab}F_a,\quad
F = \sum_a F_a,\quad
E = \sum_{a,b}\hat n_{ab} - 1,\quad
\hat\ext \;=\; \frac{F}{F+E},\quad
\end{equation}""",
r"""\begin{gathered}
F_a \;=\; \hat n''_{aa}\,\frac{\ext}{\ext+(1-\ext)\,\tau'_{aa}},\quad
\hat n_{ab} = \hat n''_{ab}-\delta_{ab}F_a,\quad
F = \sum_a F_a,\\
E = \sum_{a,b}\hat n_{ab} - 1,\quad
\hat\ext \;=\; \frac{F}{F+E},\quad
\end{gathered}
\end{equation}""",
     "eq:resolve reflowed onto two lines (was 25.5pt overfull)")

sub1(r"\subsection{Sufficient statistics by automatic differentiation through $\Zsum$}",
     "\\subsection{Sufficient statistics by automatic differentiation through\n"
     "  \\texorpdfstring{$\\Zsum$}{Z}}",
     "\\texorpdfstring for the one heading containing math")

# --------------------------------------------- acknowledgments + backmatter --
sub1(r"""\section*{Acknowledgments}
We thank Jeff Thorne""",
r"""%% The style file's ack environment hides this section in the anonymised
%% submission and titles it "Acknowledgments and Disclosure of Funding" in
%% the camera-ready copy.
\begin{ack}
We thank Jeff Thorne""", "acknowledgments -> ack environment")

sub1(r"""HG013117.

\clearpage
\bibliographystyle{ws-procs11x85}
\bibliography{../math-paper/refs,paper1-refs}

\end{document}""",
r"""HG013117.
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
     "bibliography -> plainnat + local .bib files; checklist appended")

DST.write_text(s)
print("\nwrote %s (%d bytes)" % (DST.name, len(s)))
