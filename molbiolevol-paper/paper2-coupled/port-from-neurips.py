#!/usr/bin/env python3
"""Record of how mbe_paper2.tex was first produced from the NeurIPS source.

Reads   ../../neurips-stody-paper/paper2-coupled/neurips_workshop_paper2.tex
Writes  ./mbe_paper2.tex

DO NOT RUN THIS.  mbe_paper2.tex is now hand-maintained and has diverged: the
lumpability material that the NeurIPS version cut for space has been restored,
which touches the figure, two tables, the notation appendix and several places
in the text.  Re-running this script would throw all of that away.  It is kept
because it documents, edit by edit, what the port to MBE changed.

The port is deliberately minimal: no prose is rewritten.  Only four kinds of
change are made.

  1. Template swap.  The NeurIPS class/style block is replaced by a plain
     `article` preamble following MBE's manuscript-formatting section: ~25 mm
     margins, double spacing, line numbers, author-year citations.  Every
     macro definition from the source is carried over verbatim.

  2. Section demotion.  MBE's Methods article type takes the headings
     Introduction / Results / Discussion / Materials and Methods, and states
     that the order may be changed for clarity (e.g. Materials and Methods
     before Results).  We use Introduction / Materials and Methods / Results
     / Discussion.  The theory sections become subsections of Materials and
     Methods and their subsections become subsubsections.

     "Many components and variational inference" is the one section that
     straddles the boundary: the derivation stays in Materials and Methods,
     while its figure and the closing paragraph that evaluates the bound move
     to Results.  That move needs one heading, which is the only sentence of
     new text in the port.

  3. Appendix folded in.  MBE sets no page limit, so the notation appendix
     becomes a further subsection of Materials and Methods rather than
     separate supplementary material.

  4. MBE-required apparatus.  A cover page (title, authors, affiliations with
     country, corresponding-author e-mail), key words, figure alt text, and the
     Data availability / Acknowledgements / Funding / Conflict of interest back
     matter.  The abstract is carried over verbatim; the script reports its
     word count against MBE's 250-word limit but does not rewrite it.

Every edit asserts its expected match count and aborts on a mismatch, so if
the NeurIPS source changes upstream this fails loudly instead of writing a
half-converted file.  Run it only to re-port from an updated source -- edits
made directly to mbe_paper2.tex will be overwritten.

Nothing outside this directory is written.
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = (HERE.parent.parent / 'neurips-stody-paper' / 'paper2-coupled'
       / 'neurips_workshop_paper2.tex')
DST = HERE / 'mbe_paper2.tex'

if not SRC.exists():
    sys.exit("NeurIPS source not found: %s\n(This script is the only thing "
             "that needs it; building this directory does not.)" % SRC)

src = SRC.read_text()


def fail(msg):
    sys.exit("FAIL: " + msg)


def ok(msg):
    print("ok   " + msg)


def cut(text, marker, label):
    """Index of `marker` in `text`, asserting it occurs exactly once."""
    n = text.count(marker)
    if n != 1:
        fail("[%s] expected 1 occurrence of %r, found %d" % (label, marker[:60], n))
    return text.index(marker)


def words(tex):
    """Word count of a LaTeX fragment, the way a copy editor would count it."""
    t = re.sub(r'(?m)^\s*%.*$', '', tex)
    t = re.sub(r'(?<!\\)%.*', '', t)
    t = re.sub(r'\$[^$]*\$', ' Xmath ', t)
    t = re.sub(r'\\[a-zA-Z@]+\*?', ' ', t)
    t = t.replace('{', ' ').replace('}', ' ').replace('\\', ' ')
    return sum(1 for w in t.split() if re.search(r'[A-Za-z0-9]', w))


def demote(tex, label, expect):
    """Drop every sectioning command in `tex` one level.

    Deepest first, so a heading is never demoted twice in the same pass.
    \\paragraph is left alone: this paper uses it as a run-in lead-in, not as
    a sectioning level.
    """
    counts = tuple(len(re.findall(r'(?m)^\\%s\{' % k, tex))
                   for k in ('section', 'subsection', 'subsubsection'))
    if counts != expect:
        fail("[%s] expected %s section/subsection/subsubsection headings, found %s"
             % (label, expect, counts))
    if counts[2]:
        fail("[%s] a subsubsection would demote past \\paragraph, which this "
             "paper uses as a run-in lead-in" % label)
    tex = re.sub(r'(?m)^\\subsection\{', r'\\subsubsection{', tex)
    tex = re.sub(r'(?m)^\\section\{', r'\\subsection{', tex)
    ok("demoted %s: %d sections, %d subsections" % (label, counts[0], counts[1]))
    return tex


# ------------------------------------------------------------------ macros --
head = src[:cut(src, '\\begin{document}', 'begin{document}')]
macro_lines = [l for l in head.splitlines()
               if l.startswith(('\\newcommand{\\', '\\newtheorem{', '\\crefname{',
                                '\\Crefname{', '\\graphicspath{'))]
MACROS = "\n".join(macro_lines)
N_MACROS = MACROS.count('\\newcommand')
if N_MACROS != 15:
    fail("expected 15 \\newcommand macros, found %d" % N_MACROS)
ok("carried %d preamble definitions" % len(macro_lines))

# The notation appendix needs these column types, which the source defines
# just after \appendix; they are lifted into the preamble here.
COLTYPES = r"""
%% Column layout for the notation tables (defined after \appendix in the source).
\newcolumntype{Y}{>{\raggedright\arraybackslash}p{0.19\textwidth}}
\newcolumntype{Z}{>{\raggedright\arraybackslash}p{0.19\textwidth}}
\newcolumntype{X}{>{\raggedright\arraybackslash}p{0.53\textwidth}}
\newcommand{\notationhead}{\toprule Symbol & Size & Description\\ \midrule}
\newcommand{\nodim}{\textemdash}
"""

# ------------------------------------------------------------- body slices --
i_intro = cut(src, '\\section{Introduction}', 'Introduction')
i_single = cut(src, '\\section{Single-site models}', 'Single-site models')
i_compare = cut(src, '\\section{Comparing the pair couplings}', 'Comparing the pair couplings')
i_multi = cut(src, '\\section{Many components and variational inference}', 'Many components')
i_discuss = cut(src, '\\section{Discussion}', 'Discussion')
i_bib = cut(src, '\\clearpage\n\\bibliographystyle{plainnat}', 'bibliography')
i_app = cut(src, '\\newpage\n\\appendix', 'appendix')
i_end = cut(src, '\\end{document}', 'end{document}')

if not (i_intro < i_single < i_compare < i_multi < i_discuss < i_bib < i_app < i_end):
    fail("body slices are out of order")

intro = src[i_intro:i_single]
theory = src[i_single:i_compare]        # Single-site models + Two-site models
compare = src[i_compare:i_multi]        # the empirical comparison -> Results
multi = src[i_multi:i_discuss]          # derivation + its evaluation
discussion = src[i_discuss:i_bib]
appendix = src[i_app + len('\\newpage\n\\appendix'):i_end]
ok("sliced body into intro / theory / compare / multi / discussion / appendix")

# The appendix carries the column-type definitions inline; they move to the
# preamble so the folded-in subsection is plain body text.
for line in COLTYPES.strip().splitlines():
    if line.startswith('%'):
        continue
    if appendix.count(line + '\n') != 1:
        fail("expected the appendix to define %r exactly once" % line[:50])
    appendix = appendix.replace(line + '\n', '', 1)
ok("lifted the notation-table column types into the preamble")

# ------------------------------------- split "Many components" at the figure -
FIG_START = '\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.78\\textwidth]{figures/elbo_vs_expm_combined_legend.pdf}'
FIG_END = '\\label{fig:elbovsexpm}\n\\end{figure}\n'
RESULT_START = 'The bound is a faithful surrogate for the exact coupled matrix exponential'

f0 = cut(multi, FIG_START, 'elbo figure start')
f1 = cut(multi, FIG_END, 'elbo figure end') + len(FIG_END)
r0 = cut(multi, RESULT_START, 'elbo evaluation paragraph')
if not (f0 < f1 < r0):
    fail("the elbo figure does not precede its evaluation paragraph")

elbo_fig = multi[f0:f1]
multi_methods = multi[:f0] + multi[f1:r0]
elbo_results = multi[r0:]
ok("split 'Many components' into derivation (Methods) and evaluation (Results)")

# ----------------------------------------------------------- 2+3. demotion ---
theory = demote(theory, 'theory', (2, 2, 0))
multi_methods = demote(multi_methods, 'variational derivation', (1, 0, 0))
compare = demote(compare, 'pair comparison', (1, 0, 0))
appendix = demote(appendix, 'notation appendix', (1, 0, 0))

APPENDIX_NOTE = r"""
%% The notation subsection below was an appendix in earlier versions of this
%% paper.  Molecular Biology and Evolution sets no page limit, so it is folded
%% into Materials and Methods rather than shipped as supplementary material.
%% Its text is unchanged.
"""

methods = ('\\section{Materials and Methods}\n\\label{sec:methods}\n\n'
           + theory + multi_methods + APPENDIX_NOTE + appendix)

# The one heading this port invents, for the evaluation moved out of the
# derivation section.
ELBO_RESULTS_HEAD = ('\n% =============================================================\n'
                     '\\subsection{Accuracy and stability of the path ELBO}\n'
                     '\\label{sec:elboresults}\n'
                     '% =============================================================\n\n')

results = ('\\section{Results}\n\\label{sec:results}\n\n'
           + compare + ELBO_RESULTS_HEAD + elbo_fig + '\n' + elbo_results)

body = intro + methods + results + discussion

# ------------------------------------------ author-year citation commands ---
n_cite = len(re.findall(r'\\cite\{', body))
if n_cite != 33:
    fail("expected 33 \\cite{} commands, found %d" % n_cite)
body = re.sub(r'\\cite\{', r'\\citep{', body)
ok("converted %d \\cite -> \\citep (MBE cites by author and year)" % n_cite)

# ------------------------------------------------- sectional cross-references -
# The introduction's roadmap pointed at a single section for the ELBO; that
# material is now split across Materials and Methods and Results.
OLD_PTR = ('closed-form for coupled pairs and more stable to compute than their '
           'tighter ODE-based bound (\\Cref{sec:multi}).')
NEW_PTR = ('closed-form for coupled pairs and more stable to compute than their '
           'tighter ODE-based bound (\\Cref{sec:multi}; evaluated in '
           '\\Cref{sec:elboresults}).')
if body.count(OLD_PTR) != 1:
    fail("could not find the introduction's pointer at \\Cref{sec:multi}")
body = body.replace(OLD_PTR, NEW_PTR)
ok("updated the introduction's roadmap cross-reference")

# ----------------------------------------------------------- figure alt text -
# MBE requires an alt-text description under each figure legend.
ALT_TEXT = {
    'fig:params': (
        "Alt text: Four square grids side by side, one per parameterization, "
        "each a four-by-four block over the states 00, 01, 10 and 11. Cells "
        "are colored to show which flux parameters are free, which are zero "
        "and which are the stationary diagonal."),
    'fig:components': (
        "Alt text: Eight heat-map panels, one per fitted interaction class, "
        "each covering the 400 ordered amino-acid pairs with red for favored "
        "and blue for disfavored pairs. Residues are grouped along both axes, "
        "and the panels are ordered by mixture weight."),
    'fig:elbovsexpm': (
        "Alt text: Two line plots against branch length. The left plot shows "
        "the rank correlation between the ELBO and the exact matrix "
        "exponential, with the coupling-free baseline below it; the right "
        "plot shows the mean gap between exact and approximate values."),
}

for label, alt in ALT_TEXT.items():
    marker = '\\label{%s}\n' % label
    if body.count(marker) != 1:
        fail("could not place alt text: %r not found exactly once" % marker)
    body = body.replace(marker, marker + '\n{\\footnotesize\\itshape %s}\n' % alt)
ok("added alt text for %d figures" % len(ALT_TEXT))

live = re.sub(r'(?m)^\s*%.*$', '', body)   # commented-out figures don't count
n_fig = len(re.findall(r'\\begin\{figure\}', live))
if n_fig != len(ALT_TEXT):
    fail("%d figures but %d alt-text entries" % (n_fig, len(ALT_TEXT)))

# --------------------------------------------------------------- abstract ----
# Verbatim from the source.  MBE allows 250 words; overage is reported, not
# fixed, because trimming it is an authorial decision.
i_abs = cut(src, '\\begin{abstract}', 'begin{abstract}') + len('\\begin{abstract}')
ABSTRACT = src[i_abs:cut(src, '\\end{abstract}', 'end{abstract}')].strip()

n_abs = words(ABSTRACT)
if n_abs <= 250:
    ok("abstract is %d words (MBE limit 250)" % n_abs)
else:
    print("WARN abstract is %d words; MBE allows 250 -- trim %d words before "
          "submitting" % (n_abs, n_abs - 250))

m = re.search(r'%% Keywords: (.*?)\.\n', src, re.S)
if not m:
    fail("could not find the commented-out keyword list in the source")
KEYWORDS = re.sub(r'\s*\n%%\s*', ' ', m.group(1)).strip()
ok("key words: %s" % KEYWORDS)

# ------------------------------------------------------------------ preamble -
PREAMBLE = r"""%% Variational inference in coupled models of amino acid substitution
%% Molecular Biology and Evolution -- Methods manuscript, initial submission.
%%
%% GENERATED FILE.  Produced from
%%   ../../neurips-stody-paper/paper2-coupled/neurips_workshop_paper2.tex
%% by port-from-neurips.py.  Edit the NeurIPS source and re-run that script;
%% direct edits here are overwritten.
%%
%% Formatting follows the "Manuscript formatting" section of the MBE author
%% guidelines: one PDF carrying main text, references, tables, figures and
%% captions; line numbers; double spacing; ~25 mm margins.  There is no
%% separate supplementary file -- the former appendix is a subsection of
%% Materials and Methods.

\documentclass[11pt]{article}

\usepackage[letterpaper,margin=25mm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

%% MBE cites by author and year, so natbib runs in author-year mode with the
%% comma between author and year suppressed: "(Cohn et al. 2010)".
\usepackage[round,authoryear]{natbib}
\setcitestyle{aysep={}}

\usepackage{setspace}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{makecell}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\usepackage{array}
\usepackage{longtable}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage[mathlines]{lineno}
\usepackage{cleveref}   % must be loaded after hyperref

%% lineno does not number displayed equations unless the amsmath environments
%% are wrapped in linenomath.  This is the standard patch.
\newcommand*\patchAmsMathEnvironmentForLineno[1]{%
  \expandafter\let\csname old#1\expandafter\endcsname\csname #1\endcsname
  \expandafter\let\csname oldend#1\expandafter\endcsname\csname end#1\endcsname
  \renewenvironment{#1}%
     {\linenomath\csname old#1\endcsname}%
     {\csname oldend#1\endcsname\endlinenomath}}
\newcommand*\patchBothAmsMathEnvironmentsForLineno[1]{%
  \patchAmsMathEnvironmentForLineno{#1}%
  \patchAmsMathEnvironmentForLineno{#1*}}
\AtBeginDocument{%
  \patchBothAmsMathEnvironmentsForLineno{equation}%
  \patchBothAmsMathEnvironmentsForLineno{align}%
  \patchBothAmsMathEnvironmentsForLineno{flalign}%
  \patchBothAmsMathEnvironmentsForLineno{alignat}%
  \patchBothAmsMathEnvironmentsForLineno{gather}%
  \patchBothAmsMathEnvironmentsForLineno{multline}%
}

%% lineno cannot number rows inside a longtable, and the notation tables are
%% longtables; they switch numbering off and back on around themselves.
\let\oldlongtable\longtable
\let\oldendlongtable\endlongtable
\renewenvironment{longtable}{\nolinenumbers\oldlongtable}%
                            {\oldendlongtable\linenumbers}

%% cleveref calls every sectioning unit a "section", which is what the prose
%% assumes now that the theory sits under Materials and Methods.
\crefname{subsection}{section}{sections}
\Crefname{subsection}{Section}{Sections}
\crefname{subsubsection}{section}{sections}
\Crefname{subsubsection}{Section}{Sections}

%% ---- macros carried over unchanged from the NeurIPS source ----
__MACROS__
__COLTYPES__

\begin{document}
\doublespacing
\linenumbers

%% ------------------------------------------------------------ cover page ---
\begin{singlespace}
\thispagestyle{empty}
\begin{center}
{\normalsize Article type: Methods\par}
\vspace{1.5em}
{\Large\bfseries Variational inference in coupled models of amino acid
substitution\par}
\vspace{1.5em}
{\large Annabel Large$^{1}$ and Ian Holmes$^{1,\ast}$\par}
\vspace{1.2em}
\begin{minipage}{0.85\textwidth}\centering
$^{1}$Department of Bioengineering, University of California, Berkeley,
Berkeley, CA 94720, United States of America\\[0.6em]
$^{\ast}$Corresponding author: Ian Holmes, \texttt{ihh@berkeley.edu}
\end{minipage}
\end{center}
\end{singlespace}

\vspace{2em}

\begin{singlespace}
\noindent\textbf{Abstract}

\noindent __ABSTRACT__

\vspace{1em}
\noindent\textbf{Key words:} __KEYWORDS__.
\end{singlespace}

\newpage
"""

BACKMATTER = r"""
% =============================================================
\section*{Data availability}
% =============================================================

This study generated no new sequence or structural data. The coupled
substitution count tensors were assembled from the publicly available
trRosetta training set of structures and alignments \citep{Yang2020trRosetta}.
Source code for the models, the count-tensor pipeline and the ELBO, together
with the fitted parameters and the scripts that regenerate every table and
figure in this paper, is available at
\url{https://github.com/evoldoers/tkfdp}.

% =============================================================
\section*{Acknowledgements}
% =============================================================

We thank Debora Marks, Yun Song, and Sebastian Prillo for helpful discussions.
Some of the mathematical derivations, exposition and code in this work were
drafted with the assistance of large language models (products of Anthropic and
OpenAI) and were then checked and edited by the authors, who take full
responsibility for the content.

\subsection*{Funding}

This work was supported by the National Institutes of Health
(HG004483, GM080203, HG013117).

% =============================================================
\section*{Conflict of interest}
% =============================================================

The authors declare no conflict of interest.

\newpage
\bibliographystyle{abbrvnat}
\bibliography{refs}

\end{document}
"""

out = (PREAMBLE
       .replace('__MACROS__', MACROS)
       .replace('__COLTYPES__', COLTYPES)
       .replace('__ABSTRACT__', ABSTRACT)
       .replace('__KEYWORDS__', KEYWORDS)
       + body + BACKMATTER)

DST.write_text(out)
ok("wrote mbe_paper2.tex (%d bytes)" % len(out))
print("\nmain-text word count (body only, excluding abstract and references): ~%d"
      % words(body))
