#!/usr/bin/env python3
"""Record of how mbe_paper1.tex was first produced from the NeurIPS source.

Reads   ../../neurips-stody-paper/paper1-baumwelch/neurips_workshop_paper1.tex
Writes  ./mbe_paper1.tex

DO NOT RUN THIS.  mbe_paper1.tex is hand-maintained now.  Running this script
overwrites it wholesale from the NeurIPS source, which silently discards any
edit made to the MBE version -- a trimmed abstract, say.  It is kept because it
documents, edit by edit, what the port to MBE changed.

The port is deliberately minimal.  Only four kinds of change are made:

  1. Template swap.  The NeurIPS class/style block is replaced by a plain
     `article` preamble following MBE's manuscript-formatting section: ~25 mm
     margins, double spacing, line numbers, author-year citations.  Every
     macro definition from the source is carried over verbatim.

  2. Section demotion.  MBE's Methods article type takes the headings
     Introduction / Results / Discussion / Materials and Methods, and states
     that the order may be changed for clarity (e.g. Materials and Methods
     before Results).  We use Introduction / Materials and Methods / Results
     / Discussion.  The four theory sections between the Introduction and the
     Results become subsections of Materials and Methods, and everything below
     them drops one level too.  No prose is reordered.

  3. Appendix folded in or dropped.  MBE sets no page limit, so the two
     appendices that are still wanted become subsubsections of the Materials
     and Methods subsection each one supports: the svi-bw implementation
     analyses under MixFrag, and the gravestone pair SCFG under the
     indel-history sampler.  The remaining two are dropped -- "The
     substitution model M-step" restates material derived at length in the
     companion manuscript, and "Exploded MixDom pair HMM" is a state-by-state
     transition listing that nothing else in the text depends on.  The
     "see Appendix N" pointers become ordinary cross-references, or are
     removed with the appendix they pointed at.  Nothing is left as separate
     supplementary material.

  4. MBE-required apparatus.  A cover page (title, authors, affiliations with
     country, corresponding-author e-mail), key words, figure alt text, and the
     Data availability / Acknowledgements / Funding / Conflict of interest back
     matter.  The abstract and every other piece of prose are carried over
     verbatim; the script reports the abstract word count against MBE's 250-word
     limit but does not rewrite it.

Every edit asserts its expected match count and aborts on a mismatch, so if
the NeurIPS source changes upstream this fails loudly instead of writing a
half-converted file.  Run it only to re-port from an updated source -- edits
made directly to mbe_paper1.tex will be overwritten.

Nothing outside this directory is written.
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = (HERE.parent.parent / 'neurips-stody-paper' / 'paper1-baumwelch'
       / 'neurips_workshop_paper1.tex')
DST = HERE / 'mbe_paper1.tex'

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
    """Word count of a LaTeX fragment, the way a copy editor would count it.

    Comments are dropped, each inline-math group counts as one word, control
    sequences and braces vanish, and a hyphenated compound counts as one word.
    """
    t = re.sub(r'(?m)^\s*%.*$', '', tex)
    t = re.sub(r'(?<!\\)%.*', '', t)
    t = re.sub(r'\$[^$]*\$', ' Xmath ', t)
    t = re.sub(r'\\[a-zA-Z@]+\*?', ' ', t)
    t = t.replace('{', ' ').replace('}', ' ').replace('\\', ' ')
    return sum(1 for w in t.split() if re.search(r'[A-Za-z0-9]', w))


def demote(tex, label, expect, levels=1):
    """Drop every sectioning command in `tex` by `levels`.

    Deepest first within each pass, so a heading is never demoted twice by
    the same pass.  \\paragraph is the floor; the preamble makes it a
    numbered block heading, and this paper uses no \\paragraph of its own.
    """
    counts = tuple(len(re.findall(r'(?m)^\\%s\{' % k, tex))
                   for k in ('section', 'subsection', 'subsubsection'))
    if counts != expect:
        fail("[%s] expected %s section/subsection/subsubsection headings, found %s"
             % (label, expect, counts))
    for _ in range(levels):
        if re.search(r'(?m)^\\paragraph\{', tex):
            fail("[%s] a \\paragraph would demote past the deepest level" % label)
        tex = re.sub(r'(?m)^\\subsubsection\{', r'\\paragraph{', tex)
        tex = re.sub(r'(?m)^\\subsection\{', r'\\subsubsection{', tex)
        tex = re.sub(r'(?m)^\\section\{', r'\\subsection{', tex)
    ok("demoted %s by %d level(s): %d sections, %d subsections, %d subsubsections"
       % ((label, levels) + counts))
    return tex


# ------------------------------------------------------------------ macros --
# Carry the paper's own definitions across verbatim so no body text changes.
head = src[:cut(src, '\\begin{document}', 'begin{document}')]
macro_lines = [l for l in head.splitlines()
               if l.startswith(('\\newcommand{\\', '\\newtheorem{', '\\crefname{',
                                '\\Crefname{', '\\graphicspath{'))]
MACROS = "\n".join(macro_lines)
if MACROS.count('\\newcommand') != 16:
    fail("expected 16 \\newcommand macros, found %d" % MACROS.count('\\newcommand'))
ok("carried %d preamble definitions" % len(macro_lines))

# ------------------------------------------------------------- body slices --
i_intro = cut(src, '\\section{Introduction}', 'Introduction')
i_theory = cut(src, '\\section{Closed-form Baum--Welch for TKF}', 'first theory section')
i_results = cut(src, '\\section{Results: \\Pfam\\ fits and BAliBASE accuracies}', 'Results')
i_discuss = cut(src, '\\section{Discussion}', 'Discussion')
i_bib = cut(src, '\\newpage\n\\bibliographystyle{plainnat}', 'bibliography')
i_app = cut(src, '\\newpage\n\\appendix', 'appendix')
i_end = cut(src, '\\end{document}', 'end{document}')

if not (i_intro < i_theory < i_results < i_discuss < i_bib < i_app < i_end):
    fail("body slices are out of order")

intro = src[i_intro:i_theory]
theory = src[i_theory:i_results]
results = src[i_results:i_discuss]
discussion = src[i_discuss:i_bib]
appendix = src[i_app + len('\\newpage\n\\appendix'):i_end]
ok("sliced body into intro / theory / results / discussion / appendix")

# --------------------------------------------- 3. what happens to the appendix
# The source appendix had four sections.  Two are folded into the section of
# Materials and Methods they support; two are dropped.
M_SUBST = '\\section{The substitution model M-step}'
M_SVIBW = '\\section{\\svibw\\ implementation analyses}'
M_EXPL = ('% ==============================================================\n'
          '% Self-contained appendix: Exploded MixDom Pair HMM')
M_SCFG = ('\\section{Gravestone-augmented pair SCFG: sampling latent histories '
          'conditioned on observed alignments}')

a_subst = cut(appendix, M_SUBST, 'appendix: substitution M-step')
a_svibw = cut(appendix, M_SVIBW, 'appendix: svi-bw analyses')
a_expl = cut(appendix, M_EXPL, 'appendix: exploded MixDom')
a_scfg = cut(appendix, M_SCFG, 'appendix: gravestone SCFG')
if not (a_subst < a_svibw < a_expl < a_scfg):
    fail("appendix blocks are out of order")

svibw_block = appendix[a_svibw:a_expl]
scfg_block = appendix[a_scfg:]

# Dropped: "The substitution model M-step" restates the substitution-side
# M-step derived at length in the companion manuscript, and the exploded
# MixDom pair HMM is a state-by-state transition listing that nothing else in
# the text depends on.  Both pointers into them are removed with them.
DROP_SUBST_PTR = ('\nFor an analysis of the M-step for the innermost-level '
                  'substitution process, see Appendix~\\ref{appendix:subst-mstep}.\n')
DROP_EXPL_PTR = (' being non-empty\n(see Appendix~\\ref{appendix:exploded-mixdom}).')
for old, new, label in ((DROP_SUBST_PTR, '\n', 'substitution M-step pointer'),
                        (DROP_EXPL_PTR, ' being non-empty.', 'exploded MixDom pointer')):
    if theory.count(old) != 1:
        fail("could not find the %s to remove" % label)
    theory = theory.replace(old, new)
ok("dropped the substitution M-step and exploded MixDom appendices, "
   "and the two pointers into them")

# Folded in.  Each block drops one level here and another with the rest of the
# theory below, so its section ends up a subsubsection of the Materials and
# Methods subsection it belongs to.
svibw_block = demote(svibw_block, 'svi-bw analyses', (1, 2, 0))
scfg_block = demote(scfg_block, 'gravestone SCFG', (1, 2, 0))

# svi-bw analyses go at the end of the MixFrag section, where \svibw is
# introduced; the SCFG goes at the end of the gravestone section, which is the
# last section of the theory.
MIXDOM_HEAD = '\\section{MixDom: null-path collapse and an autodiff E-step}'
i_mixdom = cut(theory, MIXDOM_HEAD, 'MixDom heading')
theory = theory[:i_mixdom] + svibw_block + theory[i_mixdom:] + scfg_block
ok("folded the svi-bw and gravestone-SCFG appendices into the sections they support")

# ------------------------------------------------------- 2. demotion --------
theory = demote(theory, 'theory', (4, 13, 4))

methods = ('\\section{Materials and Methods}\n\\label{sec:methods}\n\n' + theory)

results = results.replace('\\section{Results: \\Pfam\\ fits and BAliBASE accuracies}',
                          '\\section{Results}', 1)
ok("renamed the Results heading")

# ------------------------------------- introduction roadmap cross-references -
# The introduction's roadmap named only the four theory sections.  Two
# subsections folded in from the appendix now sit under them, so the roadmap
# points at those too.  The wording is taken from the pointers already in the
# body.
ROADMAP = [
    ("then takes the same closed-form M-step on each minibatch's expected counts.\n",
     "then takes the same closed-form M-step on each minibatch's expected counts.\n"
     "\\Cref{appendix:svibw_appendix} analyzes aggregation methods and the\n"
     "precision of the resulting parameter estimates.\n",
     "svi-bw analyses"),
    ("\\Cref{sec:gravestone} describes the recovery of birth and death times \n"
     "for unobserved and partially observed lineages.\n",
     "\\Cref{sec:gravestone} describes the recovery of birth and death times \n"
     "for unobserved and partially observed lineages;\n"
     "\\Cref{appendix:sampling_scfg} gives the generating function that closes\n"
     "the corresponding sum, and the stochastic traceback that draws histories.\n",
     "gravestone SCFG"),
]
for old, new, label in ROADMAP:
    if intro.count(old) != 1:
        fail("could not find the introduction sentence to extend (%s)" % label)
    intro = intro.replace(old, new)
ok("extended the introduction's roadmap over the %d folded-in sections" % len(ROADMAP))

body = intro + methods + results + discussion

# ------------------------------------------ author-year citation commands ---
n_cite = len(re.findall(r'\\cite\{', body))
if n_cite != 22:
    fail("expected 22 \\cite{} commands, found %d" % n_cite)
body = re.sub(r'\\cite\{', r'\\citep{', body)
ok("converted %d \\cite -> \\citep (MBE cites by author and year)" % n_cite)

# --------------------------------------- appendix pointers -> cross-refs ----
n_app_ref = body.count('Appendix~\\ref{')
if n_app_ref != 3:
    fail("expected 3 'Appendix~\\ref{...}' pointers, found %d" % n_app_ref)
body = body.replace('Appendix~\\ref{', '\\Cref{')
ok("rewrote %d appendix pointers as ordinary cross-references" % n_app_ref)

# ----------------------------------------------------------- figure alt text -
# MBE requires an alt-text description under each figure legend.
ALT_TEXT = {
    'fig:tkf-trajectory': (
        "Alt text: A parse tree drawn above a time course. Each leaf of the "
        "tree sits directly above the residue lineage it generates in the "
        "trajectory below, where births, survivals and deaths are marked by "
        "open circles, filled circles and crosses."),
}

for label, alt in ALT_TEXT.items():
    marker = '\\label{%s}\n\\end{figure}' % label
    if body.count(marker) != 1:
        fail("could not place alt text: %r not found exactly once" % marker)
    body = body.replace(
        marker, '\\label{%s}\n\n{\\footnotesize\\itshape %s}\n\\end{figure}'
        % (label, alt))
ok("added alt text for %d figure(s)" % len(ALT_TEXT))

n_fig = len(re.findall(r'\\begin\{figure\}', body))
if n_fig != len(ALT_TEXT):
    fail("%d figures but %d alt-text entries" % (n_fig, len(ALT_TEXT)))

# --------------------------------------------------------------- abstract ----
# Taken verbatim from the source; no prose is rewritten here.  MBE allows 250
# words, and the source abstract is longer, so the overage is reported rather
# than fixed -- trimming it is an authorial decision, not a porting one.
i_abs = cut(src, '\\begin{abstract}', 'begin{abstract}') + len('\\begin{abstract}')
ABSTRACT = src[i_abs:cut(src, '\\end{abstract}', 'end{abstract}')].strip()

n_abs = words(ABSTRACT)
if n_abs <= 250:
    ok("abstract is %d words (MBE limit 250)" % n_abs)
else:
    print("WARN abstract is %d words; MBE allows 250 -- trim %d words before "
          "submitting" % (n_abs, n_abs - 250))

# The source carries these as a commented-out block; MBE prints key words.
m = re.search(r'%% Keywords: (.*?)\.\n', src, re.S)
if not m:
    fail("could not find the commented-out keyword list in the source")
KEYWORDS = re.sub(r'\s*\n%%\s*', ' ', m.group(1)).strip()
ok("key words: %s" % KEYWORDS)

# ------------------------------------------------------------------ preamble -
PREAMBLE = r"""%% Closed-form Baum-Welch for TKF-based evolutionary models
%% Molecular Biology and Evolution -- Methods manuscript, initial submission.
%%
%% GENERATED FILE.  Produced from
%%   ../../neurips-stody-paper/paper1-baumwelch/neurips_workshop_paper1.tex
%% by port-from-neurips.py.  Edit the NeurIPS source and re-run that script;
%% direct edits here are overwritten.
%%
%% Formatting follows the "Manuscript formatting" section of the MBE author
%% guidelines: one PDF carrying main text, references, tables, figures and
%% captions; line numbers; double spacing; ~25 mm margins.  There is no
%% separate supplementary file -- the former appendices are subsections of
%% Materials and Methods.

\documentclass[11pt]{article}

\usepackage[letterpaper,margin=25mm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

%% MBE cites by author and year, so natbib runs in author-year mode with the
%% comma between author and year suppressed: "(Thorne et al. 1991)".
\usepackage[round,authoryear]{natbib}
\setcitestyle{aysep={}}

\usepackage{setspace}
\usepackage{amsmath,amssymb,amsfonts,bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{makecell}
\usepackage{array}
\usepackage{enumitem}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage[mathlines]{lineno}
\usepackage{cleveref}   % must be loaded after hyperref

%% Folding the appendices into Materials and Methods puts a few headings at
%% the fourth level.  Number them, and set them as blocks rather than as
%% run-in paragraph headings.  (The source uses no \paragraph of its own.)
\setcounter{secnumdepth}{4}
\setcounter{tocdepth}{4}
\makeatletter
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
  {-3.25ex \@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\normalfont\normalsize\bfseries}}
\makeatother

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

%% cleveref calls every sectioning unit a "section", which is what the prose
%% assumes now that the theory sits under Materials and Methods.
\crefname{subsection}{section}{sections}
\Crefname{subsection}{Section}{Sections}
\crefname{subsubsection}{section}{sections}
\Crefname{subsubsection}{Section}{Sections}
\crefname{paragraph}{section}{sections}
\Crefname{paragraph}{Section}{Sections}

%% ---- macros carried over unchanged from the NeurIPS source ----
__MACROS__

\begin{document}
\doublespacing
\linenumbers

%% ------------------------------------------------------------ cover page ---
\begin{singlespace}
\thispagestyle{empty}
\begin{center}
{\normalsize Article type: Methods\par}
\vspace{1.5em}
{\Large\bfseries Closed-form Baum--Welch for TKF-based evolutionary models\par}
\vspace{1.5em}
{\large Ian Holmes$^{1,\ast}$ and Annabel Large$^{1}$\par}
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

This study generated no new sequence data. It uses two public resources: the
\Pfam\ protein families database \citep{Pfam2021}, from which the training and
validation splits of \Cref{sec:results} were drawn, and the BAliBASE 3
reference alignment benchmark \citep{Thompson1999}. Source code for all models
and trainers described here, together with the trained checkpoints and the
scripts that regenerate every table in this paper, is available at
\url{https://github.com/evoldoers/tkf-mixdom} \citep{TKFMixDomCode}.

% =============================================================
\section*{Acknowledgements}
% =============================================================

We thank Jeff Thorne, Jotun Hein, Marc Suchard, Nicola De Maio, Elena Rivas,
Sean Eddy, and David MacKay. Some of the mathematical derivations, exposition
and code in this work were drafted with the assistance of large language models
(Anthropic's Claude and OpenAI's ChatGPT) and were then checked and edited by the
authors, who take full responsibility for the content.

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
       .replace('__ABSTRACT__', ABSTRACT)
       .replace('__KEYWORDS__', KEYWORDS)
       + body + BACKMATTER)

DST.write_text(out)
ok("wrote mbe_paper1.tex (%d bytes)" % len(out))
print("\nmain-text word count (body only, excluding abstract and references): ~%d"
      % words(body))
