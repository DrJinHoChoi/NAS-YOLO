% Simplified ECCV 2026 style (based on Springer LNCS llncs.cls)
% This is a minimal reproduction for paper drafting.
% For final submission, replace with official eccv2026.cls from the Author Kit.

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{eccv2026}[2026/01/01 ECCV 2026 Paper Style]

\DeclareOption{review}{\def\eccvreview{}}
\DeclareOption{final}{\def\eccvfinal{}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\LoadClass[10pt,twoside]{article}

% Page geometry (Springer LNCS style)
\RequirePackage[
  papersize={6.125in,9.25in},
  textheight={7.3in},
  textwidth={4.8in},
  centering,
  headheight=14pt
]{geometry}

% Essential packages
\RequirePackage{times}
\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage{xcolor}
\RequirePackage{hyperref}
\RequirePackage[capitalize]{cleveref}
\RequirePackage{algorithm}
\RequirePackage{algorithmic}
\RequirePackage{subcaption}
\RequirePackage{enumitem}
\RequirePackage{tabularx}

% Colors
\definecolor{eccvblue}{RGB}{0,51,102}

% Hyperref setup
\hypersetup{
  colorlinks=true,
  linkcolor=eccvblue,
  citecolor=eccvblue,
  urlcolor=eccvblue,
}

% Review mode: line numbers + anonymous
\ifdefined\eccvreview
  \RequirePackage{lineno}
  \linenumbers
  \renewcommand{\thefootnote}{\fnsymbol{footnote}}
\fi

% Title formatting
\renewcommand{\maketitle}{
  \begin{center}
    {\Large\bfseries \@title \par}
    \vspace{12pt}
    \ifdefined\eccvreview
      {\normalsize ECCV 2026 Submission \#XXXX. CONFIDENTIAL REVIEW COPY.\\
       DO NOT DISTRIBUTE.}
    \else
      {\normalsize \@author \par}
    \fi
    \vspace{12pt}
  \end{center}
}

% Abstract environment
\renewenvironment{abstract}{
  \begin{center}
    \begin{minipage}{0.9\textwidth}
    \small
    \textbf{Abstract.}
}{
    \end{minipage}
  \end{center}
  \vspace{12pt}
}

% Section formatting
\RequirePackage{titlesec}
\titleformat{\section}{\normalfont\large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalfont\normalsize\itshape}{\thesubsubsection}{1em}{}

% Compact spacing
\setlength{\parskip}{0pt}
\setlength{\parsep}{0pt}
\setlength{\headsep}{10pt}

% Paragraph indent
\setlength{\parindent}{1em}

% Caption formatting
\RequirePackage[font=small,labelfont=bf]{caption}

% Page style
\pagestyle{plain}

\endinput
