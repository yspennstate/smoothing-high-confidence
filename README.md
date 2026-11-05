# Randomized smoothing at high confidence: the Gaussian is optimal for every noise law

Yitzchak Shmalo

This repository contains the paper and two short programs that go with it.

Randomized smoothing certifies a classifier by averaging a base classifier over a noise law $q$ on $\mathbb{R}^d$.
If $\mathbb{E}\|\eta\|_2^2\le d\sigma^2$ for $\eta\sim q$, then for $0<\alpha<1/4$ and
$d\ge 1012\,\alpha^{-2}\log(16/\alpha)$, no certificate that holds for every base classifier whose top class has
probability at least $1-\alpha$ exceeds $8\sigma\sqrt{\log(16/\alpha)}\,d^{1/p-1/2}$ in the $\ell_p$ norm,
$2\le p\le\infty$. The Gaussian $N(0,\sigma^2 I)$ certifies $\sigma\Phi^{-1}(1-\alpha)\,d^{1/p-1/2}$, so it is
optimal up to an absolute constant factor at every confidence level.

## Contents

- `paper/main.tex`, `paper/main.pdf`: the paper.
- `code/consts.py`: the constants in Theorem 1.1 and Corollary 1.2.
- `code/check.py`: a numerical check of the construction in the proof of Theorem 1.1.

## The checks

| Result in the paper | What is checked | Program |
|---|---|---|
| Theorem 1.1 | the constant $c=\sqrt{18\log 24}+\sqrt{18\log(16/\alpha)}$ over $\sqrt{\log(16/\alpha)}$ stays below 8, and $16c^2\le 1012$ | `consts.py` |
| Corollary 1.2 | the ratio of the bound to the Gaussian radius is at most 25 on $0<\alpha<1/4$ | `consts.py` |
| Lemmas 3.1, 3.2 and the proof of Theorem 1.1 | for three noise laws at the smallest admissible $d$ ($\alpha=0.24$, $d=73{,}787$): the shell radius $\rho$, the set $B_0$ and the shift $v=-\varepsilon s$ are built from one sample, and a fresh sample gives $q(B_0)\ge1/2$ and $q(v+B_0)\le\alpha$ | `check.py` |

The three laws in `check.py` are the standard Gaussian, a multivariate $t$ with five degrees of freedom scaled to
unit second moment (dependent, heavy-tailed coordinates), and the uniform law on the cube
$[-\sqrt3,\sqrt3]^d$. Membership in $B_0$ and in $v+B_0$ depends on $\eta$ only through $\|\eta\|_2^2$ and
$\langle\eta,s\rangle$, so the first two laws are sampled through these two statistics exactly and the cube is
sampled coordinate by coordinate.

## Running the checks

    python code/consts.py
    python code/check.py

Each takes under a minute and `check.py` prints `ALL PASS` at the end.

## Requirements

Python 3.10 or later, with `numpy`.
