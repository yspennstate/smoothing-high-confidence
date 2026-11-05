"""Numerical check of the construction in the proof of Theorem 1.1.

For three noise laws with E||eta||^2 = d sigma^2 (sigma = 1) and the smallest d allowed by the theorem, this
builds rho (Lemma 3.1), s, B0 and v = -eps s (Lemma 3.2 and the proof) and measures, on a fresh sample,
q(B0) (must be >= 1/2) and q(v + B0) (must be <= alpha).

Every quantity in the construction depends on eta only through ||eta||_2^2 and X_s(eta) = <eta, s>/sqrt(d):
  eta in B0       iff ||eta||^2 <= rho^2 and X_s(eta) <= T
  eta in v + B0   iff ||eta + eps s||^2 = ||eta||^2 + 2 eps sqrt(d) X_s + eps^2 d <= rho^2 and X_s + eps sqrt(d) <= T
so each law is sampled through these two statistics.

  gaussian   N(0, I): X_s ~ N(0, 1) and ||eta||^2 = X_s^2 + chi^2_(d-1), exactly, for every s
  student5   multivariate t with 5 degrees of freedom scaled to unit second moment: eta = sqrt(W) g,
             W = 3 / chi^2_5 (not a product law; the coordinates are dependent and heavy-tailed)
  cube       uniform on [-sqrt 3, sqrt 3]^d, simulated coordinate by coordinate (the law is invariant under
             sign changes, so every s gives the same joint law and s = (1, ..., 1) is used)

    python check.py        (about a minute; ALL PASS at the end)
"""
import math
import sys
from statistics import NormalDist

import numpy as np

ALPHA = 0.24
SIGMA = 1.0
D = math.ceil(1012 * math.log(16 / ALPHA) / ALPHA ** 2)
T = SIGMA * math.sqrt(18 * math.log(24))
TP = SIGMA * math.sqrt(18 * math.log(16 / ALPHA))
EPS = (T + TP) / math.sqrt(D)
SQD = math.sqrt(D)


def stats_gaussian(rng, n):
    x = rng.standard_normal(n)
    return x * x + rng.chisquare(D - 1, n), x


def stats_student5(rng, n):
    w = 3.0 / rng.chisquare(5, n)
    x = rng.standard_normal(n)
    r2 = x * x + rng.chisquare(D - 1, n)
    return w * r2, np.sqrt(w) * x


def stats_cube(rng, n, chunk=50):
    a = math.sqrt(3.0)
    r2 = np.empty(n)
    xs = np.empty(n)
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        e = rng.uniform(-a, a, size=(m, D))
        r2[i:i + m] = np.einsum("ij,ij->i", e, e)
        xs[i:i + m] = e.sum(axis=1) / SQD
    return r2, xs


def build_rho(r2):
    """Lemma 3.1 on an empirical sample: the thinnest of the N shells in [2 sqrt d, 3 sqrt d]."""
    r = np.sqrt(r2)
    n_sh = int(math.floor(SIGMA / EPS))
    w = EPS * SQD
    starts = 2 * SIGMA * SQD + w * np.arange(n_sh)
    mass = np.array([np.mean((r > a) & (r <= a + w)) for a in starts])
    k = int(np.argmin(mass))
    return float(starts[k]), float(mass[k]), n_sh


def measure(r2, xs, rho):
    in_b0 = (r2 <= rho * rho) & (xs <= T)
    r2v = r2 + 2 * EPS * SQD * xs + EPS * EPS * D
    in_vb0 = (r2v <= rho * rho) & (xs + EPS * SQD <= T)
    return float(in_b0.mean()), float(in_vb0.mean())


def main():
    print("alpha %.2f  d %d  T %.3f  T' %.3f  eps %.5f  eps*sqrt(d) %.3f" % (ALPHA, D, T, TP, EPS, EPS * SQD))
    gauss_cert = NormalDist().inv_cdf(1 - ALPHA) / SQD
    print("theorem bound 8 sqrt(log(16/alpha))/sqrt(d) = %.5f; Gaussian certificate Phi^-1(1-alpha)/sqrt(d) = %.5f"
          % (8 * math.sqrt(math.log(16 / ALPHA)) / SQD, gauss_cert))
    ok_all = True
    laws = [("gaussian", stats_gaussian, 400000), ("student5", stats_student5, 400000), ("cube", stats_cube, 20000)]
    for name, fn, n in laws:
        rng_fit = np.random.default_rng(1)
        rng_test = np.random.default_rng(2)
        r2, _ = fn(rng_fit, n)
        print("  %-9s second moment / d = %.4f (sample of %d)" % (name, r2.mean() / D, n))
        rho, shell, n_sh = build_rho(r2)
        r2t, xst = fn(rng_test, n)
        qb0, qvb0 = measure(r2t, xst, rho)
        ok = qb0 >= 0.5 and qvb0 <= ALPHA and shell <= 2 * EPS / SIGMA
        ok_all &= ok
        print("  %-9s rho/sqrt(d) %.4f  shells %d  shell mass %.5f (<= 2eps/sigma = %.5f)  q(B0) %.4f  q(v+B0) %.6f  %s"
              % (name, rho / SQD, n_sh, shell, 2 * EPS / SIGMA, qb0, qvb0, "PASS" if ok else "FAIL"))
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
