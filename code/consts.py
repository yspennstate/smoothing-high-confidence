"""Constants in Theorem 1.1 and Corollary 1.2."""
from math import sqrt, log
from statistics import NormalDist
inv = NormalDist().inv_cdf
s18 = sqrt(18)
# (T+T')/sigma <= c * sqrt(log(16/alpha)) for alpha < 1/4
c = s18 * (sqrt(log(24)) / sqrt(log(64)) + 1)
print("c =", c)
# d condition: sqrt(d) >= 4 (T+T')/(alpha sigma)  <=  d >= 16 c^2 log(16/alpha)/alpha^2
print("16 c^2 =", 16 * c * c)
worst = max((8 * sqrt(log(16 / a)) / inv(1 - a), a) for a in [k / 10**6 for k in range(1, 250000)])
print("max ratio 8 sqrt(log(16/a)) / Phi^-1(1-a) on (0,1/4):", worst)
