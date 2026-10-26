# CWS-FM run log

## 2026-10-26: posterior parametrization

Question: do sampled sequences follow the training grammar (palindromes, L=16, V=32)? Rule: Gate 1 passes if
>= 50% of K=8 samples are valid palindromes, >= 90% are distinct, and an untrained control is < 5% valid.

| Change | K=8 validity | K=4 validity |
|---|---|---|
| Velocity regression, forced locking (before) | gate tautological: untrained model scored 0% collapse | n/a |
| Velocity regression, no forced lock, trapezoid band | 0% | 0% |
| Posterior (CE) parametrization + exact d_tau Euler, alpha=0.5 | 96.9% / 95.3% (seeds 0/1) | 73.4% / 64.1% |
| Same, alpha=0 (no wavefront) | 100% / 100% | 96.9% / 87.5% |

2000 steps x batch 64, CPU. Untrained control 0% valid in every run.

Bugs fixed: (1) the band indicator was sampled at grid points, so Heun integrated only 75% of each position's
phase; positions now step by exact d_tau. (2) Training zeroed the target at tau=0 where the sampler queries it.
(3) Velocity regression under-predicted speed; the model now predicts p(token | u, tau) and the velocity is
sum_k p_k Log_u(e_k) / (1 - tau), the exact marginal field of the geodesic conditional path.

Decision: the wavefront does not help on this task at fixed K (each position gets about K/2 steps in its band).
It must earn its place on real text (text8 / OWT vs MDLM, Fisher-FM) or be dropped.
