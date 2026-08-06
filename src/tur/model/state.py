"""The propagation model as code.

State entering each step: clean c, poisoned-by-syntax s, poisoned-by-semantics
m, summing to 1. Given a measured depth-varying baseline p_t and the syntactic
error share f_syn(t), the occupancies evolve as

    c_{t+1} = c_t p_t + s_t r_syn + m_t r_sem
    s_{t+1} = c_t (1 - p_t) f_syn(t) + s_t (1 - r_syn)
    m_{t+1} = c_t (1 - p_t) (1 - f_syn(t)) + m_t (1 - r_sem)

and the observed global per-step correctness is

    g_t = p_t (1 - pi * x_t),   x_t = s_t + m_t.

Net propagation loss L_t = 1 - g_t / p_t = pi * x_t is fit-free given measured
p_t and g_t.
"""

from __future__ import annotations

import numpy as np


def occupancies(p: np.ndarray, f_syn: np.ndarray, r_syn: float, r_sem: float):
    """Return arrays c, s, m over depths 1..T given measured p and f_syn."""
    T = len(p)
    c = np.zeros(T); s = np.zeros(T); m = np.zeros(T)
    c[0] = 1.0
    for t in range(T - 1):
        err = (1.0 - p[t])
        c[t + 1] = c[t] * p[t] + s[t] * r_syn + m[t] * r_sem
        s[t + 1] = c[t] * err * f_syn[t] + s[t] * (1.0 - r_syn)
        m[t + 1] = c[t] * err * (1.0 - f_syn[t]) + m[t] * (1.0 - r_sem)
    return c, s, m


def g_curve(p: np.ndarray, f_syn: np.ndarray, pi: float,
            r_syn: float, r_sem: float) -> np.ndarray:
    c, s, m = occupancies(p, f_syn, r_syn, r_sem)
    x = s + m
    return p * (1.0 - pi * x)


def simulate(p: np.ndarray, f_syn: np.ndarray, pi: float, r_syn: float,
             r_sem: float, n_tasks: int, rng: np.random.Generator):
    """Simulate per-(task, depth) global-correctness outcomes.

    Returns an integer array successes[T] and the trials[T] (= n_tasks), for
    feeding a binomial likelihood. Uses the analytic g_t as the per-step
    success probability, which is the population the fit should recover.
    """
    g = g_curve(p, f_syn, pi, r_syn, r_sem)
    successes = rng.binomial(n_tasks, np.clip(g, 0, 1))
    trials = np.full(len(p), n_tasks)
    return successes, trials, g
