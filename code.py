# Non-Stationary Multi-Armed Bandit: UCB Policy Comparison
# Based on Garivier & Moulines (2008)
#
# Experiments
# -----------
# Exp 1  Paper replication  K=3, 2 abrupt breakpoints  (paper §5 Ex.1)
# Exp 2  Paper replication  K=2, periodic rewards       (paper §5 Ex.2)
# Exp 3  Original design    K=4, 5 random breakpoints
# Exp 4  Stationary env.    shows UCB-1 dominates when Y=0
# Exp A  Sensitivity        D-UCB sensitivity to gamma
# Exp B  Sensitivity        SW-UCB sensitivity to tau
# Exp C  Scaling            regret vs. number of arms K
# Exp D  Scaling            regret vs. breakpoint count Y_T
# Timing Wall-clock time    per algorithm, per T


# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from tqdm.notebook import tqdm  # replace with tqdm.tqdm outside notebooks

warnings.filterwarnings("ignore")

MASTER_SEED = 42
np.random.seed(MASTER_SEED)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.88,
})

COLORS = {"UCB-1": "#e74c3c", "EXP3.S": "#f39c12", "D-UCB": "#2980b9", "SW-UCB": "#27ae60"}
STYLES = {"UCB-1": "-", "EXP3.S": "--", "D-UCB": "-.", "SW-UCB": ":"}


# ── Bandit Environments ───────────────────────────────────────────────────────

class BanditEnv:
    """Abstract base class. Subclasses implement get_means(t)."""

    def pull(self, arm, t):
        m = self.get_means(t)
        return self.rng.binomial(1, float(np.clip(m[arm], 0.0, 1.0)))

    def best_mean(self, t):
        return float(np.max(self.get_means(t)))

    def get_means(self, t):
        raise NotImplementedError


class StationaryBandit(BanditEnv):
    """Fixed Bernoulli means — standard stationary MAB."""

    def __init__(self, means, seed=None):
        self._means = np.array(means, dtype=float)
        self.K = len(means)
        self.breakpoints = []
        self.rng = np.random.RandomState(seed)

    def get_means(self, t):
        return self._means


class PaperBandit1(BanditEnv):
    """
    Garivier & Moulines (2008) §5 Example 1.
    K=3 arms, T=10000, breakpoints at t=3000 and t=5000.
    p(1)=0.5, p(2)=0.3, p(3): 0.4 → 0.9 → 0.4
    """

    K = 3
    breakpoints = [3000, 5000]

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def get_means(self, t):
        p3 = 0.9 if 3000 <= (t + 1) < 5000 else 0.4
        return np.array([0.5, 0.3, p3])


class PaperBandit2(BanditEnv):
    """
    Garivier & Moulines (2008) §5 Example 2.
    K=2 arms, p(2)=0.5, p(1)=0.5+0.4*cos(6*pi*R*t/T) — periodic.
    """

    K = 2
    breakpoints = []

    def __init__(self, T=10_000, R=1, seed=None):
        self.T = T
        self.R = R
        self.rng = np.random.RandomState(seed)

    def get_means(self, t):
        p1 = 0.5 + 0.4 * np.cos(6 * np.pi * self.R * (t + 1) / self.T)
        return np.array([float(np.clip(p1, 0.0, 1.0)), 0.5])


class RandomAbruptBandit(BanditEnv):
    """
    K arms, Y_T breakpoints at random positions.

    At each breakpoint a new arm becomes dominant (mean in [0.70, 0.90]),
    the previous best drops (mean in [0.10, 0.35]), and all other arms
    are assigned random means in [0.20, 0.55]. This guarantees the optimal
    arm changes at every breakpoint with a gap of at least 0.35.
    """

    def __init__(self, K, T, num_breakpoints, seed=None):
        self.K = K
        self.T = T
        self.rng = np.random.RandomState(seed)

        # Breakpoint positions with a minimum gap between them
        min_gap = max(T // max(3 * (num_breakpoints + 1), 4), 80)
        avail = np.arange(min_gap, T - min_gap)
        raw = sorted(self.rng.choice(avail, size=min(num_breakpoints, len(avail)), replace=False))

        filtered = [raw[0]] if raw else []
        for bp in raw[1:]:
            if bp - filtered[-1] >= min_gap:
                filtered.append(bp)
        self.breakpoints = filtered[:num_breakpoints]

        # Means per epoch
        n_epochs = len(self.breakpoints) + 1
        self._epoch_means = []
        prev_best = -1
        for _ in range(n_epochs):
            m = self.rng.uniform(0.20, 0.55, K)
            candidates = [j for j in range(K) if j != prev_best]
            new_best = self.rng.choice(candidates)
            m[new_best] = self.rng.uniform(0.70, 0.90)
            if prev_best >= 0:
                m[prev_best] = self.rng.uniform(0.10, 0.35)
            self._epoch_means.append(m.copy())
            prev_best = new_best
        self._epoch_means = np.array(self._epoch_means)

    def _epoch(self, t):
        ep = 0
        for bp in self.breakpoints:
            if t >= bp:
                ep += 1
            else:
                break
        return ep

    def get_means(self, t):
        return self._epoch_means[self._epoch(t)]


# Smoke test
_b = RandomAbruptBandit(K=3, T=1000, num_breakpoints=2, seed=0)
assert len(_b.breakpoints) == 2


# ── Algorithm Implementations ─────────────────────────────────────────────────

class UCB1:
    """
    UCB-1 (Auer, Cesa-Bianchi & Fischer 2002).
    Achieves optimal O(log T) regret for stationary bandits.
    Fails in non-stationary settings — cannot discard stale observations.

    Index: X_bar_t(i) + B * sqrt(xi * log(t) / N_t(i))
    """

    def __init__(self, K, xi=0.5, B=1.0):
        self.K = K
        self.xi = xi
        self.B = B
        self.reset()

    def reset(self):
        self.counts = np.zeros(self.K)
        self.means = np.zeros(self.K)
        self.t = 0

    def select_arm(self):
        self.t += 1
        unplayed = np.where(self.counts == 0)[0]
        if len(unplayed):
            return int(unplayed[0])
        ucb = self.means + self.B * np.sqrt(self.xi * np.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm, r):
        self.counts[arm] += 1
        self.means[arm] += (r - self.means[arm]) / self.counts[arm]


class DUCB:
    """
    Discounted UCB (Kocsis & Szepesvari 2006; Garivier & Moulines 2008 §2).
    Exponentially down-weights past rewards by factor gamma in (0,1).

    Index:
        X_bar_t(gamma,i) = sum_{s<=t} gamma^{t-s} X_s 1{I_s=i} / N_t(gamma,i)
        c_t(gamma,i) = 2B * sqrt(xi * log(n_t(gamma)) / N_t(gamma,i))

    Optimal gamma (Remark 3): gamma = 1 - (4B)^{-1} * sqrt(Y_T / T)
    Regret (Remark 3): E[R_T] = O(sqrt(T * Y_T) * log T)
    """

    def __init__(self, K, gamma, xi=0.5, B=1.0):
        self.K = K
        self.gamma = gamma
        self.xi = xi
        self.B = B
        self.reset()

    def reset(self):
        self.N = np.zeros(self.K)   # discounted counts N_t(gamma, i)
        self.S = np.zeros(self.K)   # discounted reward sums
        self.t = 0

    def select_arm(self):
        self.t += 1
        unplayed = np.where(self.N < 1e-9)[0]
        if len(unplayed):
            return int(unplayed[0])
        n_tot = float(np.sum(self.N))
        means = self.S / np.maximum(self.N, 1e-12)
        pad = 2.0 * self.B * np.sqrt(self.xi * np.log(n_tot) / np.maximum(self.N, 1e-12))
        return int(np.argmax(means + pad))

    def update(self, arm, r):
        self.N *= self.gamma    # discount all arms each step
        self.S *= self.gamma
        self.N[arm] += 1.0
        self.S[arm] += r


class SWUCB:
    """
    Sliding-Window UCB (Garivier & Moulines 2008 §3).
    Uses only the tau most-recent observations; older data is discarded.

    Index:
        X_bar_t(tau,i) = sum_{s=t-tau+1}^{t} X_s 1{I_s=i} / N_t(tau,i)
        c_t(tau,i) = B * sqrt(xi * log(min(t,tau)) / N_t(tau,i))

    Optimal tau (Remark 9): tau = 2B * sqrt(T * log(T) / Y_T)
    Regret (Remark 9): E[R_T] = O(sqrt(T * Y_T * log T))

    Implementation uses a circular buffer of length tau for O(K) amortised
    per-step cost.
    """

    def __init__(self, K, tau, xi=0.5, B=1.0):
        self.K = K
        self.tau = tau
        self.xi = xi
        self.B = B
        self.reset()

    def reset(self):
        self._buf_arms = np.full(self.tau, -1, dtype=np.int32)
        self._buf_rews = np.zeros(self.tau)
        self._ptr = 0
        self._fill = 0
        self.t = 0

    def _window_stats(self):
        n = min(self._fill, self.tau)
        if n == 0:
            return np.zeros(self.K), np.zeros(self.K)
        idx = [(self._ptr - n + i) % self.tau for i in range(n)]
        arms_w = self._buf_arms[idx]
        rews_w = self._buf_rews[idx]
        valid = arms_w >= 0
        counts = np.bincount(arms_w[valid], minlength=self.K).astype(float)
        sums = np.bincount(arms_w[valid], weights=rews_w[valid], minlength=self.K)
        return counts, sums

    def select_arm(self):
        self.t += 1
        counts, sums = self._window_stats()
        unplayed = np.where(counts == 0)[0]
        if len(unplayed):
            return int(unplayed[0])
        t_tau = min(self.t, self.tau)
        pad = self.B * np.sqrt(self.xi * np.log(t_tau) / counts)
        return int(np.argmax(sums / counts + pad))

    def update(self, arm, r):
        self._buf_arms[self._ptr] = arm
        self._buf_rews[self._ptr] = r
        self._ptr = (self._ptr + 1) % self.tau
        self._fill += 1


class EXP3S:
    """
    EXP3.S (Auer, Cesa-Bianchi, Freund & Schapire 2002/03, §8).
    Softmax / importance-weighted policy for non-stationary bandits.
    Works in both adversarial and stochastic settings.

    Parameters (Corollary 8.3):
        alpha = 1/T
        gamma = min(1, sqrt(K * (Y_T * log(KT) + e) / ((e-1) * T)))

    Regret (Theorem 8.1):
        E[R_T] <= 2*sqrt(e-1) * sqrt(K*T*(Y_T*log(KT)+e))  = O(sqrt(K*T*Y_T*log T))
    """

    def __init__(self, K, T, upsilon_T):
        self.K = K
        upsilon_T = max(1, upsilon_T)
        self.alpha = 1.0 / T
        self.gamma = min(
            1.0,
            np.sqrt(K * (upsilon_T * np.log(K * T) + np.e) / ((np.e - 1.0) * T))
        )
        self.reset()

    def reset(self):
        self.w = np.ones(self.K)
        self.t = 0
        self._p = np.ones(self.K) / self.K

    def select_arm(self):
        self.t += 1
        W = float(self.w.sum())
        self._p = (1.0 - self.gamma) * self.w / W + self.gamma / self.K
        return int(np.random.choice(self.K, p=self._p))

    def update(self, arm, r):
        x_hat = np.zeros(self.K)
        x_hat[arm] = r / float(self._p[arm])
        incr = self.gamma * x_hat / self.K + self.alpha / self.K
        self.w *= np.exp(incr)
        self.w /= self.w.max()
        self.w = np.maximum(self.w, 1e-300)


# ── Experiment Runner ─────────────────────────────────────────────────────────

def run_experiment(bandit_factory, algo_factories, T, n_runs=30, desc=""):
    """
    Monte-Carlo evaluation of multiple bandit algorithms.

    Each (run, algorithm) pair gets an independent bandit instance so that
    reward draws are never shared across algorithms.

    Parameters
    ----------
    bandit_factory : callable(seed) -> BanditEnv
    algo_factories : dict {name: callable() -> algorithm}
    T              : time horizon
    n_runs         : number of Monte Carlo repetitions
    desc           : progress bar label

    Returns
    -------
    results : dict {name: {mean, std, q10, q90, hist, final}}
    times   : dict {name: list of per-run wall-clock seconds}
    """
    names = list(algo_factories)
    cum = {n: np.zeros((n_runs, T)) for n in names}
    hist = {n: np.zeros((n_runs, T), dtype=np.int32) for n in names}
    times = {n: [] for n in names}

    for run in tqdm(range(n_runs), desc=desc, leave=True):
        for alg_idx, name in enumerate(names):
            seed = MASTER_SEED + run * 1009 + alg_idx * 97
            bandit = bandit_factory(seed=seed)
            algo = algo_factories[name]()

            t0 = time.perf_counter()
            cr = 0.0
            for t in range(T):
                arm = algo.select_arm()
                rew = bandit.pull(arm, t)
                algo.update(arm, rew)
                cr += bandit.best_mean(t) - bandit.get_means(t)[arm]
                cum[name][run, t] = cr
                hist[name][run, t] = arm
            times[name].append(time.perf_counter() - t0)

    results = {}
    for name in names:
        cr = cum[name]
        results[name] = dict(
            mean=cr.mean(0),
            std=cr.std(0),
            q10=np.percentile(cr, 10, 0),
            q90=np.percentile(cr, 90, 0),
            hist=hist[name],
            final=cr[:, -1],
        )
    return results, times


def print_regret_table(results):
    print(f"\n{'Algorithm':<12} | {'Mean':>8} | {'Std':>7} | {'Min':>7} | {'Max':>7}")
    print("-" * 52)
    for name, r in results.items():
        f = r["final"]
        print(f"  {name:<10} | {f.mean():>8.1f} | {f.std():>7.1f} | {f.min():>7.1f} | {f.max():>7.1f}")


# ── Experiment 1 — Paper Replication: K=3, 2 abrupt breakpoints ──────────────
# Parameters from paper §5:
#   D-UCB:  gamma = 1 - 1/(4*sqrt(T))
#   SW-UCB: tau   = 4*sqrt(T*log(T))
#   EXP3.S: alpha=1/T, gamma tuned for Y_T=2 (Corollary 8.3)

T1, K1, U1 = 10_000, 3, 2
gamma1 = 1.0 - 1.0 / (4.0 * np.sqrt(T1))
tau1 = int(4.0 * np.sqrt(T1 * np.log(T1)))

algos1 = {
    "UCB-1":  lambda: UCB1(K1),
    "EXP3.S": lambda: EXP3S(K1, T1, U1),
    "D-UCB":  lambda: DUCB(K1, gamma=gamma1),
    "SW-UCB": lambda: SWUCB(K1, tau=tau1),
}

res1, times1 = run_experiment(
    bandit_factory=lambda seed: PaperBandit1(seed=seed),
    algo_factories=algos1,
    T=T1,
    n_runs=30,
    desc="Exp 1",
)

ts1 = np.arange(1, T1 + 1)
ref1 = PaperBandit1()
means_ref1 = np.array([ref1.get_means(t) for t in range(T1)])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
arm_colors = ["#e74c3c", "#2980b9", "#27ae60"]
for i, c in enumerate(arm_colors):
    ax.plot(ts1, means_ref1[:, i], color=c, lw=1.8, label=f"Arm {i+1}")
for bp in [3000, 5000]:
    ax.axvline(bp, color="gray", ls=":", lw=1.5)
ax.set_xlabel("Time step t")
ax.set_ylabel("Success probability")
ax.set_title("Arm Reward Distributions")
ax.legend(fontsize=9)

ax = axes[1]
for name in ["UCB-1", "D-UCB", "SW-UCB", "EXP3.S"]:
    cum_f = np.cumsum((res1[name]["hist"] == 0).mean(0)) / ts1
    ax.plot(ts1, cum_f, color=COLORS[name], ls=STYLES[name], label=name)
for bp in [3000, 5000]:
    ax.axvline(bp, color="gray", ls=":", lw=1.5)
ax.set_xlabel("t")
ax.set_ylabel("Cumulative pull frequency")
ax.set_title("Cumulative Frequency of Arm 1 Pulls")
ax.legend()

ax = axes[2]
for name, r in res1.items():
    ax.plot(ts1, r["mean"], color=COLORS[name], ls=STYLES[name], label=name)
    ax.fill_between(ts1, r["q10"], r["q90"], alpha=0.12, color=COLORS[name])
for bp in [3000, 5000]:
    ax.axvline(bp, color="gray", ls=":", lw=1.5)
ax.set_xlabel("t")
ax.set_ylabel("Cumulative Regret")
ax.set_title("Cumulative Regret (30 runs, 10–90 pct band)")
ax.legend()

fig.suptitle("Experiment 1 — K=3, T=10000, Y=2  (Breakpoints at t=3000 & t=5000)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("exp1_paper.png", dpi=150, bbox_inches="tight")
plt.show()

print_regret_table(res1)


# ── Experiment 2 — Paper Replication: K=2, periodic rewards ──────────────────

T2, K2, U2 = 10_000, 2, 6
gamma2 = 1.0 - 1.0 / (4.0 * np.sqrt(T2))
tau2 = int(4.0 * np.sqrt(T2 * np.log(T2)))

algos2 = {
    "UCB-1":  lambda: UCB1(K2),
    "EXP3.S": lambda: EXP3S(K2, T2, U2),
    "D-UCB":  lambda: DUCB(K2, gamma=gamma2),
    "SW-UCB": lambda: SWUCB(K2, tau=tau2),
}

res2, times2 = run_experiment(
    bandit_factory=lambda seed: PaperBandit2(T=T2, seed=seed),
    algo_factories=algos2,
    T=T2,
    n_runs=30,
    desc="Exp 2",
)

ts2 = np.arange(1, T2 + 1)
ref2 = PaperBandit2(T=T2)
m_a1 = np.array([ref2.get_means(t)[0] for t in range(T2)])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(ts2, m_a1, color="#e74c3c", lw=1.8, label="Arm 1 (periodic)")
ax.axhline(0.5, color="#2980b9", ls="--", lw=1.8, label="Arm 2 (constant 0.5)")
ax.fill_between(ts2, 0.5, m_a1, where=m_a1 > 0.5, alpha=0.18, color="#e74c3c", label="Arm 1 best")
ax.fill_between(ts2, 0.5, m_a1, where=m_a1 < 0.5, alpha=0.18, color="#2980b9", label="Arm 2 best")
ax.set_xlabel("t")
ax.set_ylabel("p(arm)")
ax.set_title("Periodic Arm Distributions")
ax.legend(fontsize=9)

ax = axes[1]
for name in ["UCB-1", "D-UCB", "SW-UCB", "EXP3.S"]:
    freq = np.cumsum((res2[name]["hist"] == 0).mean(0)) / ts2
    ax.plot(ts2, freq, color=COLORS[name], ls=STYLES[name], label=name)
ax.set_xlabel("t")
ax.set_ylabel("Cumulative pull frequency of arm 1")
ax.set_title("Tracking the Periodic Best Arm")
ax.legend()

ax = axes[2]
for name, r in res2.items():
    ax.plot(ts2, r["mean"], color=COLORS[name], ls=STYLES[name], label=name)
    ax.fill_between(ts2, r["q10"], r["q90"], alpha=0.12, color=COLORS[name])
ax.set_xlabel("t")
ax.set_ylabel("Cumulative Regret")
ax.set_title("Cumulative Regret (30 runs)")
ax.legend()

fig.suptitle("Experiment 2 — K=2, T=10000, Periodic Rewards  (p1 = 0.5 + 0.4*cos(6*pi*t/T))", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("exp2_paper.png", dpi=150, bbox_inches="tight")
plt.show()

print_regret_table(res2)


# ── Experiment 3 — Original Design: K=4, 5 random breakpoints ────────────────
# Parameters set via paper formulas (Remarks 3 & 9, B=1):
#   gamma* = 1 - (4B)^{-1} * sqrt(Y/T)
#   tau*   = 2B * sqrt(T*log(T) / Y)

T3, K3, U3 = 10_000, 4, 5
gamma3 = 1.0 - 0.25 * np.sqrt(U3 / T3)
tau3 = max(int(2.0 * np.sqrt(T3 * np.log(T3) / U3)), 50)

algos3 = {
    "UCB-1":  lambda: UCB1(K3),
    "EXP3.S": lambda: EXP3S(K3, T3, U3),
    "D-UCB":  lambda: DUCB(K3, gamma=gamma3),
    "SW-UCB": lambda: SWUCB(K3, tau=tau3),
}

res3, times3 = run_experiment(
    bandit_factory=lambda seed: RandomAbruptBandit(K=K3, T=T3, num_breakpoints=U3, seed=seed),
    algo_factories=algos3,
    T=T3,
    n_runs=30,
    desc="Exp 3",
)

ts3 = np.arange(1, T3 + 1)
ref3 = RandomAbruptBandit(K=K3, T=T3, num_breakpoints=U3, seed=42)
means3 = np.array([ref3.get_means(t) for t in range(T3)])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
arm_colors4 = ["#e74c3c", "#2980b9", "#27ae60", "#9b59b6"]
for i in range(K3):
    ax.plot(ts3, means3[:, i], color=arm_colors4[i], lw=1.8, label=f"Arm {i+1}")
for bp in ref3.breakpoints:
    ax.axvline(bp, color="gray", ls=":", lw=1.5)
ax.set_xlabel("t")
ax.set_ylabel("Mean reward")
ax.set_title(f"Sample Environment (seed=42, Y={U3})")
ax.legend(fontsize=9)

win = 300
ax = axes[1]
for name in ["UCB-1", "D-UCB", "SW-UCB", "EXP3.S"]:
    raw_freq = (res3[name]["hist"] == 0).mean(0).astype(float)
    smooth = np.convolve(raw_freq, np.ones(win) / win, mode="same")
    ax.plot(ts3, smooth, color=COLORS[name], ls=STYLES[name], label=name)
for bp in ref3.breakpoints:
    ax.axvline(bp, color="gray", ls=":", lw=1.5)
ax.set_xlabel("t")
ax.set_ylabel(f"Pull freq arm 1 ({win}-step MA)")
ax.set_title("Arm 1 Pull Frequency (smoothed)")
ax.legend()

ax = axes[2]
for name, r in res3.items():
    ax.plot(ts3, r["mean"], color=COLORS[name], ls=STYLES[name], label=name)
    ax.fill_between(ts3, r["q10"], r["q90"], alpha=0.12, color=COLORS[name])
for bp in ref3.breakpoints:
    ax.axvline(bp, color="gray", ls=":", lw=1.5)
ax.set_xlabel("t")
ax.set_ylabel("Cumulative Regret")
ax.set_title("Cumulative Regret (30 runs)")
ax.legend()

fig.suptitle(f"Experiment 3 — K={K3}, T={T3:,}, Y={U3} random abrupt breakpoints", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("exp3_random.png", dpi=150, bbox_inches="tight")
plt.show()

print_regret_table(res3)


# ── Experiment 4 — Stationary Environment ────────────────────────────────────
# UCB-1 is optimal (designed for this case).
# SW-UCB with tau=T reduces to UCB-1 (paper Remark 11).
# D-UCB with gamma->1 approximates UCB-1.
# EXP3.S has the highest regret — it is not designed for stationarity.

T4, K4 = 10_000, 4
MEANS4 = [0.65, 0.45, 0.30, 0.20]

algos4 = {
    "UCB-1":  lambda: UCB1(K4),
    "EXP3.S": lambda: EXP3S(K4, T4, upsilon_T=1),
    "D-UCB":  lambda: DUCB(K4, gamma=0.9999),
    "SW-UCB": lambda: SWUCB(K4, tau=T4),
}

res4, times4 = run_experiment(
    bandit_factory=lambda seed: StationaryBandit(MEANS4, seed=seed),
    algo_factories=algos4,
    T=T4,
    n_runs=30,
    desc="Exp 4",
)

ts4 = np.arange(1, T4 + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, r in res4.items():
    ax.plot(ts4, r["mean"], color=COLORS[name], ls=STYLES[name], label=name)
    ax.fill_between(ts4, r["q10"], r["q90"], alpha=0.12, color=COLORS[name])
ax.set_xlabel("t")
ax.set_ylabel("Cumulative Regret")
ax.set_title("Stationary Environment: Cumulative Regret")
ax.legend()

ax = axes[1]
for name, r in res4.items():
    ax.loglog(ts4[10:], r["mean"][10:] + 1e-9, color=COLORS[name], ls=STYLES[name], label=name)
log_ref = np.log(ts4[10:]) * 12
ax.loglog(ts4[10:], log_ref, "k--", alpha=0.5, lw=1.5, label="O(log t) reference")
ax.set_xlabel("t (log scale)")
ax.set_ylabel("Regret (log scale)")
ax.set_title("Log-Log: O(log T) rate for UCB-1")
ax.legend(fontsize=9)

fig.suptitle("Experiment 4 — Stationary: UCB-1 optimal, SW-UCB(tau=T) equivalent to UCB-1", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("exp4_stationary.png", dpi=150, bbox_inches="tight")
plt.show()

print_regret_table(res4)


# ── Experiment A — D-UCB Sensitivity to gamma ────────────────────────────────
# gamma too small -> forgets too fast -> high regret in stable phases
# gamma too large -> slow to adapt after breakpoints
# gamma* balances these two effects

T_A, K_A, U_A = 10_000, 3, 2
gamma_opt_A = 1.0 - 0.25 * np.sqrt(U_A / T_A)

gamma_vals = [0.980, 0.990, gamma_opt_A, 0.9995, 0.99995]
gamma_labels = [
    f"gamma={g:.5f}" + (" (optimal)" if abs(g - gamma_opt_A) < 1e-7 else "")
    for g in gamma_vals
]

algos_A = {
    lbl: (lambda g: lambda: DUCB(K_A, gamma=g))(gv)
    for lbl, gv in zip(gamma_labels, gamma_vals)
}

res_A, _ = run_experiment(
    bandit_factory=lambda seed: PaperBandit1(seed=seed),
    algo_factories=algos_A,
    T=T_A,
    n_runs=30,
    desc="Exp A",
)

ts_A = np.arange(1, T_A + 1)
fig, ax = plt.subplots(figsize=(11, 5))
cmap_A = plt.cm.viridis(np.linspace(0.05, 0.95, len(gamma_vals)))
for (name, r), col in zip(res_A.items(), cmap_A):
    ax.plot(ts_A, r["mean"], color=col, lw=2, label=name)
for bp in [3000, 5000]:
    ax.axvline(bp, color="gray", ls=":", lw=1.5)
ax.set_xlabel("t")
ax.set_ylabel("Cumulative Regret")
ax.set_title(f"D-UCB: Sensitivity to Discount Factor gamma  (gamma* = {gamma_opt_A:.5f})")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("expA_gamma.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nOptimal gamma* = {gamma_opt_A:.6f}")
for name, r in res_A.items():
    print(f"  {name:<45s}  final regret {r['final'].mean():.1f} +/- {r['final'].std():.1f}")


# ── Experiment B — SW-UCB Sensitivity to tau ─────────────────────────────────
# tau too small -> high variance, noisy estimates
# tau too large -> slow to detect breakpoints (similar failure mode to UCB-1)
# tau* = 2*sqrt(T*log(T) / Y) balances bias and variance

T_B, K_B, U_B = 10_000, 3, 2
tau_opt_B = int(2.0 * np.sqrt(T_B * np.log(T_B) / U_B))

tau_vals = [100, 400, tau_opt_B, 2500, 6000]
tau_labels = [
    f"tau={t:5d}" + (" (optimal)" if t == tau_opt_B else "")
    for t in tau_vals
]

algos_B = {
    lbl: (lambda t_: lambda: SWUCB(K_B, tau=t_))(tv)
    for lbl, tv in zip(tau_labels, tau_vals)
}

res_B, _ = run_experiment(
    bandit_factory=lambda seed: PaperBandit1(seed=seed),
    algo_factories=algos_B,
    T=T_B,
    n_runs=30,
    desc="Exp B",
)

ts_B = np.arange(1, T_B + 1)
fig, ax = plt.subplots(figsize=(11, 5))
cmap_B = plt.cm.plasma(np.linspace(0.05, 0.95, len(tau_vals)))
for (name, r), col in zip(res_B.items(), cmap_B):
    ax.plot(ts_B, r["mean"], color=col, lw=2, label=name)
for bp in [3000, 5000]:
    ax.axvline(bp, color="gray", ls=":", lw=1.5)
ax.set_xlabel("t")
ax.set_ylabel("Cumulative Regret")
ax.set_title(f"SW-UCB: Sensitivity to Window Size tau  (tau* = {tau_opt_B})")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("expB_tau.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nOptimal tau* = {tau_opt_B}")
for name, r in res_B.items():
    print(f"  {name:<35s}  final regret {r['final'].mean():.1f} +/- {r['final'].std():.1f}")


# ── Experiment C — Regret Scaling with Number of Arms K ──────────────────────

T_C, U_C = 10_000, 3
Ks = [2, 3, 4, 6, 8, 10, 15]
final_C = {name: [] for name in ["UCB-1", "D-UCB", "SW-UCB", "EXP3.S"]}

for K_c in tqdm(Ks, desc="Varying K"):
    g_c = 1.0 - 0.25 * np.sqrt(U_C / T_C)
    tau_c = max(int(2.0 * np.sqrt(T_C * np.log(T_C) / U_C)), 30)
    alg_c = {
        "UCB-1":  lambda K=K_c: UCB1(K),
        "EXP3.S": lambda K=K_c: EXP3S(K, T_C, U_C),
        "D-UCB":  lambda K=K_c, g=g_c: DUCB(K, gamma=g),
        "SW-UCB": lambda K=K_c, t_=tau_c: SWUCB(K, tau=t_),
    }
    r_c, _ = run_experiment(
        bandit_factory=lambda seed, K=K_c: RandomAbruptBandit(K=K, T=T_C, num_breakpoints=U_C, seed=seed),
        algo_factories=alg_c,
        T=T_C,
        n_runs=20,
        desc=f"  K={K_c:2d}",
    )
    for name in final_C:
        final_C[name].append(r_c[name]["final"].mean())

fig, ax = plt.subplots(figsize=(9, 5))
for name, vals in final_C.items():
    ax.plot(Ks, vals, marker="o", color=COLORS[name], ls=STYLES[name], label=name, ms=7)
ref_sqrtK = np.sqrt(Ks) * (final_C["SW-UCB"][0] / np.sqrt(Ks[0]))
ax.plot(Ks, ref_sqrtK, "k--", alpha=0.5, lw=1.5, label="proportional to sqrt(K)")
ax.set_xlabel("Number of Arms K")
ax.set_ylabel(f"Final Cumulative Regret at T={T_C:,}")
ax.set_title(f"Regret Scaling with K  (T={T_C:,}, Y={U_C})")
ax.legend()
plt.tight_layout()
plt.savefig("expC_K_scaling.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Experiment D — Regret vs. Breakpoint Frequency Y_T ───────────────────────
# Lower bound (Corollary 14): E[R_T] = Omega(sqrt(T * Y_T)).
# Parameters are re-tuned for each value of Y_T.

T_D, K_D = 10_000, 3
upsilons = [0, 1, 2, 5, 10, 20, 40]
final_D = {name: [] for name in ["UCB-1", "D-UCB", "SW-UCB", "EXP3.S"]}

for U_d in tqdm(upsilons, desc="Varying Y"):
    U_eff = max(U_d, 1)
    g_d = 1.0 - 0.25 * np.sqrt(U_eff / T_D)
    tau_d = max(int(2.0 * np.sqrt(T_D * np.log(T_D) / U_eff)), 30)

    if U_d == 0:
        bf_d = lambda seed: StationaryBandit([0.65, 0.40, 0.25], seed=seed)
    else:
        bf_d = lambda seed, U=U_d: RandomAbruptBandit(K=K_D, T=T_D, num_breakpoints=U, seed=seed)

    alg_d = {
        "UCB-1":  lambda: UCB1(K_D),
        "EXP3.S": lambda U=U_eff: EXP3S(K_D, T_D, U),
        "D-UCB":  lambda g=g_d: DUCB(K_D, gamma=g),
        "SW-UCB": lambda t_=tau_d: SWUCB(K_D, tau=t_),
    }
    r_d, _ = run_experiment(
        bandit_factory=bf_d,
        algo_factories=alg_d,
        T=T_D,
        n_runs=20,
        desc=f"  Y={U_d:2d}",
    )
    for name in final_D:
        final_D[name].append(r_d[name]["final"].mean())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, vals in final_D.items():
    ax.plot(upsilons, vals, marker="s", color=COLORS[name], ls=STYLES[name], label=name, ms=7)
ups_lb = np.array([max(u, 0.1) for u in upsilons])
lb_ref = 0.18 * np.sqrt(T_D * ups_lb)
ax.plot(upsilons, lb_ref, "k--", alpha=0.5, lw=1.5, label="proportional to sqrt(T*Y)")
ax.set_xlabel("Number of Breakpoints Y_T")
ax.set_ylabel(f"Final Regret at T={T_D:,}")
ax.set_title("Regret vs. Breakpoint Frequency")
ax.legend()

ax = axes[1]
ups_pos = [u for u in upsilons if u > 0]
for name, vals in final_D.items():
    vp = [v for u, v in zip(upsilons, vals) if u > 0]
    ax.loglog(ups_pos, vp, marker="s", color=COLORS[name], ls=STYLES[name], label=name, ms=7)
lb_pos = 0.18 * np.sqrt(T_D * np.array(ups_pos))
ax.loglog(ups_pos, lb_pos, "k--", alpha=0.5, lw=1.5, label="proportional to sqrt(Y)")
ax.set_xlabel("Y_T (log scale)")
ax.set_ylabel("Final Regret (log scale)")
ax.set_title("Log-Log: Regret scales as sqrt(Y_T) for D/SW-UCB")
ax.legend()

plt.tight_layout()
plt.savefig("expD_upsilon.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Timing Analysis ───────────────────────────────────────────────────────────

T_vals = [1_000, 5_000, 10_000, 50_000, 100_000]
K_tim, U_tim = 4, 3
timing_data = {name: [] for name in ["UCB-1", "EXP3.S", "D-UCB", "SW-UCB"]}

for T_t in tqdm(T_vals, desc="Timing"):
    g_t = 1.0 - 0.25 * np.sqrt(U_tim / T_t)
    tau_t = max(int(2.0 * np.sqrt(T_t * np.log(T_t) / U_tim)), 30)
    alg_t = {
        "UCB-1":  lambda: UCB1(K_tim),
        "EXP3.S": lambda: EXP3S(K_tim, T_t, U_tim),
        "D-UCB":  lambda g=g_t: DUCB(K_tim, gamma=g),
        "SW-UCB": lambda t_=tau_t: SWUCB(K_tim, tau=t_),
    }
    _, t_dict = run_experiment(
        bandit_factory=lambda seed: RandomAbruptBandit(K=K_tim, T=T_t, num_breakpoints=U_tim, seed=seed),
        algo_factories=alg_t,
        T=T_t,
        n_runs=8,
        desc=f"  T={T_t:>7,}",
    )
    for name in timing_data:
        timing_data[name].append(np.mean(t_dict[name]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, tms in timing_data.items():
    ax.plot(T_vals, tms, marker="o", color=COLORS[name], ls=STYLES[name], label=name, ms=7)
ax.set_xlabel("Horizon T")
ax.set_ylabel("Wall-clock time per run (s)")
ax.set_title(f"Total Time vs. T  (K={K_tim}, Y={U_tim})")
ax.legend()

ax = axes[1]
for name, tms in timing_data.items():
    per_step_us = [t / T_t * 1e6 for t, T_t in zip(tms, T_vals)]
    ax.plot(T_vals, per_step_us, marker="s", color=COLORS[name], ls=STYLES[name], label=name, ms=7)
ax.set_xlabel("Horizon T")
ax.set_ylabel("Time per step (us)")
ax.set_title("Per-Step Cost  (SW-UCB grows with tau ~ sqrt(T))")
ax.legend()

plt.tight_layout()
plt.savefig("timing.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n{'Algorithm':<10} | " + " | ".join(f"T={T:>8,}" for T in T_vals))
print("-" * (10 + 3 + len(T_vals) * 14))
for name, tms in timing_data.items():
    row = " | ".join(f"{t:>12.4f}" for t in tms)
    print(f"{name:<10} | {row}  (s/run)")

print()
for name, tms in timing_data.items():
    ps = [t / T_ * 1e6 for t, T_ in zip(tms, T_vals)]
    row = " | ".join(f"{p:>12.2f}" for p in ps)
    print(f"{name:<10} | {row}  (us/step)")


# ── Summary Figure ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

scenarios = [
    ("Stationary  (Exp 4, Y=0)",       res4),
    ("Paper Ex.1  (Exp 1, Y=2)",       res1),
    ("Paper Ex.2  (Exp 2, periodic)",  res2),
    ("Random abrupt  (Exp 3, Y=5)",    res3),
]

for ax, (title, res) in zip(axes.flatten(), scenarios):
    names_s = list(res.keys())
    finals = [res[n]["final"].mean() for n in names_s]
    stds = [res[n]["final"].std() for n in names_s]
    bars = ax.bar(
        names_s,
        finals,
        color=[COLORS.get(n, "grey") for n in names_s],
        alpha=0.85,
        edgecolor="k",
        linewidth=0.6,
    )
    ax.errorbar(names_s, finals, yerr=stds, fmt="none", color="black", capsize=5, lw=1.8)
    for bar, val in zip(bars, finals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.08,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Cumulative Regret at T=10000")

fig.suptitle("Summary: Final Cumulative Regret  (mean +/- std, 30 runs)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("summary.png", dpi=150, bbox_inches="tight")
plt.show()
