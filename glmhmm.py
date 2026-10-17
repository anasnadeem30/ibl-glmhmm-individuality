"""GLM-HMM with Bernoulli (logistic) emissions, fit by EM. Written for this project.

Model: hidden state z_t in {0..K-1}, Markov transitions A (K x K), initial dist pi.
Given state k, P(y_t = 1 | x_t) = sigmoid(W[k] . x_t). Sessions are independent sequences.
"""
import numpy as np
from numba import njit
from sklearn.linear_model import LogisticRegression


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def emission_logp(X, y, W):
    """log P(y_t | x_t, z_t=k) for every trial and state -> (T, K)."""
    a = X @ W.T                                   # (T, K) logits
    # log sigmoid(a) = -log1p(exp(-a)); numerically stable via logaddexp
    return -np.logaddexp(0, -a) * y[:, None] - np.logaddexp(0, a) * (1 - y[:, None])


@njit(cache=True)
def _fb(logB, A, pi):
    """Scaled forward-backward on one sequence. Returns gamma (T,K), xi_sum (K,K), loglik."""
    T, K = logB.shape
    B = np.exp(logB)
    alpha = np.zeros((T, K)); c = np.zeros(T)
    alpha[0] = pi * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        c[t] = alpha[t].sum(); alpha[t] /= c[t]
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[t + 1] * beta[t + 1]) / c[t + 1]
    gamma = alpha * beta
    xi = np.zeros((K, K))
    for t in range(T - 1):
        m = np.outer(alpha[t], B[t + 1] * beta[t + 1]) * A / c[t + 1]
        xi += m
    return gamma, xi, np.log(c).sum()


@njit(cache=True)
def _viterbi_free_loglik(logB, A, pi):
    """Forward pass only: log-likelihood of one sequence."""
    T, K = logB.shape
    B = np.exp(logB)
    a = pi * B[0]; s = a.sum(); ll = np.log(s); a /= s
    for t in range(1, T):
        a = (a @ A) * B[t]; s = a.sum(); ll += np.log(s); a /= s
    return ll


class GLMHMM:
    def __init__(self, K, n_iter=300, tol=1e-7, C=100.0, stickiness=0.95, seed=0):
        self.K, self.n_iter, self.tol, self.C, self.stick, self.seed = K, n_iter, tol, C, stickiness, seed

    def _init(self, X, y):
        rng = np.random.default_rng(self.seed); K, D = self.K, X.shape[1]
        base = LogisticRegression(C=self.C, max_iter=300, fit_intercept=False).fit(X, y).coef_[0]
        self.W = base[None, :] + rng.normal(0, 1.0, (K, D)) * (K > 1)
        self.A = np.full((K, K), (1 - self.stick) / max(K - 1, 1)) if K > 1 else np.ones((1, 1))
        np.fill_diagonal(self.A, self.stick if K > 1 else 1.0)
        self.pi = np.full(K, 1.0 / K)

    def fit(self, X, y, groups):
        """X: (N, D) incl. a constant column for bias; y: (N,) in {0,1}; groups: (N,) session ids."""
        idx = [np.where(groups == g)[0] for g in np.unique(groups)]
        self._init(X, y); K = self.K; prev = -np.inf; self.ll_trace = []
        for it in range(self.n_iter):
            # ---- E-step: posterior over states for every trial
            gam = np.zeros((len(y), K)); xi = np.zeros((K, K)); first = np.zeros(K); tot = 0.0
            for ix in idx:
                g, x, ll = _fb(emission_logp(X[ix], y[ix], self.W), self.A, self.pi)
                gam[ix] = g; xi += x; first += g[0]; tot += ll
            self.ll_trace.append(tot)
            if tot - prev < self.tol * len(y) and it > 5: break
            prev = tot
            # ---- M-step: closed form for pi, A; weighted logistic regression for each state's weights
            self.pi = first / first.sum()
            self.A = xi / xi.sum(1, keepdims=True)
            for k in range(K):
                Xr = np.vstack([X, X]); yr = np.r_[np.ones(len(y)), np.zeros(len(y))]
                wr = np.r_[gam[:, k] * y, gam[:, k] * (1 - y)]       # soft labels -> weighted samples
                m = LogisticRegression(C=self.C, max_iter=200, fit_intercept=False)
                m.fit(Xr, yr, sample_weight=wr + 1e-12); self.W[k] = m.coef_[0]
        self.train_ll = tot; self.gamma = gam; self.idx = idx
        return self

    def loglik(self, X, y, groups):
        """Total held-out log-likelihood (nats) with parameters frozen."""
        return sum(_viterbi_free_loglik(emission_logp(X[ix], y[ix], self.W), self.A, self.pi)
                   for ix in (np.where(groups == g)[0] for g in np.unique(groups)))

    def posteriors(self, X, y, groups):
        out = np.zeros((len(y), self.K))
        for g in np.unique(groups):
            ix = np.where(groups == g)[0]
            out[ix] = _fb(emission_logp(X[ix], y[ix], self.W), self.A, self.pi)[0]
        return out
