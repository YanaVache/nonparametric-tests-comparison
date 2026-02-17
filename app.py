# app.py
# Streamlit app: exact vs chi-square approximation for 4 nonparametric tests
# Pearson chi^2 (GOF), Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis
#
# Run:
#   pip install streamlit numpy pandas scipy
#   streamlit run app.py

from __future__ import annotations

import io
import math
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats


# ----------------------------
# Helpers: parsing + perturbation
# ----------------------------

def parse_number_list(text: str) -> np.ndarray:
    """
    Parse a list of numbers from text.
    Accepts separators: comma, semicolon, whitespace, newlines.
    """
    if text is None:
        return np.array([], dtype=float)
    text = text.strip()
    if not text:
        return np.array([], dtype=float)

    # Normalize separators to spaces
    for ch in [",", ";", "\t"]:
        text = text.replace(ch, " ")
    parts = [p for p in text.split() if p]
    try:
        arr = np.array([float(p) for p in parts], dtype=float)
    except ValueError as e:
        raise ValueError("Nie udało się sparsować listy liczb. Upewnij się, że wpisujesz tylko liczby.") from e
    return arr


def add_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return x.copy()
    return x + rng.normal(loc=0.0, scale=sigma, size=x.shape)


def round_values(x: np.ndarray, decimals: int) -> np.ndarray:
    return np.round(x, decimals=decimals)


def shuffle_fraction(x: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly shuffle a fraction of elements (in-place selection) to imitate data disturbance.
    """
    x2 = x.copy()
    n = len(x2)
    if n <= 1 or frac <= 0:
        return x2
    m = int(round(frac * n))
    m = min(max(m, 0), n)
    if m <= 1:
        return x2
    idx = rng.choice(n, size=m, replace=False)
    shuffled = x2[idx].copy()
    rng.shuffle(shuffled)
    x2[idx] = shuffled
    return x2


# ----------------------------
# Core: Chi-square approximations
# ----------------------------

def p_from_chi2(stat: float, df: int) -> float:
    stat = float(stat)
    df = int(df)
    if stat < 0 or df <= 0:
        return float("nan")
    return float(stats.chi2.sf(stat, df))


def z_to_chi2_pvalue(z: float) -> float:
    # If Z ~ N(0,1) then Z^2 ~ chi2_1 approximately
    return p_from_chi2(z * z, df=1)


# ----------------------------
# Pearson chi-square GOF
# ----------------------------

@dataclass
class PearsonResult:
    chi2_stat: float
    df: int
    p_approx_chi2: float
    p_mc_exact: float
    observed: np.ndarray
    expected: np.ndarray


def pearson_gof(observed: np.ndarray, expected_probs: np.ndarray, mc_B: int, seed: int) -> PearsonResult:
    """
    observed: counts in k categories
    expected_probs: probabilities under H0 (length k, sum to 1)
    Exact-like: Monte Carlo p-value under multinomial(n, p0)
    Approx: chi-square(df=k-1) if conditions ok
    """
    obs = np.array(observed, dtype=int)
    if obs.ndim != 1 or len(obs) < 2:
        raise ValueError("Pearson: potrzebujesz co najmniej 2 kategorii (wektor obserwacji).")
    if np.any(obs < 0):
        raise ValueError("Pearson: obserwacje muszą być nieujemne (liczebności).")
    n = int(obs.sum())
    p0 = np.array(expected_probs, dtype=float)
    if p0.shape != obs.shape:
        raise ValueError("Pearson: wektor prawdopodobieństw musi mieć ten sam rozmiar co obserwacje.")
    if np.any(p0 < 0):
        raise ValueError("Pearson: prawdopodobieństwa muszą być nieujemne.")
    s = float(p0.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Pearson: suma prawdopodobieństw musi być dodatnia.")
    p0 = p0 / s

    exp = n * p0
    # Pearson chi2 stat
    # handle exp=0:
    if np.any(exp == 0):
        raise ValueError("Pearson: nie można liczyć χ² gdy oczekiwana liczebność E_i=0.")
    chi2_stat = float(np.sum((obs - exp) ** 2 / exp))
    df = len(obs) - 1
    p_approx = p_from_chi2(chi2_stat, df=df)

    rng = np.random.default_rng(seed)
    # MC exact-like
    sims = rng.multinomial(n=n, pvals=p0, size=int(mc_B))
    chi2_sims = np.sum((sims - exp) ** 2 / exp, axis=1)
    # p-value with >=
    p_mc = float((np.sum(chi2_sims >= chi2_stat) + 1) / (len(chi2_sims) + 1))

    return PearsonResult(
        chi2_stat=chi2_stat,
        df=df,
        p_approx_chi2=p_approx,
        p_mc_exact=p_mc,
        observed=obs,
        expected=exp,
    )


# ----------------------------
# Mann–Whitney U: exact + approx (chi-square via Z^2)
# ----------------------------

@dataclass
class MWResult:
    U: float
    z: float
    p_exact: float
    p_approx_chi2: float
    method_exact: str


def mann_whitney_exact_and_approx(x: np.ndarray, y: np.ndarray, alternative: str) -> MWResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 1 or len(y) < 1:
        raise ValueError("Mann–Whitney: obie próby muszą mieć co najmniej 1 obserwację.")

    # Approx z (normal, with tie correction if needed)
    # We'll compute U via ranks manually to also compute z.
    combined = np.concatenate([x, y])
    ranks = stats.rankdata(combined, method="average")
    r1 = np.sum(ranks[: len(x)])
    n1 = len(x)
    n2 = len(y)
    U1 = r1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    # Use U consistent with alternative for z center
    # For two-sided we use min(U1, U2) for z magnitude typically.
    if alternative == "less":
        U = U1
    elif alternative == "greater":
        U = U2  # greater means x tends to be larger -> U2? safer use scipy for p; for z we use U1 center anyway.
        # We'll still compute z from U1 vs mu; use sign by (U1 - mu).
    else:
        U = min(U1, U2)

    mu = n1 * n2 / 2.0

    # Tie correction for variance:
    # var(U) = n1*n2/12 * (N+1 - sum(t^3 - t)/(N*(N-1)))
    N = n1 + n2
    _, counts = np.unique(combined, return_counts=True)
    tie_term = np.sum(counts**3 - counts)
    if N > 1:
        sigma2 = (n1 * n2 / 12.0) * (N + 1 - tie_term / (N * (N - 1)))
    else:
        sigma2 = 0.0
    sigma = math.sqrt(sigma2) if sigma2 > 0 else float("nan")

    # Continuity correction (optional): use 0.5
    # We'll apply it only if sigma is finite.
    cc = 0.5 if np.isfinite(sigma) else 0.0
    z = (U1 - mu - np.sign(U1 - mu) * cc) / sigma if np.isfinite(sigma) and sigma > 0 else float("nan")

    # Exact p-value:
    # Try scipy exact method (available in newer scipy). Fallback to permutation.
    method_exact = "scipy-exact"
    p_exact = float("nan")
    try:
        res = stats.mannwhitneyu(x, y, alternative=alternative, method="exact")
        # SciPy returns U statistic based on ranks; we use res.statistic for U.
        U_stat = float(res.statistic)
        p_exact = float(res.pvalue)
        U_for_report = U_stat
    except TypeError:
        # older SciPy: no 'method' argument
        try:
            res = stats.mannwhitneyu(x, y, alternative=alternative)
            U_for_report = float(res.statistic)
            p_exact = float(res.pvalue)  # asymptotic in older scipy
            method_exact = "no-exact-in-scipy"
        except Exception:
            U_for_report = float(U1)

    # If we didn't get exact, do permutation exact-ish for small sizes; otherwise Monte Carlo.
    if not np.isfinite(p_exact) or method_exact in ("no-exact-in-scipy",):
        method_exact = "permutation"
        p_exact = permutation_pvalue_mw(x, y, alternative=alternative, B=20000, seed=12345)
        U_for_report = float(U1)

    # Approx via chi-square from Z^2
    p_approx_chi2 = z_to_chi2_pvalue(float(z)) if np.isfinite(z) else float("nan")

    return MWResult(U=U_for_report, z=float(z), p_exact=p_exact, p_approx_chi2=p_approx_chi2, method_exact=method_exact)


def permutation_pvalue_mw(x: np.ndarray, y: np.ndarray, alternative: str, B: int, seed: int) -> float:
    """
    Permutation p-value for Mann-Whitney based on U1 computed from ranks.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n1 = len(x)
    n2 = len(y)
    N = n1 + n2
    data = np.concatenate([x, y])
    ranks = stats.rankdata(data, method="average")
    r1_obs = np.sum(ranks[:n1])
    U1_obs = r1_obs - n1 * (n1 + 1) / 2.0

    rng = np.random.default_rng(seed)
    # Monte Carlo permutation
    count = 0
    for _ in range(int(B)):
        perm = rng.permutation(N)
        idx_x = perm[:n1]
        U1 = np.sum(ranks[idx_x]) - n1 * (n1 + 1) / 2.0
        if alternative == "two-sided":
            # two-sided uses distance from mean
            mu = n1 * n2 / 2.0
            if abs(U1 - mu) >= abs(U1_obs - mu):
                count += 1
        elif alternative == "less":
            if U1 <= U1_obs:
                count += 1
        else:  # greater
            if U1 >= U1_obs:
                count += 1
    return float((count + 1) / (B + 1))


# ----------------------------
# Wilcoxon signed-rank: exact + approx (chi-square via Z^2)
# ----------------------------

@dataclass
class WilcoxonResult:
    W_plus: float
    z: float
    p_exact: float
    p_approx_chi2: float
    method_exact: str
    n_used: int


def wilcoxon_exact_and_approx(x: np.ndarray, y: np.ndarray, alternative: str) -> WilcoxonResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Wilcoxon: potrzebujesz dwóch list tej samej długości (>=2).")

    d = x - y
    # Remove zeros (as standard)
    mask = d != 0
    d2 = d[mask]
    if len(d2) < 1:
        raise ValueError("Wilcoxon: wszystkie różnice są 0, test nie ma sensu.")
    n = len(d2)

    abs_d = np.abs(d2)
    ranks = stats.rankdata(abs_d, method="average")
    W_plus = float(np.sum(ranks[d2 > 0]))
    W_minus = float(np.sum(ranks[d2 < 0]))

    # Normal approx for W_plus with tie correction:
    # E(W+) = n(n+1)/4
    mu = n * (n + 1) / 4.0
    # Var(W+) without ties: n(n+1)(2n+1)/24
    # With ties in abs differences: var = (1/24)*(n(n+1)(2n+1) - sum(t^3 - t)/2)
    _, counts = np.unique(abs_d, return_counts=True)
    tie_term = np.sum(counts**3 - counts)
    sigma2 = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    sigma = math.sqrt(sigma2) if sigma2 > 0 else float("nan")

    cc = 0.5 if np.isfinite(sigma) else 0.0
    # choose statistic direction
    if alternative == "two-sided":
        # use W_plus (and compare to mu)
        z = (W_plus - mu - np.sign(W_plus - mu) * cc) / sigma if np.isfinite(sigma) and sigma > 0 else float("nan")
    elif alternative == "greater":
        # x > y -> positive differences -> large W_plus
        z = (W_plus - mu - cc) / sigma if np.isfinite(sigma) and sigma > 0 else float("nan")
    else:  # less
        # x < y -> negative differences -> small W_plus
        z = (W_plus - mu + cc) / sigma if np.isfinite(sigma) and sigma > 0 else float("nan")

    # Exact p-value: try scipy exact if available; else DP exact distribution of W+
    method_exact = "scipy-exact"
    p_exact = float("nan")
    try:
        res = stats.wilcoxon(x, y, alternative=alternative, method="exact", zero_method="wilcox")
        p_exact = float(res.pvalue)
    except TypeError:
        # old SciPy: no method=
        method_exact = "dp-exact"
        p_exact = wilcoxon_dp_exact_pvalue(d2, alternative=alternative)

    # Approx via chi-square from Z^2
    p_approx_chi2 = z_to_chi2_pvalue(float(z)) if np.isfinite(z) else float("nan")

    return WilcoxonResult(
        W_plus=W_plus,
        z=float(z),
        p_exact=float(p_exact),
        p_approx_chi2=float(p_approx_chi2),
        method_exact=method_exact,
        n_used=n
    )


def wilcoxon_dp_exact_pvalue(d_nonzero: np.ndarray, alternative: str) -> float:
    """
    Exact p-value for Wilcoxon signed-rank using DP over integer ranks.
    Works best when ranks are integers (no ties). With ties, ranks are fractional -> DP is harder.
    We'll handle ties by scaling ranks to integers (LCM of denominators), which is fine for small n.
    """
    d = np.asarray(d_nonzero, float)
    abs_d = np.abs(d)
    ranks = stats.rankdata(abs_d, method="average")  # may be fractional

    # Convert ranks to rational grid: multiply by 2 (covers .5) or by 12 to be safe
    # We'll determine multiplier from decimals.
    # Robust: use denominator up to 12 by rounding to nearest 1/12.
    mult = 12
    r_int = np.rint(ranks * mult).astype(int)

    W_obs = int(np.sum(r_int[d > 0]))
    total = int(np.sum(r_int))
    # Under H0, each rank gets + or - with prob 1/2 (independent),
    # so distribution of W+ is distribution of subset-sum of ranks.
    # DP counts ways to achieve each sum.
    dp = np.zeros(total + 1, dtype=np.int64)
    dp[0] = 1
    for r in r_int:
        dp[r: ] += dp[: total + 1 - r]

    # total outcomes: 2^n
    denom = 2 ** len(r_int)

    # compute p-value
    # W+ ranges 0..total
    probs = dp / denom

    # two-sided: based on distance from mean
    mu = total / 2.0
    if alternative == "two-sided":
        dist_obs = abs(W_obs - mu)
        p = float(np.sum(probs[np.abs(np.arange(total + 1) - mu) >= dist_obs]))
    elif alternative == "greater":
        p = float(np.sum(probs[np.arange(total + 1) >= W_obs]))
    else:  # less
        p = float(np.sum(probs[np.arange(total + 1) <= W_obs]))

    # small-sample adjustment
    return min(max(p, 0.0), 1.0)


# ----------------------------
# Kruskal–Wallis: permutation exact + chi-square approx
# ----------------------------

@dataclass
class KWResult:
    H: float
    df: int
    p_exact_perm: float
    p_approx_chi2: float


def kruskal_exact_and_approx(groups: List[np.ndarray], B: int, seed: int) -> KWResult:
    groups = [np.asarray(g, float) for g in groups]
    if any(len(g) < 1 for g in groups) or len(groups) < 2:
        raise ValueError("Kruskal–Wallis: potrzebujesz co najmniej 2 grup, każda >=1 obserwacja.")

    # observed H
    H_obs = float(stats.kruskal(*groups).statistic)
    k = len(groups)
    df = k - 1
    p_approx = p_from_chi2(H_obs, df=df)

    # permutation exact-ish: permute labels on pooled data
    rng = np.random.default_rng(seed)
    pooled = np.concatenate(groups)
    n = len(pooled)
    sizes = [len(g) for g in groups]

    # precompute ranks once (permutation of labels only)
    ranks = stats.rankdata(pooled, method="average")

    def H_from_partition(ranks_perm: np.ndarray) -> float:
        # Compute H from ranks and group sizes
        start = 0
        Rbars = []
        for sz in sizes:
            Rbars.append(np.mean(ranks_perm[start:start + sz]))
            start += sz
        # H = (12/(N(N+1))) * sum n_i (Rbar_i - (N+1)/2)^2
        N = n
        grand = (N + 1) / 2.0
        return float((12.0 / (N * (N + 1))) * np.sum([sizes[i] * (Rbars[i] - grand) ** 2 for i in range(k)]))

    Hvals = np.empty(int(B), dtype=float)
    idx = np.arange(n)
    for b in range(int(B)):
        rng.shuffle(idx)
        ranks_perm = ranks[idx]
        Hvals[b] = H_from_partition(ranks_perm)

    p_exact = float((np.sum(Hvals >= H_obs) + 1) / (len(Hvals) + 1))

    return KWResult(H=H_obs, df=df, p_exact_perm=p_exact, p_approx_chi2=p_approx)


# ----------------------------
# Streamlit UI
# ----------------------------

def main():
    st.set_page_config(page_title="Testy nieparametryczne: exact vs χ²", layout="wide")
    st.title("Część programistyczna: wynik dokładny vs aproksymacja χ²")

    st.markdown(
    """
    Ta aplikacja liczy **p-value dokładne** (exact/permutacje/Monte Carlo)
    oraz **p-value aproksymowane**.

    Dla testów MW i Wilcoxona korzystamy z faktu:
    """
    )

    st.latex(r"Z \approx \mathcal{N}(0,1) \Rightarrow Z^2 \approx \chi^2_1")


    rng_seed = st.sidebar.number_input(
        "Seed (losowość)",
        min_value=0,
        max_value=99_999_999,
        value=20260216,
        step=1
    )
    rng = np.random.default_rng(int(rng_seed))

    test = st.sidebar.selectbox(
        "Wybierz test",
        ["Pearson χ² (zgodności)", "Mann–Whitney U", "Wilcoxon signed-rank", "Kruskal–Wallis"],
    )

    st.sidebar.subheader("Zaburzenie danych (opcjonalne)")
    perturb = st.sidebar.selectbox("Rodzaj zaburzenia", ["brak", "szum addytywny", "zaokrąglenie", "shuffle części danych"])
    sigma = st.sidebar.number_input("σ (dla szumu)", min_value=0.0, value=0.0, step=0.1)
    decimals = st.sidebar.number_input("Liczba miejsc po przecinku (zaokr.)", min_value=0, max_value=6, value=0, step=1)
    frac = st.sidebar.slider("Ułamek elementów do shuffle", 0.0, 1.0, 0.0, 0.05)

    st.sidebar.subheader("Ustawienia 'dokładności'")
    B_perm = st.sidebar.number_input("Liczba permutacji (KW / MW fallback)", min_value=1000, max_value=200000, value=20000, step=1000)
    B_mc = st.sidebar.number_input("Monte Carlo (Pearson)", min_value=1000, max_value=500000, value=50000, step=5000)

    alternative = st.sidebar.selectbox("Alternatywa", ["two-sided", "less", "greater"])

    # Input
    st.header("Dane wejściowe")

    if test == "Pearson χ² (zgodności)":
        col1, col2 = st.columns(2)
        with col1:
            obs_txt = st.text_area("Obserwacje (liczebności w k kategoriach)", "12, 8, 15, 5")
        with col2:
            p_txt = st.text_area("Prawdopodobieństwa H0 (k liczb, suma=1)", "0.25, 0.25, 0.25, 0.25")

        if st.button("Policz"):
            try:
                obs = parse_number_list(obs_txt).astype(int)
                p0 = parse_number_list(p_txt).astype(float)

                # perturbation for counts doesn't make sense (keep as is)
                res = pearson_gof(obs, p0, mc_B=int(B_mc), seed=int(rng_seed))

                st.subheader("Wyniki")
                st.write(f"Statystyka χ² = **{res.chi2_stat:.6g}**, df = **{res.df}**")
                st.write(f"p-value aproksymacja χ²(df) = **{res.p_approx_chi2:.6g}**")
                st.write(f"p-value Monte Carlo (multinomial, 'dokładne' symulacyjnie) = **{res.p_mc_exact:.6g}**")
                st.write(f"|p_exact - p_approx| = **{abs(res.p_mc_exact - res.p_approx_chi2):.6g}**")

                table = pd.DataFrame({"Observed": res.observed, "Expected (n·p0)": res.expected})
                st.dataframe(table, use_container_width=True)

                # Warning about small expected
                if np.any(res.expected < 5):
                    st.warning("Uwaga: są kategorie z E_i < 5 — aproksymacja χ² może być słaba.")

            except Exception as e:
                st.error(str(e))

    elif test == "Mann–Whitney U":
        col1, col2 = st.columns(2)
        with col1:
            x_txt = st.text_area("Próba X (niezależna)", "1.2, 2.1, 2.0, 3.3, 2.7, 1.9")
        with col2:
            y_txt = st.text_area("Próba Y (niezależna)", "0.9, 1.0, 1.8, 2.2, 1.4, 1.6")

        if st.button("Policz"):
            try:
                x = parse_number_list(x_txt)
                y = parse_number_list(y_txt)

                # perturb
                if perturb == "szum addytywny":
                    x = add_noise(x, sigma=float(sigma), rng=rng)
                    y = add_noise(y, sigma=float(sigma), rng=rng)
                elif perturb == "zaokrąglenie":
                    x = round_values(x, int(decimals))
                    y = round_values(y, int(decimals))
                elif perturb == "shuffle części danych":
                    x = shuffle_fraction(x, float(frac), rng=rng)
                    y = shuffle_fraction(y, float(frac), rng=rng)

                res = mann_whitney_exact_and_approx(x, y, alternative=alternative)

                st.subheader("Wyniki")
                st.write(f"U = **{res.U:.6g}**")
                st.write(f"Z (aproks. normalna) = **{res.z:.6g}**")
                st.write(f"p-value exact (metoda: {res.method_exact}) = **{res.p_exact:.6g}**")
                st.write(f"p-value aproksymacja χ² (Z² ~ χ²₁) = **{res.p_approx_chi2:.6g}**")
                st.write(f"|p_exact - p_approx| = **{abs(res.p_exact - res.p_approx_chi2):.6g}**")

                if np.any(stats.rankdata(np.concatenate([x, y]), method="average") != stats.rankdata(np.concatenate([x, y]), method="ordinal")):
                    st.info("W danych występują remisy (ties) — aproksymacja może się pogarszać.")

            except Exception as e:
                st.error(str(e))

    elif test == "Wilcoxon signed-rank":
        col1, col2 = st.columns(2)
        with col1:
            x_txt = st.text_area("Pomiar X (sparowany)", "10, 12, 9, 11, 8, 10, 13")
        with col2:
            y_txt = st.text_area("Pomiar Y (sparowany)", "9, 12, 10, 10, 8, 9, 12")

        if st.button("Policz"):
            try:
                x = parse_number_list(x_txt)
                y = parse_number_list(y_txt)

                # perturb
                if perturb == "szum addytywny":
                    x = add_noise(x, sigma=float(sigma), rng=rng)
                    y = add_noise(y, sigma=float(sigma), rng=rng)
                elif perturb == "zaokrąglenie":
                    x = round_values(x, int(decimals))
                    y = round_values(y, int(decimals))
                elif perturb == "shuffle części danych":
                    # for paired test, shuffle the pair order together
                    idx = np.arange(len(x))
                    x = shuffle_fraction(x, float(frac), rng=rng)
                    y = shuffle_fraction(y, float(frac), rng=rng)

                res = wilcoxon_exact_and_approx(x, y, alternative=alternative)

                st.subheader("Wyniki")
                st.write(f"W⁺ = **{res.W_plus:.6g}** (po odrzuceniu zer: n = {res.n_used})")
                st.write(f"Z (aproks. normalna) = **{res.z:.6g}**")
                st.write(f"p-value exact (metoda: {res.method_exact}) = **{res.p_exact:.6g}**")
                st.write(f"p-value aproksymacja χ² (Z² ~ χ²₁) = **{res.p_approx_chi2:.6g}**")
                st.write(f"|p_exact - p_approx| = **{abs(res.p_exact - res.p_approx_chi2):.6g}**")

            except Exception as e:
                st.error(str(e))

    else:  # Kruskal–Wallis
        st.markdown("Wklej dane jako kilka grup (każda grupa w osobnym polu).")
        k = st.number_input("Liczba grup", min_value=2, max_value=8, value=3, step=1)

        cols = st.columns(int(k))
        group_texts = []
        for i in range(int(k)):
            with cols[i]:
                group_texts.append(st.text_area(f"Grupa {i+1}", "1, 2, 3, 4" if i == 0 else "2, 3, 4, 5"))

        if st.button("Policz"):
            try:
                groups = [parse_number_list(t) for t in group_texts]

                # perturb (applied per group)
                if perturb == "szum addytywny":
                    groups = [add_noise(g, sigma=float(sigma), rng=rng) for g in groups]
                elif perturb == "zaokrąglenie":
                    groups = [round_values(g, int(decimals)) for g in groups]
                elif perturb == "shuffle części danych":
                    groups = [shuffle_fraction(g, float(frac), rng=rng) for g in groups]

                res = kruskal_exact_and_approx(groups, B=int(B_perm), seed=int(rng_seed))

                st.subheader("Wyniki")
                st.write(f"H = **{res.H:.6g}**, df = **{res.df}**")
                st.write(f"p-value exact (permutacje) = **{res.p_exact_perm:.6g}**")
                st.write(f"p-value aproksymacja χ²(df) = **{res.p_approx_chi2:.6g}**")
                st.write(f"|p_exact - p_approx| = **{abs(res.p_exact_perm - res.p_approx_chi2):.6g}**")

            except Exception as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
