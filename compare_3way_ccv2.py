"""
compare_3way_ccv2.py — CDF exact vs Nina's Sinkhorn vs EquiPy's Sinkhorn
=========================================================================
Three methods, same data, pure NumPy. No JAX, no sklearn, no POT, no pandas.

Method 1: CDF repair (exact closed form via quantile averaging)
Method 2: Nina's Sinkhorn (barycenter + CDF matching for transport maps)
Method 3: EquiPy's Sinkhorn (barycenter + barycentric projection + isotonic regression)

The core algorithmic difference between methods 2 and 3 is how the transport
map T_g is recovered from the Sinkhorn output:
  - Nina: T_g = F_bary^{-1} ∘ F_g  (CDF inversion on histograms)
  - EquiPy: T_g[i] = sum_j pi_{ij} * y_j / mu_g[i], then isotonic regression
            where pi = diag(u) K diag(v) is the transport plan

Run on RECOD.AI:
    cd ~/ot_faces && python compare_3way_ccv2.py

Reads:  output/comparison/scores_1d.npz  (from compare_cdf_sinkhorn_ccv2.py)
Writes: output/comparison/compare_3way.png
        output/comparison/results_3way.csv
"""

import numpy as np
import os, time, csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = os.path.expanduser("~/ot_faces/output/comparison")
SEED = 42
np.random.seed(SEED)


# =====================================================================
# SHARED: histogram on grid
# =====================================================================

def make_grid(scores, n_bins=200, pad=1e-6):
    """Same as EquiPy's make_grid."""
    lo, hi = scores.min(), scores.max()
    if hi <= lo:
        hi = lo + 1.0
    lo -= pad * (hi - lo)
    hi += pad * (hi - lo)
    return np.linspace(lo, hi, n_bins)


def hist_on_grid(scores, grid):
    """Same as EquiPy's hist_on_grid: nearest-bin histogram."""
    edges = (grid[:-1] + grid[1:]) / 2.0
    idx = np.digitize(scores, edges)
    counts = np.bincount(idx, minlength=len(grid)).astype(float)
    total = counts.sum()
    if total <= 0:
        return np.ones(len(grid)) / len(grid)
    return counts / total


# =====================================================================
# SHARED: Sinkhorn barycenter (log-domain, same algorithm as EquiPy)
# =====================================================================

def sinkhorn_barycenter_log(measures, C, weights, reg, maxiter=10000, tol=1e-8):
    """
    Log-domain Sinkhorn barycenter. Same algorithm as EquiPy's
    barycenter_sinkhorn but in pure NumPy.

    Args:
        measures: (G, m) array of histograms
        C: (m, m) cost matrix (normalized)
        weights: (G,) group sizes (will be normalized to sum=1)
        reg: entropic regularization

    Returns:
        bary: (m,) barycenter histogram
        lnus: (G, m) log scaling factors u
        lnvs: (G, m) log scaling factors v
        info: dict with diagnostics
    """
    G, m = measures.shape
    lam = (weights / weights.sum())[:, None]  # (G, 1)
    log_measures = np.log(np.maximum(measures, 1e-30))
    lnK = -C / reg  # (m, m)

    lnus = np.zeros((G, m))
    lnvs = np.zeros((G, m))
    lnb = np.zeros(m)

    def logsumexp(X, axis):
        mx = X.max(axis=axis, keepdims=True)
        return np.squeeze(mx, axis=axis) + np.log(
            np.sum(np.exp(X - mx), axis=axis))

    for it in range(maxiter):
        lnb_old = lnb.copy()

        # u_g = a_g / (K v_g)  in log domain
        # ln_Kv[g, i] = logsumexp_j(lnvs[g,j] + lnK[i,j])
        ln_Kv = np.array([logsumexp(lnvs[g, :][None, :] + lnK, axis=1) for g in range(G)])
        lnus = log_measures - ln_Kv

        # ln_Ktu[g, j] = logsumexp_i(lnus[g,i] + lnK[i,j])
        ln_Ktu = np.array([logsumexp(lnus[g, :, None] + lnK, axis=0) for g in range(G)])

        # barycenter = geometric mean: log(b) = sum_g lambda_g * ln_Ktu[g]
        lnb = np.sum(lam * ln_Ktu, axis=0)
        lnb = lnb - logsumexp(lnb, axis=0)  # normalize

        # v_g = b / (K^T u_g)  in log domain
        lnvs = lnb[None, :] - ln_Ktu

        if np.max(np.abs(lnb - lnb_old)) < tol:
            break

    bary = np.exp(lnb)
    bary /= bary.sum()

    return bary, lnus, lnvs, {'iterations': it + 1}


# =====================================================================
# METHOD 1: CDF repair (exact)
# =====================================================================

def cdf_repair(scores_dict, pi, country_order):
    """Exact barycenter via quantile averaging."""
    t_grid = np.linspace(0.001, 0.999, 2000)
    bary_q = np.zeros_like(t_grid)

    cdfs = {}
    for i, c in enumerate(country_order):
        sv = np.sort(scores_dict[c])
        cv = np.arange(1, len(sv) + 1) / len(sv)
        cdfs[c] = (sv, cv)
        bary_q += pi[i] * np.interp(t_grid, cv, sv)

    repaired = {}
    for c in country_order:
        sv, cv = cdfs[c]
        pct = np.interp(scores_dict[c], sv, cv, left=0.0, right=1.0)
        repaired[c] = np.interp(pct, t_grid, bary_q)

    return repaired


# =====================================================================
# METHOD 2: Nina's Sinkhorn (barycenter + CDF matching)
# =====================================================================

def nina_sinkhorn_repair(scores_dict, pi, country_order, n_bins=200, reg=1e-3):
    """Sinkhorn barycenter, transport via CDF matching on histograms."""
    all_s = np.concatenate([scores_dict[c] for c in country_order])
    grid = make_grid(all_s, n_bins)

    hists = np.array([hist_on_grid(scores_dict[c], grid) for c in country_order])
    C = (grid[:, None] - grid[None, :]) ** 2
    C /= C.max()

    bary, _, _, info = sinkhorn_barycenter_log(hists, C, pi, reg)
    print(f"    Nina barycenter: {info['iterations']} iters")

    # CDF matching: T_g = F_bary^{-1} ∘ F_g
    bary_cdf = np.cumsum(bary)
    bary_cdf = np.clip(bary_cdf, 0, 1)
    for i in range(1, len(bary_cdf)):
        if bary_cdf[i] <= bary_cdf[i - 1]:
            bary_cdf[i] = bary_cdf[i - 1] + 1e-12

    repaired = {}
    for gi, c in enumerate(country_order):
        gcdf = np.cumsum(hists[gi])
        pct = np.interp(grid, grid, gcdf)
        T = np.interp(pct, bary_cdf, grid)
        s = np.clip(scores_dict[c], grid[0], grid[-1])
        repaired[c] = np.interp(s, grid, T)

    return repaired, grid, bary


# =====================================================================
# METHOD 3: EquiPy's Sinkhorn (barycenter + barycentric projection)
# =====================================================================

def isotonic_regression(x, y):
    """
    Pool Adjacent Violators Algorithm (PAVA).
    Same result as sklearn.isotonic.IsotonicRegression(increasing=True).
    Returns y_hat such that y_hat is non-decreasing and minimizes ||y - y_hat||^2.
    """
    n = len(y)
    y_hat = y.copy().astype(float)
    w = np.ones(n, dtype=float)

    # Forward pass: merge violating blocks
    i = 0
    blocks = []  # (start, end, value, weight)
    while i < n:
        blocks.append([i, i, y_hat[i], 1.0])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            # Merge last two blocks
            b1 = blocks.pop()
            b0 = blocks[-1]
            total_w = b0[3] + b1[3]
            b0[1] = b1[1]
            b0[2] = (b0[2] * b0[3] + b1[2] * b1[3]) / total_w
            b0[3] = total_w
        i += 1

    # Write back
    result = np.empty(n, dtype=float)
    for start, end, val, _ in blocks:
        result[start:end + 1] = val

    # Clip to input range (like sklearn out_of_bounds="clip")
    return np.clip(result, y.min(), y.max())


def barycentric_projection(ln_u, ln_v, lnK, mu, grid, eps_mass=1e-16):
    """
    Same as EquiPy's barycentric_projection_from_log_scalings_stable.
    T[i] = sum_j pi[i,j] * y[j] / mu[i]
    where pi = diag(exp(ln_u)) * exp(lnK) * diag(exp(ln_v))
    """
    m = len(grid)
    log_pi = ln_u[:, None] + lnK + ln_v[None, :]  # (m, m)

    # Numerator: sum_j exp(log_pi_ij) * y_j
    # Handle positive and negative y separately for numerical stability
    y = grid.copy()
    num = np.zeros(m)

    pos = y > 0
    neg = y < 0

    def stable_logsumexp(X, axis):
        mx = X.max(axis=axis, keepdims=True)
        return np.squeeze(mx, axis=axis) + np.log(np.sum(np.exp(X - mx), axis=axis))

    if np.any(pos):
        log_y_pos = np.log(y[pos])
        num += np.exp(stable_logsumexp(log_pi[:, pos] + log_y_pos[None, :], axis=1))

    if np.any(neg):
        log_y_neg = np.log(-y[neg])
        num -= np.exp(stable_logsumexp(log_pi[:, neg] + log_y_neg[None, :], axis=1))

    T = np.empty_like(num)
    good = mu > eps_mass
    T[good] = num[good] / mu[good]
    T[~good] = y[~good]
    return T


def equipy_sinkhorn_repair(scores_dict, pi, country_order, n_bins=200, reg=1e-3):
    """
    EquiPy pipeline: Sinkhorn barycenter + barycentric projection + isotonic.
    Reimplemented in pure NumPy (no JAX, no sklearn, no POT, no pandas).
    """
    all_s = np.concatenate([scores_dict[c] for c in country_order])
    grid = make_grid(all_s, n_bins)

    hists = np.array([hist_on_grid(scores_dict[c], grid) for c in country_order])
    C = (grid[:, None] - grid[None, :]) ** 2
    C /= C.max()

    bary, lnus, lnvs, info = sinkhorn_barycenter_log(hists, C, pi, reg)
    lnK = -C / reg
    print(f"    EquiPy barycenter: {info['iterations']} iters")

    # Barycentric projection + isotonic regression for each group
    repaired = {}
    for gi, c in enumerate(country_order):
        T_raw = barycentric_projection(lnus[gi], lnvs[gi], lnK, hists[gi], grid)
        T_mono = isotonic_regression(grid, T_raw)

        s = np.clip(scores_dict[c], grid[0], grid[-1])
        repaired[c] = np.interp(s, grid, T_mono)

    return repaired, grid, bary


# =====================================================================
# METRICS
# =====================================================================

def dp_gap(d, co):
    return max(d[c].mean() for c in co) - min(d[c].mean() for c in co)

def ks_2samp(x, y):
    combined = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(np.sort(x), combined, side='right') / len(x)
    cdf_y = np.searchsorted(np.sort(y), combined, side='right') / len(y)
    return np.max(np.abs(cdf_x - cdf_y))

def dp_ks(d, co):
    mx = 0
    for i, c1 in enumerate(co):
        for c2 in co[i + 1:]:
            mx = max(mx, ks_2samp(d[c1], d[c2]))
    return mx


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # Load precomputed 1D scores
    scores_path = os.path.join(OUTPUT_DIR, 'scores_1d.npz')
    if not os.path.exists(scores_path):
        print(f"ERROR: {scores_path} not found.")
        print("Run compare_cdf_sinkhorn_ccv2.py first to generate 1D scores.")
        exit(1)

    data = np.load(scores_path, allow_pickle=True)
    country_order = list(data['country_order'])
    pi = data['pi'].astype(float)
    scores_by_group = {c: data[c] for c in country_order}

    print("Loaded 1D scores:")
    for c in country_order:
        s = scores_by_group[c]
        print(f"  {c:12s}: n={len(s):3d}  mean={s.mean():.4f}  std={s.std():.4f}")

    # Original metrics
    orig_dp = dp_gap(scores_by_group, country_order)
    orig_ks = dp_ks(scores_by_group, country_order)
    print(f"\nOriginal: DP(means)={orig_dp:.4f}  DP(KS)={orig_ks:.4f}")

    # Run all three methods at multiple regularization levels
    reg_values = [1e-2, 5e-3, 1e-3, 5e-4]
    results = {}

    for reg in reg_values:
        print(f"\n=== eps = {reg:.0e} ===")

        # Method 1: CDF (exact, doesn't depend on reg)
        if reg == reg_values[0]:  # only compute once
            print("  CDF (exact):")
            rep_cdf = cdf_repair(scores_by_group, pi, country_order)
            results['cdf'] = rep_cdf
            cdf_dp = dp_gap(rep_cdf, country_order)
            cdf_ks = dp_ks(rep_cdf, country_order)
            print(f"    DP(means)={cdf_dp:.4f}  DP(KS)={cdf_ks:.4f}")

        # Method 2: Nina's Sinkhorn
        print("  Nina's Sinkhorn:")
        rep_nina, grid_n, bary_n = nina_sinkhorn_repair(
            scores_by_group, pi, country_order, n_bins=200, reg=reg)
        results[('nina', reg)] = rep_nina

        # Method 3: EquiPy's Sinkhorn
        print("  EquiPy's Sinkhorn:")
        rep_equipy, grid_e, bary_e = equipy_sinkhorn_repair(
            scores_by_group, pi, country_order, n_bins=200, reg=reg)
        results[('equipy', reg)] = rep_equipy

    # =====================================================================
    # COMPARISON TABLE
    # =====================================================================
    print("\n" + "=" * 85)
    print(f"{'Method':<20s} {'eps':>8s} {'DP(mean)':>10s} {'DP(KS)':>10s} "
          f"{'vs CDF Δ':>10s} {'vs CDF r':>10s} {'N vs E Δ':>10s} {'N vs E r':>10s}")
    print("=" * 85)

    # Original
    print(f"{'Original':<20s} {'---':>8s} {orig_dp:>10.4f} {orig_ks:>10.4f}")

    # CDF
    rep_cdf = results['cdf']
    cdf_all = np.concatenate([rep_cdf[c] for c in country_order])
    print(f"{'CDF (exact)':<20s} {'---':>8s} {dp_gap(rep_cdf, country_order):>10.4f} "
          f"{dp_ks(rep_cdf, country_order):>10.4f}")

    # Per regularization level
    csv_rows = []
    csv_rows.append(['original', '', f'{orig_dp:.6f}', f'{orig_ks:.6f}', '', '', '', ''])
    csv_rows.append(['cdf', '', f'{dp_gap(rep_cdf, country_order):.6f}',
                     f'{dp_ks(rep_cdf, country_order):.6f}', '', '1.000000', '', ''])

    for reg in reg_values:
        rep_n = results[('nina', reg)]
        rep_e = results[('equipy', reg)]

        nina_all = np.concatenate([rep_n[c] for c in country_order])
        equipy_all = np.concatenate([rep_e[c] for c in country_order])

        # vs CDF
        n_vs_cdf_delta = max(abs(rep_cdf[c].mean() - rep_n[c].mean()) for c in country_order)
        n_vs_cdf_r = np.corrcoef(cdf_all, nina_all)[0, 1]

        e_vs_cdf_delta = max(abs(rep_cdf[c].mean() - rep_e[c].mean()) for c in country_order)
        e_vs_cdf_r = np.corrcoef(cdf_all, equipy_all)[0, 1]

        # Nina vs EquiPy
        n_vs_e_delta = max(abs(rep_n[c].mean() - rep_e[c].mean()) for c in country_order)
        n_vs_e_r = np.corrcoef(nina_all, equipy_all)[0, 1]

        print(f"{'Nina Sinkhorn':<20s} {reg:>8.0e} {dp_gap(rep_n, country_order):>10.4f} "
              f"{dp_ks(rep_n, country_order):>10.4f} {n_vs_cdf_delta:>10.4f} "
              f"{n_vs_cdf_r:>10.4f} {'':>10s} {'':>10s}")
        print(f"{'EquiPy Sinkhorn':<20s} {reg:>8.0e} {dp_gap(rep_e, country_order):>10.4f} "
              f"{dp_ks(rep_e, country_order):>10.4f} {e_vs_cdf_delta:>10.4f} "
              f"{e_vs_cdf_r:>10.4f} {n_vs_e_delta:>10.4f} {n_vs_e_r:>10.4f}")

        csv_rows.append(['nina_sinkhorn', f'{reg:.0e}',
            f'{dp_gap(rep_n, country_order):.6f}', f'{dp_ks(rep_n, country_order):.6f}',
            f'{n_vs_cdf_delta:.6f}', f'{n_vs_cdf_r:.6f}', '', ''])
        csv_rows.append(['equipy_sinkhorn', f'{reg:.0e}',
            f'{dp_gap(rep_e, country_order):.6f}', f'{dp_ks(rep_e, country_order):.6f}',
            f'{e_vs_cdf_delta:.6f}', f'{e_vs_cdf_r:.6f}',
            f'{n_vs_e_delta:.6f}', f'{n_vs_e_r:.6f}'])

    print("=" * 85)

    # Per-group detail at best reg
    best_reg = 1e-3
    print(f"\nPer-group means at eps={best_reg:.0e}:")
    print(f"  {'Country':<12s} {'Original':>10s} {'CDF':>10s} {'Nina':>10s} {'EquiPy':>10s}")
    for c in country_order:
        print(f"  {c:<12s} {scores_by_group[c].mean():>10.4f} {rep_cdf[c].mean():>10.4f} "
              f"{results[('nina', best_reg)][c].mean():>10.4f} "
              f"{results[('equipy', best_reg)][c].mean():>10.4f}")

    # =====================================================================
    # FIGURE
    # =====================================================================
    rep_nina_best = results[('nina', best_reg)]
    rep_equipy_best = results[('equipy', best_reg)]

    colors = {}
    cc = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i, c in enumerate(country_order):
        colors[c] = cc[i % 4]

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, hspace=0.35, wspace=0.3)

    # Row 1: distributions (original, CDF, Nina, EquiPy)
    for col, (title, data) in enumerate([
        ('Original', scores_by_group),
        ('CDF (exact)', rep_cdf),
        (f'Nina Sinkhorn (ε={best_reg:.0e})', rep_nina_best),
        (f'EquiPy Sinkhorn (ε={best_reg:.0e})', rep_equipy_best),
    ]):
        ax = fig.add_subplot(gs[0, col])
        for c in country_order:
            ax.hist(data[c], bins=30, density=True, alpha=0.4, color=colors[c], label=c)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.legend(fontsize=6)

    # Row 2: scatter plots (CDF vs Nina, CDF vs EquiPy, Nina vs EquiPy, convergence)
    for col, (title, xdata, ydata, xl, yl) in enumerate([
        ('CDF vs Nina', rep_cdf, rep_nina_best, 'CDF', 'Nina'),
        ('CDF vs EquiPy', rep_cdf, rep_equipy_best, 'CDF', 'EquiPy'),
        ('Nina vs EquiPy', rep_nina_best, rep_equipy_best, 'Nina', 'EquiPy'),
    ]):
        ax = fig.add_subplot(gs[1, col])
        for c in country_order:
            ax.scatter(xdata[c], ydata[c], s=10, alpha=0.5, color=colors[c], label=c)
        lims = ax.get_xlim()
        ax.plot(lims, lims, 'k--', alpha=0.3, lw=1)
        all_x = np.concatenate([xdata[c] for c in country_order])
        all_y = np.concatenate([ydata[c] for c in country_order])
        r = np.corrcoef(all_x, all_y)[0, 1]
        ax.set_title(f'{title} (r={r:.4f})', fontweight='bold', fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=6)

    # Convergence: MAE to CDF for both methods
    ax_conv = fig.add_subplot(gs[1, 3])
    x_pos = np.arange(len(reg_values))
    width = 0.35
    nina_maes, equipy_maes = [], []
    for reg in reg_values:
        rn = results[('nina', reg)]
        re = results[('equipy', reg)]
        nina_all = np.concatenate([rn[c] for c in country_order])
        equipy_all = np.concatenate([re[c] for c in country_order])
        nina_maes.append(np.mean(np.abs(cdf_all - nina_all)))
        equipy_maes.append(np.mean(np.abs(cdf_all - equipy_all)))

    ax_conv.bar(x_pos - width/2, nina_maes, width, label='Nina', color='steelblue', alpha=0.7)
    ax_conv.bar(x_pos + width/2, equipy_maes, width, label='EquiPy', color='coral', alpha=0.7)
    ax_conv.set_xticks(x_pos)
    ax_conv.set_xticklabels([f'ε={r:.0e}' for r in reg_values], fontsize=8)
    ax_conv.set_title('MAE vs CDF (exact)', fontweight='bold', fontsize=10)
    ax_conv.legend(fontsize=8)

    # Row 3: tradeoff curves and per-group detail
    ax_trade = fig.add_subplot(gs[2, 0:2])
    alphas = np.linspace(0, 1, 21)
    for method, rep, fmt, label in [
        ('CDF', rep_cdf, 'k-o', 'CDF (exact)'),
        ('Nina', rep_nina_best, 'b--s', 'Nina Sinkhorn'),
        ('EquiPy', rep_equipy_best, 'r-.^', 'EquiPy Sinkhorn'),
    ]:
        ks_vals = []
        for a in alphas:
            mixed = {c: (1-a)*rep[c] + a*scores_by_group[c] for c in country_order}
            ks_vals.append(dp_ks(mixed, country_order))
        ax_trade.plot(alphas, ks_vals, fmt, ms=4, label=label)
    ax_trade.set_title('Fairness–Accuracy Tradeoff (Random Repair)', fontweight='bold', fontsize=10)
    ax_trade.set_xlabel('α (0=fair, 1=original)')
    ax_trade.set_ylabel('DP gap (KS)')
    ax_trade.legend(); ax_trade.grid(True, alpha=0.3)

    # Per-group means bar chart
    ax_bar = fig.add_subplot(gs[2, 2:4])
    x = np.arange(len(country_order))
    w = 0.2
    for i, (label, data) in enumerate([
        ('Original', scores_by_group),
        ('CDF', rep_cdf),
        ('Nina', rep_nina_best),
        ('EquiPy', rep_equipy_best),
    ]):
        means = [data[c].mean() for c in country_order]
        ax_bar.bar(x + i*w, means, w, label=label, alpha=0.8)
    ax_bar.set_xticks(x + 1.5*w)
    ax_bar.set_xticklabels(country_order, fontsize=9)
    ax_bar.set_title('Per-group mean scores', fontweight='bold', fontsize=10)
    ax_bar.legend(fontsize=8)
    ax_bar.set_ylabel('Score (W to own prototype)')

    fig.suptitle('Three-Way Comparison: CDF exact vs Nina Sinkhorn vs EquiPy Sinkhorn — CCv2',
                 fontsize=13, fontweight='bold', y=0.98)

    fig_path = os.path.join(OUTPUT_DIR, 'compare_3way.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure: {fig_path}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'results_3way.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'epsilon', 'dp_mean', 'dp_ks',
                     'vs_cdf_max_delta', 'vs_cdf_r', 'nina_vs_equipy_delta', 'nina_vs_equipy_r'])
        for row in csv_rows:
            w.writerow(row)
    print(f"Results: {csv_path}")
    print(f"\nTotal: {time.time()-t0:.1f}s")
