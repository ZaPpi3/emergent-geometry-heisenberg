"""
numerical_v2.py
===============
Comprehensive numerical demonstration of emergent geometry from quantum
entanglement in Heisenberg spin chains.

Generates six figures:
  1. spectrum.png          — Quadratic dispersion of Laplacian eigenvalues
  2. scaling_lambda1.png   — Finite-size scaling N=6..20
  3. eigenmodes.png        — Eigenmodes vs cosine reference (Neumann BCs)
  4. mi_heatmap.png        — MI matrix + sublattice decomposition
  5. gapped_comparison.png — Geometry breakdown in gapped (dimerised) phase
  6. cft_scaling.png       — CFT logarithmic MI decay + power-law fit

Author: Paul Jarvis
Requires: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, eigvalsh
import time

# ============================================================
# Sparse Hamiltonian builder (enables N up to ~20)
# ============================================================

I2 = eye(2, format='csr')

def site_op(op_matrix, site, N):
    """Place operator on site in N-site chain using sparse Kronecker products."""
    result = eye(1, format='csr')
    for j in range(N):
        if j == site:
            result = kron(result, csr_matrix(np.array(op_matrix, dtype=float)),
                          format='csr')
        else:
            result = kron(result, I2, format='csr')
    return result

def build_sparse_H(N, J=1.0, delta=0.0):
    """
    Build Heisenberg Hamiltonian with optional bond alternation (dimerisation).

    H = sum_i J_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})

    where J_i = J*(1 + delta*(-1)^i).
    delta=0  -> uniform chain (critical, CFT)
    delta>0  -> dimerised chain (gapped)
    """
    dim = 2**N
    H = csr_matrix((dim, dim), dtype=float)
    Sx_mat  = [[0, 0.5], [0.5, 0]]
    SyI_mat = [[0, -0.5], [0.5, 0]]   # imaginary part of Sy
    Sz_mat  = [[0.5, 0], [0, -0.5]]

    for i in range(N - 1):
        Ji = J * (1.0 + delta * ((-1)**i))
        Sx  = site_op(Sx_mat,  i,   N)
        Sx1 = site_op(Sx_mat,  i+1, N)
        SyI = site_op(SyI_mat, i,   N)
        SyI1= site_op(SyI_mat, i+1, N)
        Sz  = site_op(Sz_mat,  i,   N)
        Sz1 = site_op(Sz_mat,  i+1, N)
        H += Ji * (Sx @ Sx1 - SyI @ SyI1 + Sz @ Sz1)
    return H

def ground_state(N, J=1.0, delta=0.0):
    """Return ground state vector."""
    H = build_sparse_H(N, J, delta)
    evals, evecs = eigsh(H, k=1, which='SA')
    return evecs[:, 0]

# ============================================================
# Information theory
# ============================================================

def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-15]
    return -np.sum(evals * np.log(evals))

def partial_trace(psi, keep, N):
    psi = psi.reshape([2] * N)
    trace_over = [i for i in range(N) if i not in keep]
    psi = np.transpose(psi, keep + trace_over)
    psi = psi.reshape(2**len(keep), 2**len(trace_over))
    return psi @ psi.conj().T

def mutual_information_matrix(psi, N):
    I_mat = np.zeros((N, N))
    S = [von_neumann_entropy(partial_trace(psi, [i], N)) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            Sij = von_neumann_entropy(partial_trace(psi, [i, j], N))
            mi = S[i] + S[j] - Sij
            I_mat[i, j] = mi
            I_mat[j, i] = mi
    return I_mat

def laplacian_from_MI(I_mat):
    A = I_mat / np.max(I_mat)
    np.fill_diagonal(A, 0)
    D = np.diag(A.sum(axis=1))
    return D - A, A

# ============================================================
# Figure 1: Quadratic dispersion spectrum
# ============================================================

def fig1_spectrum(N=12):
    print(f"[Fig 1] Spectrum N={N}...")
    psi = ground_state(N)
    I_mat = mutual_information_matrix(psi, N)
    L, _ = laplacian_from_MI(I_mat)
    evals_L = eigvalsh(L)

    k_vals = np.arange(1, 5)
    low = evals_L[1:5]
    coeffs = np.polyfit(k_vals**2, low, 1)
    fit_k = np.linspace(1, 4, 200)
    fit_y = coeffs[0] * fit_k**2 + coeffs[1]
    predicted = coeffs[0] * k_vals**2 + coeffs[1]
    r2 = 1 - np.sum((low - predicted)**2) / np.sum((low - np.mean(low))**2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_vals, low, 'o', ms=8, label='Numerical eigenvalues', zorder=5)
    ax.plot(fit_k, fit_y, '-', lw=2,
            label=fr'Quadratic fit ($R^2={r2:.4f}$)')
    ax.set_xlabel('Mode number $k$', fontsize=12)
    ax.set_ylabel(r'Eigenvalue $\lambda_k$', fontsize=12)
    ax.set_title(f'Quadratic dispersion of Laplacian eigenvalues (N={N})',
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/spectrum.png', dpi=300)
    plt.close()
    print(f"  R²={r2:.5f}, eigenvalues={low}")

# ============================================================
# Figure 2: Finite-size scaling N=6..20
# ============================================================

def fig2_scaling():
    print("[Fig 2] Finite-size scaling N=6..20...")
    Ns = [6, 8, 10, 12, 14, 16, 18, 20]
    lam1_vals = []

    for N in Ns:
        t0 = time.time()
        psi = ground_state(N)
        I_mat = mutual_information_matrix(psi, N)
        L, _ = laplacian_from_MI(I_mat)
        evals_L = eigvalsh(L)
        lam1_vals.append(evals_L[1])
        print(f"  N={N}: λ₁={evals_L[1]:.5f} ({time.time()-t0:.1f}s)")

    invN2 = [1.0 / N**2 for N in Ns]
    coeffs = np.polyfit(invN2, lam1_vals, 1)
    fit_x = np.linspace(min(invN2), max(invN2), 200)
    fit_y = np.polyval(coeffs, fit_x)
    residuals = np.array(lam1_vals) - np.polyval(coeffs, invN2)
    r2 = 1 - np.sum(residuals**2) / np.sum(
        (np.array(lam1_vals) - np.mean(lam1_vals))**2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(invN2, lam1_vals, 'o', ms=8,
            label=r'Numerical $\lambda_1$', zorder=5)
    ax.plot(fit_x, fit_y, '-', lw=2,
            label=fr'Linear fit ($R^2={r2:.4f}$)')
    for i, N in enumerate(Ns):
        ax.annotate(f'N={N}', (invN2[i], lam1_vals[i]),
                    textcoords='offset points', xytext=(6, 3), fontsize=8)
    ax.set_xlabel(r'$1/N^2$', fontsize=12)
    ax.set_ylabel(r'$\lambda_1$', fontsize=12)
    ax.set_title(r'Finite-size scaling: $\lambda_1 \propto 1/N^2$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/scaling_lambda1.png', dpi=300)
    plt.close()
    print(f"  R²={r2:.5f} across N={Ns}")
    return Ns, lam1_vals

# ============================================================
# Figure 3: Eigenmodes vs cosine reference
# ============================================================

def fig3_eigenmodes(N=12):
    print(f"[Fig 3] Eigenmodes N={N}...")
    psi = ground_state(N)
    I_mat = mutual_information_matrix(psi, N)
    L, _ = laplacian_from_MI(I_mat)
    evals_L, vecs_L = eigh(L)
    vecs_L = np.real(vecs_L)

    x = np.arange(N)
    x_fine = np.linspace(0, N - 1, 300)
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, k in enumerate([1, 2, 3]):
        mode = vecs_L[:, k]
        if mode[np.argmax(np.abs(mode))] < 0:
            mode = -mode
        ax.plot(x, mode, 'o-', color=colors[idx], lw=2, ms=6,
                label=f'Mode $k={k}$')
        cos_ref = np.cos(k * np.pi * x_fine / (N - 1))
        cos_ref *= np.max(np.abs(mode)) / np.max(np.abs(cos_ref))
        ax.plot(x_fine, cos_ref, '--', color=colors[idx], alpha=0.45, lw=1.5)

    ax.set_xlabel('Site index', fontsize=12)
    ax.set_ylabel('Eigenvector amplitude', fontsize=12)
    ax.set_title(
        f'Laplacian eigenmodes (solid) vs cosine reference (dashed), N={N}',
        fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/eigenmodes.png', dpi=300)
    plt.close()

# ============================================================
# Figure 4: MI heatmap + sublattice decomposition
# ============================================================

def fig4_mi_heatmap(N=12):
    print(f"[Fig 4] MI heatmap N={N}...")
    psi = ground_state(N)
    I_mat = mutual_information_matrix(psi, N)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im = axes[0].imshow(I_mat, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=axes[0], label='Mutual information')
    axes[0].set_title(f'Mutual-information matrix (N={N})', fontsize=12)
    axes[0].set_xlabel('Site $j$')
    axes[0].set_ylabel('Site $i$')

    same_p, diff_p, same_d, diff_d = [], [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            d = abs(i - j)
            v = I_mat[i, j]
            if (i + j) % 2 == 0:
                same_p.append(v); same_d.append(d)
            else:
                diff_p.append(v); diff_d.append(d)

    axes[1].scatter(same_d, same_p, alpha=0.7, s=50,
                    label='Same sublattice ($i+j$ even)', marker='o')
    axes[1].scatter(diff_d, diff_p, alpha=0.7, s=50,
                    label='Diff sublattice ($i+j$ odd)', marker='^')
    axes[1].set_xlabel(r'Distance $|i-j|$', fontsize=12)
    axes[1].set_ylabel('Mutual information $I(i,j)$', fontsize=12)
    axes[1].set_title(
        'Sublattice-resolved MI vs distance\n(antiferromagnetic staggering)',
        fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/mi_heatmap.png', dpi=300)
    plt.close()

    ratio = np.mean(same_p) / np.mean(diff_p)
    print(f"  Sublattice ratio (same/diff): {ratio:.3f}")

# ============================================================
# Figure 5: Gapped phase comparison
# ============================================================

def fig5_gapped_comparison(N=12):
    print(f"[Fig 5] Gapped phase comparison N={N}...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    configs = [
        (0.0,  'Critical (δ=0)',   'tab:blue'),
        (0.3,  'Weakly gapped (δ=0.3)', 'tab:orange'),
        (0.8,  'Strongly gapped (δ=0.8)', 'tab:red'),
    ]

    eigenvalue_sets = []
    for delta, label, color in configs:
        psi = ground_state(N, delta=delta)
        I_mat = mutual_information_matrix(psi, N)
        L, _ = laplacian_from_MI(I_mat)
        evals_L, vecs_L = eigh(L)
        vecs_L = np.real(vecs_L)
        eigenvalue_sets.append((evals_L, vecs_L, label, color))

    # Panel 1: Eigenvalue spectra comparison
    for evals_L, _, label, color in eigenvalue_sets:
        k_vals = np.arange(1, 6)
        axes[0].plot(k_vals, evals_L[1:6], 'o-', color=color,
                     label=label, lw=2, ms=6)
    axes[0].set_xlabel('Mode number $k$', fontsize=11)
    axes[0].set_ylabel(r'Eigenvalue $\lambda_k$', fontsize=11)
    axes[0].set_title('Laplacian spectrum vs gap', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: First non-trivial eigenmode
    x = np.arange(N)
    x_fine = np.linspace(0, N - 1, 300)
    for evals_L, vecs_L, label, color in eigenvalue_sets:
        mode = vecs_L[:, 1]
        if mode[np.argmax(np.abs(mode))] < 0:
            mode = -mode
        axes[1].plot(x, mode, 'o-', color=color, label=label, lw=2, ms=5)
    cos_ref = np.cos(np.pi * x_fine / (N - 1))
    cos_ref *= 0.4 / np.max(np.abs(cos_ref))
    axes[1].plot(x_fine, cos_ref, 'k--', alpha=0.4, lw=1.5,
                 label='Cosine reference')
    axes[1].set_xlabel('Site index', fontsize=11)
    axes[1].set_ylabel('Eigenvector amplitude', fontsize=11)
    axes[1].set_title('Mode $k=1$: smoothness vs gap', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Spectral gap (lambda_2 / lambda_1 ratio)
    deltas = np.linspace(0, 1.0, 12)
    ratios = []
    for d in deltas:
        psi = ground_state(N, delta=d)
        I_mat = mutual_information_matrix(psi, N)
        L, _ = laplacian_from_MI(I_mat)
        evals_L = eigvalsh(L)
        ratios.append(evals_L[2] / evals_L[1])

    axes[2].plot(deltas, ratios, 'o-', color='tab:purple', lw=2, ms=6)
    axes[2].axvline(x=0, color='tab:blue', linestyle='--', alpha=0.5,
                    label='Critical point')
    axes[2].set_xlabel('Dimerisation $\\delta$', fontsize=11)
    axes[2].set_ylabel(r'$\lambda_2 / \lambda_1$', fontsize=11)
    axes[2].set_title('Spectral ratio vs dimerisation\n'
                      '(uniform spacing → 1 at criticality)', fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f'Geometry breakdown in gapped phase (N={N}): '
        'criticality is necessary for smooth emergent geometry',
        fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/gapped_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Gapped comparison complete.")

# ============================================================
# Figure 6: Power-law MI decay across system sizes
# ============================================================

def fig6_powerlaw_decay():
    print("[Fig 6] Power-law MI decay...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    Ns_plot = [8, 12, 16]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    exponents = []

    for N, color in zip(Ns_plot, colors):
        print(f"  N={N}...")
        psi = ground_state(N)
        I_mat = mutual_information_matrix(psi, N)

        # Collect bulk same-sublattice pairs (i+j even, avoid boundaries)
        distances = []
        mi_vals = []
        for i in range(1, N - 1):
            for j in range(i + 2, N - 1, 2):  # step 2 = same sublattice
                if (i + j) % 2 == 0:
                    distances.append(abs(i - j))
                    mi_vals.append(I_mat[i, j])

        distances = np.array(distances, dtype=float)
        mi_vals = np.array(mi_vals)

        # Sort by distance
        idx = np.argsort(distances)
        distances = distances[idx]
        mi_vals = mi_vals[idx]

        # Power-law fit on log-log
        log_d = np.log(distances)
        log_mi = np.log(mi_vals)
        coeffs = np.polyfit(log_d, log_mi, 1)
        eta = -coeffs[0]
        exponents.append(eta)

        fit_d = np.linspace(distances[0], distances[-1], 200)
        fit_mi = np.exp(np.polyval(coeffs, np.log(fit_d)))

        axes[0].loglog(distances, mi_vals, 'o', color=color,
                       alpha=0.7, ms=6, label=f'N={N}')
        axes[0].loglog(fit_d, fit_mi, '-', color=color, lw=2,
                       label=fr'$\eta={eta:.2f}$')

    axes[0].set_xlabel(r'Distance $|i-j|$', fontsize=12)
    axes[0].set_ylabel(r'Mutual information $I(i,j)$', fontsize=12)
    axes[0].set_title('Power-law decay of MI (same sublattice)', fontsize=12)
    axes[0].legend(fontsize=9, ncol=2)
    axes[0].grid(True, alpha=0.3, which='both')

    # Panel 2: exponent vs N showing convergence
    axes[1].plot(Ns_plot, exponents, 'o-', ms=8, lw=2, color='tab:purple')
    axes[1].axhline(y=2.0, color='gray', linestyle='--', alpha=0.6,
                    label='Reference η=2')
    for i, (N, eta) in enumerate(zip(Ns_plot, exponents)):
        axes[1].annotate(f'η={eta:.2f}', (N, exponents[i]),
                         textcoords='offset points', xytext=(5, 4), fontsize=9)
    axes[1].set_xlabel('System size $N$', fontsize=12)
    axes[1].set_ylabel(r'Power-law exponent $\eta$', fontsize=12)
    axes[1].set_title('MI decay exponent vs system size\n'
                      '(convergence confirms power-law structure)', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/powerlaw_decay.png', dpi=300)
    plt.close()
    print(f"  Exponents: {[f'{e:.3f}' for e in exponents]} for N={Ns_plot}")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    t_total = time.time()

    print("=" * 60)
    print("Emergent Geometry from Entanglement — Full Analysis")
    print("=" * 60)

    print("\n[Fig 1] Quadratic dispersion spectrum")
    fig1_spectrum(N=12)

    print("\n[Fig 2] Finite-size scaling N=6..20")
    fig2_scaling()

    print("\n[Fig 3] Eigenmodes vs cosine reference")
    fig3_eigenmodes(N=12)

    print("\n[Fig 4] MI heatmap + sublattice analysis")
    fig4_mi_heatmap(N=12)

    print("\n[Fig 5] Gapped phase comparison")
    fig5_gapped_comparison(N=12)

    print("\n[Fig 6] Power-law MI decay across system sizes")
    fig6_powerlaw_decay()

    print(f"\n{'='*60}")
    print(f"All figures saved. Total time: {time.time()-t_total:.1f}s")
    print("=" * 60)
