import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.linalg import eigh, eigvalsh
from scipy.sparse import kron, eye, diags
from scipy.sparse.linalg import eigsh

# ============================================================
# Spin operators (sparse numpy)
# ============================================================

sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def op_on_site(op, i, N):
    """Place 2x2 operator op at site i in N-site chain (dense)."""
    result = np.array([[1.0+0j]])
    for j in range(N):
        result = np.kron(result, op if j == i else I2)
    return result

def build_hamiltonian(N, J=1.0):
    """Build Heisenberg Hamiltonian as dense matrix."""
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N - 1):
        for op in [sx, sy, sz]:
            H += J * (op_on_site(op, i, N) @ op_on_site(op, i+1, N))
    return H

def ground_state(N, J=1.0):
    """Return ground state vector."""
    H = build_hamiltonian(N, J)
    evals, evecs = eigh(H)
    return evecs[:, 0]

def von_neumann_entropy(rho):
    """Entropy of density matrix."""
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-15]
    return -np.sum(evals * np.log(evals))

def partial_trace(psi, keep, N):
    """Partial trace keeping sites in list 'keep'."""
    dim = 2**N
    psi = psi.reshape([2]*N)
    trace_over = [i for i in range(N) if i not in keep]
    # Move traced-over indices to the end
    order = keep + trace_over
    psi = np.transpose(psi, order)
    keep_dim = 2**len(keep)
    trace_dim = 2**len(trace_over)
    psi = psi.reshape(keep_dim, trace_dim)
    rho = psi @ psi.conj().T
    return rho

def mutual_information_matrix(psi, N):
    """Compute NxN mutual information matrix."""
    I_mat = np.zeros((N, N))
    # Cache single-site entropies
    S_single = []
    for i in range(N):
        rho_i = partial_trace(psi, [i], N)
        S_single.append(von_neumann_entropy(rho_i))

    for i in range(N):
        for j in range(i+1, N):
            rho_ij = partial_trace(psi, [i, j], N)
            Sij = von_neumann_entropy(rho_ij)
            mi = S_single[i] + S_single[j] - Sij
            I_mat[i, j] = mi
            I_mat[j, i] = mi
    return I_mat

def laplacian_from_MI(I_mat):
    """Normalized adjacency and graph Laplacian."""
    A = I_mat / np.max(I_mat)
    np.fill_diagonal(A, 0)
    D = np.diag(A.sum(axis=1))
    L = D - A
    return L, A

# ============================================================
# 1. Spectrum plot
# ============================================================

def generate_spectrum_plot(N=12):
    print(f"  Computing spectrum for N={N}...")
    psi = ground_state(N)
    I_mat = mutual_information_matrix(psi, N)
    L, _ = laplacian_from_MI(I_mat)
    evals_L = eigvalsh(L)

    k_vals = np.arange(1, 5)
    low = evals_L[1:5]

    coeffs = np.polyfit(k_vals**2, low, 1)
    fit_k = np.linspace(1, 4, 200)
    fit_lambda = coeffs[0] * fit_k**2 + coeffs[1]
    r2 = 1 - np.sum((low - np.polyval([coeffs[0], 0, coeffs[1]], k_vals))**2) / np.sum((low - np.mean(low))**2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_vals, low, 'o', label='Numerical eigenvalues', zorder=5)
    ax.plot(fit_k, fit_lambda, '-', label=f'Quadratic fit (R²={r2:.4f})')
    ax.set_xlabel("Mode number k")
    ax.set_ylabel(r"Eigenvalue $\lambda_k$")
    ax.set_title(f"Quadratic scaling of low-lying Laplacian eigenvalues (N={N})")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/spectrum.png", dpi=300)
    plt.close()
    print(f"  R² = {r2:.5f}, eigenvalues: {low}")

# ============================================================
# 2. Finite-size scaling — extended to N=6,8,10,12,14,16
# ============================================================

def generate_scaling_plot():
    Ns = [6, 8, 10, 12, 14, 16]
    lam1 = []
    for N in Ns:
        print(f"  Computing N={N}...")
        psi = ground_state(N)
        I_mat = mutual_information_matrix(psi, N)
        L, _ = laplacian_from_MI(I_mat)
        evals_L = eigvalsh(L)
        lam1.append(evals_L[1])

    invN2 = [1.0 / N**2 for N in Ns]
    coeffs = np.polyfit(invN2, lam1, 1)
    fit_x = np.linspace(min(invN2), max(invN2), 200)
    fit_y = np.polyval(coeffs, fit_x)

    residuals = np.array(lam1) - np.polyval(coeffs, invN2)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.array(lam1) - np.mean(lam1))**2)
    r2 = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(invN2, lam1, 'o', label=r'Numerical $\lambda_1$', zorder=5)
    ax.plot(fit_x, fit_y, '-', label=f'Linear fit (R²={r2:.4f})')
    for i, N in enumerate(Ns):
        ax.annotate(f'N={N}', (invN2[i], lam1[i]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel(r'$1/N^2$')
    ax.set_ylabel(r'$\lambda_1$')
    ax.set_title(r"Finite-size scaling of lowest non-zero eigenvalue")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/scaling_lambda1.png", dpi=300)
    plt.close()
    print(f"  R² = {r2:.5f}")
    print(f"  N values: {Ns}")
    print(f"  λ1 values: {[f'{v:.5f}' for v in lam1]}")
    return Ns, lam1

# ============================================================
# 3. Eigenmode plot
# ============================================================

def generate_eigenmode_plot(N=12):
    print(f"  Computing eigenmodes for N={N}...")
    psi = ground_state(N)
    I_mat = mutual_information_matrix(psi, N)
    L, _ = laplacian_from_MI(I_mat)
    evals_L, vecs_L = eigh(L)
    vecs_L = np.real(vecs_L)

    x = np.arange(N)
    # Cosine reference functions (Neumann BCs)
    x_fine = np.linspace(0, N-1, 300)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for idx, k in enumerate([1, 2, 3]):
        mode = vecs_L[:, k]
        # Normalise sign so first peak is positive
        if mode[np.argmax(np.abs(mode))] < 0:
            mode = -mode
        ax.plot(x, mode, 'o-', color=colors[idx], label=f"Mode k={k}")
        # Cosine reference
        cos_ref = np.cos(k * np.pi * x_fine / (N - 1))
        cos_ref *= np.max(np.abs(mode)) / np.max(np.abs(cos_ref))
        ax.plot(x_fine, cos_ref, '--', color=colors[idx], alpha=0.4)

    ax.set_xlabel("Site index")
    ax.set_ylabel("Eigenvector amplitude")
    ax.set_title(f"Low-lying Laplacian eigenmodes with cosine reference (N={N})")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/eigenmodes.png", dpi=300)
    plt.close()

# ============================================================
# 4. MI heatmap with checkerboard analysis
# ============================================================

def generate_mi_heatmap(N=12):
    print(f"  Computing MI matrix for N={N}...")
    psi = ground_state(N)
    I_mat = mutual_information_matrix(psi, N)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Raw MI matrix
    im = axes[0].imshow(I_mat, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=axes[0], label="Mutual information")
    axes[0].set_title(f"Raw mutual-information matrix (N={N})")
    axes[0].set_xlabel("Site j")
    axes[0].set_ylabel("Site i")

    # Checkerboard analysis: separate even-even/odd-odd vs even-odd pairs
    # Antiferromagnetic staggering means I(i,j) depends on parity of i+j
    same_parity = []
    diff_parity = []
    distances = []
    same_dist = []
    diff_dist = []

    for i in range(N):
        for j in range(i+1, N):
            d = abs(i - j)
            val = I_mat[i, j]
            if (i + j) % 2 == 0:
                same_parity.append(val)
                same_dist.append(d)
            else:
                diff_parity.append(val)
                diff_dist.append(d)

    axes[1].scatter(same_dist, same_parity, alpha=0.7, label='Same parity (i+j even)', marker='o')
    axes[1].scatter(diff_dist, diff_parity, alpha=0.7, label='Diff parity (i+j odd)', marker='^')
    axes[1].set_xlabel("Distance |i-j|")
    axes[1].set_ylabel("Mutual information I(i,j)")
    axes[1].set_title("Parity-resolved MI vs distance\n(checkerboard decomposition)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/mi_heatmap.png", dpi=300)
    plt.close()

    # Print parity ratio
    mean_same = np.mean(same_parity)
    mean_diff = np.mean(diff_parity)
    print(f"  Mean MI same parity: {mean_same:.4f}")
    print(f"  Mean MI diff parity: {mean_diff:.4f}")
    print(f"  Ratio (same/diff): {mean_same/mean_diff:.3f}")
    print("  (Checkerboard arises from antiferromagnetic sublattice structure)")

# ============================================================
# Run all
# ============================================================

if __name__ == "__main__":
    print("=== 1. Spectrum plot ===")
    generate_spectrum_plot(N=12)

    print("\n=== 2. Finite-size scaling (N=6,8,10,12,14,16) ===")
    generate_scaling_plot()

    print("\n=== 3. Eigenmode plot ===")
    generate_eigenmode_plot(N=12)

    print("\n=== 4. MI heatmap + checkerboard analysis ===")
    generate_mi_heatmap(N=12)

    print("\n=== All figures saved to outputs/ ===")
