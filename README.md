# Emergent Geometry from Quantum Entanglement
### Numerical demonstration using Heisenberg spin chains

**Paul Jarvis** — Independent Researcher, UK
ORCID: [0009-0009-8933-857X](https://orcid.org/0009-0009-8933-857X)

---

## Overview

This repository provides a fully reproducible numerical demonstration that
continuous spatial geometry — specifically, the spectral properties of the
Laplace–Beltrami operator — can emerge directly from the entanglement
structure of a pure quantum state, without presupposing a background manifold
or metric.

The model system is the isotropic antiferromagnetic Heisenberg spin chain.
The ground-state mutual information matrix is computed by exact
diagonalisation, normalised to form a graph adjacency matrix, and its
graph Laplacian analysed spectrally. Three signatures of emergent
one-dimensional geometry are verified:

1. **Zero mode** — constant eigenvector with eigenvalue < 10⁻¹⁵
2. **Quadratic dispersion** — low-lying eigenvalues scale as λₖ ∝ k² (R² > 0.95)
3. **Finite-size scaling** — lowest non-zero eigenvalue scales as λ₁ ∝ 1/N² (R² = 0.952, N = 6–20)

Two additional results directly support the theoretical framework:

4. **Gapped phase comparison** — smooth emergent geometry collapses when the
   chain is dimerised, demonstrating that substrate criticality is a necessary
   condition for continuous spatial structure
5. **Power-law MI decay** — mutual information decays as I(i,j) ∝ |i−j|⁻²
   (η ≈ 2.06, consistent across N = 8, 12, 16), confirming the long-range
   entanglement structure that seeds the emergent connectivity

---

## Requirements

Standard scientific Python only — no specialist packages required:

```
numpy
scipy
matplotlib
```

Install with:

```bash
pip install numpy scipy matplotlib
```

---

## Usage

```bash
python numerical_v2.py
```

This generates six figures saved to the working directory:

| File | Content |
|------|---------|
| `spectrum.png` | Quadratic dispersion of low-lying Laplacian eigenvalues (N=12) |
| `scaling_lambda1.png` | Finite-size scaling λ₁ vs 1/N² for N = 6, 8, 10, 12, 14, 16, 18, 20 |
| `eigenmodes.png` | Low-lying eigenmodes with cosine reference curves (Neumann BCs) |
| `mi_heatmap.png` | MI matrix + sublattice-resolved MI vs distance |
| `gapped_comparison.png` | Geometry breakdown in gapped (dimerised) phase |
| `powerlaw_decay.png` | Power-law MI decay with exponent fit across system sizes |

Runtime is under 60 seconds on a standard laptop.

---

## Key Results

| N  | λ₁      | Time  |
|----|---------|-------|
| 6  | 0.37569 | 0.1s  |
| 8  | 0.30273 | 0.1s  |
| 10 | 0.25286 | 0.2s  |
| 12 | 0.21699 | 0.3s  |
| 14 | 0.19008 | 0.5s  |
| 16 | 0.16917 | 1.2s  |
| 18 | 0.15249 | 4.5s  |
| 20 | 0.13886 | 19.6s |

Finite-size scaling fit: λ₁ ∝ 1/N², R² = 0.952 across N = 6–20

Power-law MI decay exponent: η ≈ 2.06 (consistent across N = 8, 12, 16)

---

## Implementation Notes

The code uses sparse matrix methods (`scipy.sparse`) for Hamiltonian
construction, enabling system sizes up to N = 20 on a standard laptop.
The Hamiltonian supports an optional bond-alternation (dimerisation) parameter
δ, allowing direct comparison between the critical (δ=0) and gapped (δ>0)
phases.

Mutual information is computed from exact two-site reduced density matrices
via partial trace. Von Neumann entropy is computed from eigenvalues of the
reduced density matrix with a numerical floor of 10⁻¹⁵ to avoid log(0).

The gapped phase comparison uses δ = 0.0, 0.3, 0.8 to show the progressive
breakdown of smooth emergent geometry as the spectral gap opens.

---

## Context and Related Work

This code supports the following papers:

- P. Jarvis, *Emergent Geometry from Entanglement: A Numerical Demonstration
  using Heisenberg Spin Chains* (2026) — primary numerical paper
- P. Jarvis, *The Gravitational Baseline Principle: Structural Asymmetry and
  the Pre-Geometric Quantum Substrate* (2026) — theoretical framework
- P. Jarvis, *General-Relativistic Cyclic Cosmology from Holographic
  Saturation* (2026) — synthesis and applications

---

## Limitations

- System sizes are limited to N ≤ 20 by the exponential growth of the
  Hilbert space (dim = 2^N). Larger systems would require DMRG or tensor
  network methods.
- The demonstration is one-dimensional. Extension to 2D and 3D substrates
  requires higher-dimensional lattice models and is identified as future work.
- The emergent geometry is Euclidean. Recovery of Lorentzian signature and
  causal structure remains an open problem.

---

## Citation

If you use this code, please cite:

```bibtex
@misc{Jarvis2026code,
  author = {Jarvis, Paul},
  title  = {Emergent Geometry from Quantum Entanglement:
             Numerical Demonstration using Heisenberg Spin Chains},
  year   = {2026},
  url    = {https://github.com/ZaPpi3/emergent-geometry-heisenberg}
}
```

---

## Licence

MIT Licence — free to use, modify, and distribute with attribution.
