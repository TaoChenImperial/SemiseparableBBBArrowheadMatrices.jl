using BandedMatrices
using BlockArrays: Block, BlockedArray
using Test
using PiecewiseOrthogonalPolynomials, MatrixFactorizations
using SemiseparableBBBArrowheadMatrices: SemiseparableBBBArrowheadMatrix, copyBBBArrowheadMatrices, fast_ql


l = 10 # number of grid points
𝐗 = range(-1,1; length = l)
C = ContinuousPolynomial{1}(𝐗)
M = C'C
Δ = weaklaplacian(C)
N = 6 # will generate a N × N blocked the matrix
KR = Block.(Base.OneTo(N))
Mₙ = M[KR,KR]
Δₙ = Δ[KR,KR]
A = Δₙ + 100^2 * Mₙ
F = ql(A).factors
F = BlockedArray(F, axes(A))
τ_true = ql(A).τ
L, τ = fast_ql(A)
L = BlockedArray(L)

# test if τ equals τ_true except for the first 2 blocks
@test τ[l+l:end] ≈ τ_true[l+l:end]

# test if L equals F except for the upper left 3 × 2 blocks
@test L[Block.(4:N), :] ≈ F[Block.(4:N), :]
@test L[Block.(1:3), Block.(3:N)] ≈ F[Block.(1:3), Block.(3:N)]
