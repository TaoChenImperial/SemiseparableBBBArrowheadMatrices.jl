using BandedMatrices
using BlockArrays: Block, BlockedArray
using Test, LinearAlgebra
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
L, τ = fast_ql(A) # orthogonalise columns Block.(3:N)

Q̃ = BlockedArray(ql(Matrix(A[:,27:end])).Q * I, axes(A)) # Householder applied to columns Block(3:N)
@test Q̃'Q̃ ≈ I
LL = Q̃'A

#@test (Q̃'A)[:,Block.(1:2)] ≈ L[:,Block.(1:2)]




#Q̃ = BlockedArray(ql(Matrix(A[:,Block.(4:N)])).Q * I, axes(A)) # Householder applied to columns Block(4:N)

#Q̄ = BlockedArray(ql(Matrix(A[:,axes(A,2)[Block(3)[l-2]]:size(A,2)])).Q * I, axes(A)) # Householder applied to columns Block(3)[end]:end

#(Q̃'A)[:,Block.(1:4)]
#(Q̄'A)[:,Block.(1:4)]


#Q = BlockedArray(Matrix(ql(A).Q), axes(A))


# test if τ equals τ_true except for the first 2 blocks
@test τ[l+l:end] ≈ τ_true[l+l:end]

# test if L equals F except for the upper left 3 × 2 blocks
@test L[Block.(4:N), :] ≈ F[Block.(4:N), :]
@test L[Block.(1:3), Block.(3:N)] ≈ F[Block.(1:3), Block.(3:N)]

@test L[Block(1,1)] ≈ LL[Block(1,1)]
@test L[Block(3,1)] ≈ LL[Block(3,1)]

