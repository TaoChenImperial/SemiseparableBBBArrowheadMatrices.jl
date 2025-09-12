using BandedMatrices
using Test, LinearAlgebra
using SemiseparableBBBArrowheadMatrices: BandedPlusSemiseparableMatrix, SemiseparableBBBArrowheadMatrix, copyBBBArrowheadMatrices, fast_ql, fast_solver, BandedPlusSemiseparableQRPerturbedFactors


n = 10
A = BandedPlusSemiseparableQRPerturbedFactors(randn(n), randn(n), randn(n), randn(n), randn(n),randn(), randn())
(; d, u, v, W, S, κ, q) = A; w = W[:,1]; s = S[:,1]

B = BandedPlusSemiseparableMatrix(d, (u,v), (w,s))

@test_broken A == B + u*(q*s' + κ*u'B) # TODO: implement getindex for BandedPlusSemiseparableQRPerturbedFactors

function onestep_qr!(A)
    # TODO: implement
end

y = A[:,1]; y[1] += sign(y[1])*norm(y); y = y/norm(y); Q = I-2y*y'

QA = Q*A
onestep_qr!(A)
@test A.j[] == 1
@test A == QA



