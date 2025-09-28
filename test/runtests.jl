using BandedMatrices
using Test, LinearAlgebra
using SemiseparableBBBArrowheadMatrices: BandedPlusSemiseparableMatrix, SemiseparableBBBArrowheadMatrix, copyBBBArrowheadMatrices, fast_ql, fast_solver, BandedPlusSemiseparableQRPerturbedFactors
using Random
#Random.seed!(1234)

n = 10
A = BandedPlusSemiseparableQRPerturbedFactors(randn(n), randn(n), randn(n), randn(n), randn(n),randn(), randn())
(; d, u, v, W, S, κ, q) = A; w = W[:,1]; s = S[:,1]

B = BandedPlusSemiseparableMatrix(d, (u[2:end],v[1:end-1]), (w[1:end-1],s[2:end]))

@test A[:,:] ≈ B + u*(q[]*s' + κ[]*u'B)

F = qr(A[:,:]).factors
Acopy = copy(A[:,:])
m, n = size(Acopy)
τ_true = Vector{eltype(Acopy)}(undef, min(m,n))
LAPACK.geqrf!(Acopy, τ_true)

function onestep_qr!(A, τ, u_square, ϵ)
    #compute A.W[:,2]
    if A.j[] == 0
        # need to modify to make it O(n)
        A.S[:,2] = (A.u'*B)'
    end

    current = A.j[] + 1
    u_square[] = u_square[] - A.u[current]^2
    #modify to access it in O(1)
    pivot = A[current,current]
    #also need to modify to make it O(1)
    u_pB = (A.u[current:end]'B[current:end,current:end])[1]
    u_coeff = A.v[current] + A.q[] * A.S[current,1] + A.κ[] * u_pB
    pivot_current = -sign(pivot) * sqrt(u_coeff^2 * u_square[] + pivot^2)
    A.d[current] = pivot_current

    A.v[current] = u_coeff / (pivot - pivot_current)
    τ[current] = 2 * (pivot-pivot_current)^2 / ((pivot-pivot_current)^2 + u_coeff^2 * u_square[])

    # modify to make it O(1)
    u_pu = A.u[current+1:end]'A.u[current+1:end]
    γ = -τ[current] * A.q[] * A.u[current] - τ[current] * A.q[] * A.v[current] * u_pu
    ϕ = -τ[current] * A.q[] * A.v[current] * A.u[current] - τ[current] * A.q[] * A.v[current]^2 * u_pu
    ξ = -τ[current] * A.κ[] * A.u[current] - τ[current] * A.κ[] * A.v[current] * u_pu
    η = -τ[current] * A.κ[] * A.v[current] * A.u[current] - τ[current] * A.κ[] * A.v[current]^2 * u_pu

    q_new = A.q[] + A.κ[] * A.u[current] * A.W[current,1] - 
            A.v[current] * τ[current] * A.W[current,1] + 
            ϕ + η * A.u[current] * A.W[current,1]
    κ_new = A.κ[] - A.v[current]^2 * τ[current] + η

    
    α_new = A.W[current,1] + A.q[] * A.u[current] + A.κ[] * A.u[current]^2 * A.W[current,1] - 
            τ[current] * A.W[current,1] + γ - A.κ[] * A.u[current]^2 * A.W[current,1] + 
            τ[current] * A.v[current] * A.W[current,1] * A.u[current] +
            (-A.κ[] * A.u[current] + τ[current] * A.v[current] - ξ) * ϵ[]
    β_new = A.κ[] * A.u[current] - τ[current] * A.v[current] + ξ
    ϵ[] = ϵ[] + A.u[current] * A.W[current,1]

    A.q[] = q_new
    A.κ[] = κ_new

    A.W[current,1] = α_new
    A.W[current,2] = β_new

    A.j[] = A.j[] + 1
end

function fast_qr!(A)
    n = length(A.d)
    τ = zeros(n)
    if A.j[] != 0
        throw(Exception("Matrix has already been partially upper-triangularized"))
    end

    u_square = Ref(0.0)
    for i in 1 : n
        u_square[] = u_square[] + A.u[i]^2
    end

    ϵ = Ref(0.0)

    for i in 1 : n - 1
        onestep_qr!(A, τ, u_square, ϵ)
    end

    A.d[n] = A[n,n]
    A.j[] = A.j[] + 1

    τ
end

y = A[:,1]; y[1] += sign(y[1])*norm(y); y = y/norm(y); Q = I-2y*y'

QA = Q*A[:,:]
τ = fast_qr!(A)

@test A.j[] == n
#@test A[2:end,2:end] ≈ QA[2:end,2:end]
@test A[:,:] ≈ F[:,:]



