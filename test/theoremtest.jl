using BandedMatrices, SemiseparableMatrices, LinearAlgebra, Test
using LazyArrays: pad

ispadded(A::AbstractVector, a, m) = A == pad(A[1:min(a,m)], m)
ispadded(A::AbstractMatrix, (a,b), (m,n)) = A == pad(A[1:min(a,m),1:min(b,n)], m, n)

function 𝒫(A, J, K, E, X, Y, Z)
    (; B, U, V, W, S) = A
    r = size(U,2)
    p = size(W,2)
    n = size(A,1)
    ℓ,m = bandwidths(B)
    @assert size(J) == (r,p)
    @assert size(K) == (r,r)
    @assert size(E) == (r,n)
    @assert all(iszero, E[:,ℓ+m+1:end])
    @assert size(X) == (n,p)
    @assert all(iszero, X[ℓ+1:end,:])
    @assert size(Y) == (n,r)
    @assert all(iszero, Y[ℓ+1:end,:])
    @assert size(Z) == (n,n)
    @assert all(iszero, Z[ℓ+1:end,:])
    @assert all(iszero, Z[:,ℓ+m+1:end])

    U*J*S' + U*K*U'A + U*E + X*S' + Y*U'A + Z
end

@testset "Section 3" begin
    ℓ,m = 2,1
    r,p = 3,2
    n = 20
    B = brand(n,n,ℓ,m)
    U,V = randn(n,r),randn(n,r)
    W,S = randn(n,p),randn(n,p)
    A = BandedPlusSemiseparableMatrix(B, (U,V), (W,S))
    for k = 2:3
        @test A[k:n,k:n] ≈ B[k:n,k:n] + tril(U[k:n,:]V[k:n,:]',-1) + triu(W[k:n,:]S[k:n,:]',1)
    end

    @testset "Definition 3.3" begin
        for k = 1:n
            @test ispadded(B[k:n,k], ℓ+1, n-k+1)
            @test ispadded(B[k,k:n], m+1, n-k+1)
        end
    end

    @testset "Proposition 3.5" begin
        ℓ,m = 2,1
        r,p = 3,2
        n = 20
        B = brand(n,n,ℓ,m)
        U,V = randn(n,r),randn(n,r)
        W,S = randn(n,p),randn(n,p)
        A = BandedPlusSemiseparableMatrix(B, (U,V), (W,S))
        J, K, E, X, Y, Z = randn(r,p), randn(r,r), pad(randn(r, ℓ+m), r, n), pad(randn(ℓ,p), n, p), pad(randn(ℓ,r), n, r), pad(randn(ℓ,ℓ+m), n, n)

        @test "proof" begin
            @test 𝒫(A, J, K, E, X, Y, Z)[2:n,2:n] ≈ U[2:n,:]J*S[2:n,:]' + U[2:n,:]K*U'A[:,2:n] + U[2:n,:]E[:,2:n] + X[2:n,:]S[2:n,:]' + Y[2:n,:]U'A[:,2:n] + Z[2:n,2:n]
            @test U'A[:,2:n] ≈ U[2:n,:]'A[2:n,2:n] + U[1,:]A[1,2:n]'


            @test U[2:n,:]K*U[1,:]A[1,2:n]' ≈ U[2:n,:]K*U[1,:]B[1,2:n]' + U[2:n,:]K*U[1,:]*W[1,:]'S[2:n,:]'
            @test ispadded(K*U[1,:]B[1,2:n]', (r,m), (r,n-1))
            @test size(K*U[1,:]*W[1,:]') == (r,p)

            @test Y[2:n,:]*U[1,:]A[1,2:n]' ≈ Y[2:n,:]*U[1,:]B[1,2:n]' + Y[2:n,:]*U[1,:]*W[1,:]'S[2:n,:]'
            @test ispadded(Y[2:n,:]*U[1,:]B[1,2:n]', (ℓ-1,m), (n-1,n-1))
            @test ispadded(Y[2:n,:]*U[1,:]*W[1,:]', (ℓ-1,p), (n-1,p))

            Ã = BandedPlusSemiseparableMatrix(B[2:n,2:n], (U[2:n,:], V[2:n,:]), (W[2:n,:], S[2:n,:]))

            @test 𝒫(A, J, K, E, X, Y, Z)[2:n,2:n] ≈ 𝒫(Ã, J+K*U[1,:]*W[1,:]', K, E[:,2:n]+K*U[1,:]B[1,2:n]', X[2:n,:]+Y[2:n,:]*U[1,:]*W[1,:]', Y[2:n,:], Z[2:n,2:n]+Y[2:n,:]*U[1,:]B[1,2:n]')
        end
    end
end