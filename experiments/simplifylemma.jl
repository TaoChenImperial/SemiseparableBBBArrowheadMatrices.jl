using LinearAlgebra, Test

function diag_plus_semi_plus_specialrank1(u,v, d, w, s, q, k)
    B =  tril(u*v',-1) + Diagonal(d) + triu(w*s',1) # diag+Semi
    B + u * (q*s' + k * u'B) # + special rank 1 == (I+ q * u*u')*B + k*u*s'
end

n = 10; u,v,d,w,s = [randn(n) for _=1:5]; q,k = randn(2);
B = tril(u*v',-1) + Diagonal(d) + triu(w*s',1)
A = diag_plus_semi_plus_specialrank1(u,v, d, w, s, q, k)
@test count(≥(1E-13), svdvals((A)[5:end,1:4])) == 1
@test count(≥(1E-13), svdvals((A)[1:4,5:end])) == 2

y = A[:,1]; y[1] += sign(y[1])*norm(y); y = y/norm(y); Q = I-2y*y'
@test count(≥(1E-13), svdvals((Q*A)[5:end,1:4])) == 1
@test count(≥(1E-13), svdvals((Q*A)[1:4,5:end])) == 2

# just submatrix
u_2 = u[1:end]
u_2[1] = 0
τ = 2*y[1]*y[1]
k̅ = sqrt(2*y[2]^2/(τ*u[2]^2))*(sign(y[1])*sign(y[2])/sign(u[2]))
γ = -τ*q*e₁'*u-τ*q*k̅*u_2'*u
ϕ = -τ*q*k̅*e₁'*u-τ*q*k̅^2*u_2'*u
ξ = -τ*k*e₁'*u-τ*k*k̅*u_2'*u
η = -τ*k*k̅*e₁'*u-τ*k*k̅^2*u_2'*u

ũ,ṽ,d̃,w̃,s̃= u[2:end],v[2:end],d[2:end],w[2:end],s[2:end]
q̃ = q + k*u[1]*w[1]-k̅*τ*w[1]+ϕ+η*u[1]*w[1]
k̃ = k - k̅*k̅*τ + η
(I-τ*(e₁+k̅ *u_2)*(e₁'+k̅ *u_2'))*A
@test (Q*A)[2:end,2:end] ≈ diag_plus_semi_plus_specialrank1(ũ,ṽ,d̃,w̃,s̃,q̃,k̃)

α = w[1] + q*u[1] + k*u[1]^2*w[1] - τ*w[1] + γ - k*u[1]^2*w[1] + τ*k̅*w[1]*u[1]
β = k*u[1] - τ*k̅ + ξ
ss = u'B
@test (Q*A)[1,2:end]' ≈ α*s[2:end]' + β*ss[2:end]'

# actually, want to show that the factors matrix has this structure....
F = LinearAlgebra.qrfactUnblocked!(copy(A)).factors
@test count(≥(1E-13), svdvals(F[5:end,1:4])) == 1
@test count(≥(1E-13), svdvals(F[1:4,5:end])) == 2


