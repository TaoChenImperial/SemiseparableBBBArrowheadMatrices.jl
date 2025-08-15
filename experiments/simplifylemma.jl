using LinearAlgebra, Test


n = 10; u,v,d,w,s = [randn(n) for _=1:5]
A = tril(u*v',-1) + Diagonal(d) + triu(w*s',1)
uʲ = [0; u[2:end]]
e₁ = [1; zeros(n-1)]
B = A + uʲ * uʲ'A + uʲ * e₁'A
@test count(≥(1E-13), svdvals((B)[5:end,1:4])) == 1
@test count(≥(1E-13), svdvals((B)[1:4,5:end])) == 2


u¹ = u[2:end]; v¹ = v[2:end]; d¹ = d[2:end]; w¹ = w[2:end]; s¹ = s[2:end]; k,q = randn(2)
B =  tril(u¹*v¹',-1) + Diagonal(d¹) + triu(w¹*s¹',1)
C = B + k * u¹*s¹' + q * u¹ * u¹'B
y = C[:,1]
y[1] += norm(y); y = y/norm(y)
Q = I-2y*y'
@test count(≥(1E-13), svdvals((Q*C)[5:end,1:4])) == 1
@test count(≥(1E-13), svdvals((Q*C)[1:4,5:end])) == 2





function diag_plus_semi_plus_specialrank1(u,v, d, w, s, k, q)
    B =  tril(u*v',-1) + Diagonal(d) + triu(w*s',1) # diag+Semi
    B + u * (k*s' + q * u'B) # + special rank 1 == (I+ q * u*u')*B + k*u*s'
end

n = 10; u,v,d,w,s = [randn(n) for _=1:5]; k,q = randn(2);
A = diag_plus_semi_plus_specialrank1(u,v, d, w, s, k, q)
@test count(≥(1E-13), svdvals((A)[5:end,1:4])) == 1
@test count(≥(1E-13), svdvals((A)[1:4,5:end])) == 2

y = A[:,1]; y[1] += norm(y); y = y/norm(y); Q = I-2y*y'
@test count(≥(1E-13), svdvals((Q*A)[5:end,1:4])) == 1
@test count(≥(1E-13), svdvals((Q*A)[1:4,5:end])) == 2

# just submatrix
ũ,ṽ,d̃,w̃,s̃, k̃,q̃ = u[2:end],v[2:end],d[2:end],w[2:end],s[2:end],k,q # ????
@test_broken (Q*A)[2:end,2:end] == diag_plus_semi_plus_specialrank1(ũ,ṽ,d̃,w̃,s̃, k̃,q̃)

# full ,matrix
ũ,ṽ,d̃,w̃,s̃, k̃,q̃ = u,v,d,w,s,k,q # ????
@test_broken (Q*A) == diag_plus_semi_plus_specialrank1(ũ,ṽ,d̃,w̃,s̃, k̃,q̃)

# actually, want to show that the factors matrix has this structure....
F = LinearAlgebra.qrfactUnblocked!(copy(A)).factors
@test count(≥(1E-13), svdvals(F[5:end,1:4])) == 1
@test count(≥(1E-13), svdvals(F[1:4,5:end])) == 2


