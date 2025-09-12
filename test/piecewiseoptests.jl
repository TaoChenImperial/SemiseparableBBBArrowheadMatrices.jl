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

#Q̃ = BlockedArray(ql(Matrix(A[:,Block.(3:N)])).Q * I, axes(A)) # Householder applied to columns Block(3:N)
#Q̃ = BlockedArray(ql(Matrix(A[:,l :end])).Q * I, axes(A)) 
#@test Q̃'Q̃ ≈ I
#LL = Q̃'A


# test if τ equals τ_true on the 4th to Nth blocks
@test τ[l + l-1 + l-1 + 1 : end] ≈ τ_true[l + l-1 + l-1 + 1 : end]

# test if L equals F after HT on column blocks whose index are greater than 3
@test L[Block.(4:N), :] ≈ F[Block.(4:N), :] # that is L[l + l-1 + l-1 + 1 : end, :]
@test L[Block.(1:3), Block.(4:N)] ≈ F[Block.(1:3), Block.(4:N)] # that is L[Block.(1:3), Block.(4:N)]

println("Householder transformations for column blocks whose index are greater than 3 are successful")

# test if τ equals τ_true on the 3rd block
@test τ[l + l-1 + 1 : l + l-1 + l-1] ≈ τ_true[l + l-1 + 1 : l + l-1 + l-1]

# test if L equals F after HT on the 3rd column block
@test L[Block(1, 3)] ≈ F[Block(1, 3)] 
@test L[Block(3, 3)] ≈ F[Block(3, 3)] 
@test L[Block(3, 2)] ≈ F[Block(3, 2)] 
@test L[Block(3, 1)] ≈ F[Block(3, 1)] 

println("Householder transformations for the 3rd column block are successful")

# test if τ equals τ_true on the 2nd block
@test τ[l + 1 : l + l-1] ≈ τ_true[l + 1 : l + l-1]

# test if L equals F after HT on the 2nd column block
@test L[Block(1, 2)] ≈ F[Block(1, 2)] 
@test L[Block(2, 2)] ≈ F[Block(2, 2)] 
@test L[Block(2, 1)] ≈ F[Block(2, 1)] 

println("Householder transformations for the 2nd column block are successful")

# test if τ equals τ_true on the 1st block
@test τ[1 : l] ≈ τ_true[1 : l]

# test if L equals F after HT on the 1st column block
@test L[Block(1, 1)] ≈ F[Block(1, 1)] 

println("Householder transformations for the 1st column block are successful")

a,b = size(L)

X = rand(a)
sol_true = Matrix(A) \ X
sol = fast_solver(L, τ, X)

@test sol[1:end] ≈ sol_true[1:end]
println("Obtained the true solution")





