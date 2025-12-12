
T = ChebyshevT()
C = Ultraspherical(2)
D_2 = C\diff(T,2)
x = axes(T,1)
M = C \ (x .* T)

# u(-1) = 0
# u''-x*u = 0

A = [T[-1,:]'; D_2-M]

# A*k = 0
# Let A' = QR
# R'Q'*k = 0

n = 100
Q, R = qr(A[1:n,1:n]')

k = expand(T, x -> 1+x)

norm(A[1:n,1:n] * Q[:,end])

A*k.args[2]

R