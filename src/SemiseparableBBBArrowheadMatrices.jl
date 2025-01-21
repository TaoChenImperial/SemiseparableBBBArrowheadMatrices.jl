module SemiseparableBBBArrowheadMatrices
using LinearAlgebra, BlockArrays, BlockBandedMatrices, BandedMatrices, MatrixFactorizations, LazyBandedMatrices, LazyArrays
using PiecewiseOrthogonalPolynomials
#import ArrayLayouts: MemoryLayout, sublayout, sub_materialize, symmetriclayout, transposelayout, SymmetricLayout, HermitianLayout, TriangularLayout, layout_getindex, materialize!, MatLdivVec, AbstractStridedLayout, triangulardata, MatMulMatAdd, MatMulVecAdd, _fill_lmul!, layout_replace_in_print_matrix
import BandedMatrices: isbanded, bandwidths
import BlockArrays: BlockSlice, block, blockindex, blockvec
import BlockBandedMatrices: subblockbandwidths, blockbandwidths, AbstractBandedBlockBandedLayout, AbstractBandedBlockBandedMatrix
import Base: size, axes, getindex, +, -, *, /, ==, \, OneTo, oneto, replace_in_print_matrix, copy, diff, getproperty, adjoint, transpose, tail, _sum, inv, show, summary

export SemiseparableBBBArrowheadMatrix, copyBBBArrowheadMatrices, fast_ql

struct SemiseparableBBBArrowheadMatrix{T} <: AbstractBandedBlockBandedMatrix{T}
    # banded parts
    A::BandedMatrix{T}
    B::NTuple{2,BandedMatrix{T}} # first row blocks
    C::NTuple{4,BandedMatrix{T}} # first col blocks
    D::Vector{BandedMatrix{T}} # these are interlaces

    # fill parts
    Asub::NTuple{2,Vector{T}}
    Asup::NTuple{2,Matrix{T}} # matrices are m × 2

    Bsup::NTuple{2,Vector{T}}
    Bsub::NTuple{2,NTuple{2,Vector{T}}}

    Csup::NTuple{2,NTuple{2,Vector{T}}}
    Csub::NTuple{2,Vector{T}}

    A22sub::NTuple{2,Vector{T}}
    A32sup::NTuple{2,Vector{T}}

    A31extra::Vector{T}
    A32extra::Vector{T}
    A33extra::Vector{T}

end

function axes(L::SemiseparableBBBArrowheadMatrix) 
    ξ,n = size(L.A)
    m = length(L.D)
    μ,ν = size(L.D[1])
    blockedrange(Vcat(ξ, Fill(m,μ))), blockedrange(Vcat(n, Fill(m,ν)))
end

function getindex(L::SemiseparableBBBArrowheadMatrix{T}, Kk::BlockIndex{1}, Jj::BlockIndex{1})::T where T
    K,k = block(Kk),blockindex(Kk)
    J,j = block(Jj),blockindex(Jj)

    if K == Block(1) && J == Block(1)
        if k == j || k == j + 1 || k == j - 1
            return L.A[k, j]
        elseif k > j
            return L.Asub[1][k - 2] * L.Asub[2][j]
        else
            return L.Asup[1][k, 1] * L.Asup[2][j - 2, 1]+L.Asup[1][k, 2] * L.Asup[2][j - 2, 2]
        end
    end

    if K == Block(1)
        if J == Block(2)
            if k == j || k == j + 1
                return L.B[1][k, j]
            elseif k > j
                return L.Bsub[1][1][k-2] * L.Bsub[1][2][j]
            else
                return L.Bsup[1][k] * L.Bsup[2][j - 1]
            end
        elseif J == Block(3)
            if k == j || k == j + 1
                return L.B[2][k, j]
            elseif k > j
                return L.Bsub[2][1][k - 2] * L.Bsub[2][2][j]
            else
                return zero(T)
            end
        else
            return zero(T)
        end
    end


    if J == Block(1)
        if K == Block(2)
            if k == j || k == j - 1
                return L.C[1][k, j]
            elseif k > j
                return L.Csub[1][k - 1] * L.Csub[2][j]
            else
                return L.Csup[1][1][k] * L.Csup[1][2][j - 2]
            end
        elseif K == Block(3)
            if k == j || k == j - 1
                return L.C[2][k, j]
            elseif k == j + 1
                return L.A31extra[j]
            elseif k < j
                return L.Csup[2][1][k] * L.Csup[2][2][j - 2]
            else
                return zero(T)
            end
        elseif K == Block(4) || K == Block(5)
            if k == j || k == j - 1
                return L.C[Int(K) - 1][k, j]
            else
                return zero(T)
            end
        else
            return zero(T)
        end
    end


    if K == Block(2) && J == Block(2) && k > j
        return L.A22sub[1][k - 1] * L.A22sub[2][j]
    end

    if K == Block(3) && J == Block(2)
        if k < j
            return L.A32sup[1][k] * L.A32sup[2][j - 1]
        elseif k == j + 1
            return L.A32extra[j]
        end
    end

    if K == Block(3) && J == Block(3) && k == j + 1
        return L.A33extra[j]
    end

    #values stored in D
    if k == j 
        return L.D[k][Int(K) - 1, Int(J) - 1]
    else
        return zero(T)
    end
end


function getindex(L::SemiseparableBBBArrowheadMatrix, k::Int, j::Int)
    ax,bx = axes(L)
    L[findblockindex(ax, k), findblockindex(bx, j)]
end

###
# QL
####

function fast_ql(M::BBBArrowheadMatrix{T}) where T
    m,n = size(M.A)
    l = length(M.D)
    m2, n2 = size(M.D[1])
    @assert m == n == l+1
    @assert m2 == n2
    L = copyBBBArrowheadMatrices(M)
    τ = zeros(m + l * m2)
    HT_column_block_above3(L, τ)
    HT_column_block_3(L, τ)
    #HT_column_block_2(L, τ)
    #HT_column_block_1(L, τ)
    L, τ
end

#Householder transformations for different blocks of columns
function HT_column_block_above3(L::SemiseparableBBBArrowheadMatrix, τ::Vector{T}) where T
    m,n = size(L.A)
    l = length(L.D)
    m2, n2 = size(L.D[1])
    for j in m2 : -1 : 3
        for i in l : -1 : 1
            upper_entry = L[Block(j - 1)[i],Block(j + 1)[i]] #L.D[i][j-2,j]
            dia_entry = L[Block(j + 1)[i],Block(j + 1)[i]] #L.D[i][j,j]
            #perform Householder transformation
            dia_entry_new = -sign(dia_entry) * sqrt(dia_entry^2 + upper_entry^2)
            v = [upper_entry, dia_entry-dia_entry_new]
            coef = 2 / (v[1]^2 + v[2]^2)
            #denote the householder transformation as [c1 s1;c2 s2]
            c1 = 1 - coef * v[1]^2
            s1 = - coef * v[1] * v[2]
            c2 = s1
            s2 = 1 - coef * v[2]^2
            L.D[i][j, j] = dia_entry_new #update L[Block(j + 1)[i],Block(j + 1)[i]]
            L.D[i][j-2,j] = v[1]/v[2] #update L[Block(j - 1)[i],Block(j + 1)[i]]
            τ[m + (j - 1) * l + i] = coef * v[2]^2
            #row recombination(householder transformation) for other columns
            current_upper_entry = L[Block(j - 1)[i],Block(j - 1)[i]] #L.D[i][j-2,j-2]
            current_lower_entry = L[Block(j + 1)[i],Block(j - 1)[i]] #L.D[i][j,j-2]
            L.D[i][j-2,j-2] = c1 * current_upper_entry + s1 * current_lower_entry #update L[Block(j - 1)[i],Block(j - 1)[i]]
            L.D[i][j,j-2] = c2 * current_upper_entry + s2 * current_lower_entry #update L[Block(j + 1)[i],Block(j - 1)[i]]
            if j >= 5
                #Deal with L.D blocks which do not share common rows with A.C
                current_entry = L[Block(j - 1)[i],Block(j - 3)[i]] #L.D[i][j-2,j-4]
                L.D[i][j-2,j-4] = c1 * current_entry #update L[Block(j - 1)[i],Block(j - 3)[i]] 
                L.D[i][j,j-4] = c2 * current_entry #update L[Block(j + 1)[i],Block(j - 3)[i]]
            else
                #Deal with L.D blocks which share common rows with L.C
                current_entry = L[Block(j - 1)[i],Block(1)[i]] #L.C[j-2][i,i]
                L.C[j-2][i,i] = c1 * current_entry #update L[Block(j - 1)[i],Block(1)[i]] 
                L.C[j][i,i] = c2 * current_entry #update L[Block(j + 1)[i],Block(1)[i]]

                current_entry = L[Block(j - 1)[i],Block(1)[i + 1]] #L.C[j-2][i,i+1]
                L.C[j-2][i,i + 1] = c1 * current_entry #update L[Block(j - 1)[i],Block(1)[i + 1]]
                L.C[j][i,i + 1] = c2 * current_entry #update L[Block(j + 1)[i],Block(1)[i + 1]]
            end
        end
    end
end


function HT_column_block_3(L::SemiseparableBBBArrowheadMatrix, τ::Vector{T}) where T
    m,n = size(L.A)
    l = length(L.D)
    # m2, n2 = size(L.D[1])
    x_len = 0
    L.Bsub[2][2][l - 1] = 1.0 # it can be choosen arbitrarily.
    for i in l:-1:1
        a = L[Block(1)[i],Block(3)[i]] 
        b = L[Block(1)[i + 1],Block(3)[i]] 
        c = L[Block(3)[i],Block(3)[i]]
        λ = i < l ? L.Bsub[2][2][i] : 0
        v_last = c + sign(c) * sqrt(a^2 + b^2 + λ^2 * x_len^2 + c^2)
        v_len = sqrt(a^2 + b^2 + λ^2 * x_len^2 + v_last^2)
        L.D[i][2, 2] = -sign(c) * sqrt(a^2 + b^2 + λ^2 * x_len^2 + c^2)
        if 1 < i < l
            L.Bsub[2][2][i - 1] = -2 / v_len^2 * a * L[Block(1)[i],Block(3)[i - 1]] * L.Bsub[2][2][i]
        end
        if i > 1
            L.Bsub[2][1][i - 1] = -2 / v_len^2 * b * a * L[Block(1)[i],Block(3)[i - 1]] / L.Bsub[2][2][i - 1]
            L.A33extra[i - 1] = -2 / v_len^2 * v_last * a * L[Block(1)[i],Block(3)[i - 1]] 
            L.B[2][i, i - 1] = L.B[2][i, i - 1] - 2 / v_len^2 * L.B[2][i, i - 1] * a * a
        end
        #record information of v
        L.B[2][i, i] = a / v_last
        L.B[2][i + 1, i] = b / v_last
        if i < l
            L.Bsub[2][2][i] = L.Bsub[2][2][i] / v_last
        end
        τ[m + l + i] = 2 * v_last^2 / v_len^2
        if i > 1
            x_len = sqrt(x_len^2 + L.Bsub[2][1][i - 1]^2) 
        end
    end
end

function HT_column_block_2(L::SemiseparableBBBArrowheadMatrix, τ::Vector{T}) where T
end

function HT_column_block_1(L::SemiseparableBBBArrowheadMatrix, τ::Vector{T}) where T
end

function copyBBBArrowheadMatrices(M::BBBArrowheadMatrix{T}) where T
    m,n = size(M.A)
    l = length(M.D)
    m2, n2 = size(M.D[1])

    A = BandedMatrix(zeros(m, n), (1, 1))
    for i in 1 : m
        if i > 1 A[i, i - 1] = M.A[i, i - 1] end
        A[i, i] = M.A[i, i]
        if i < n A[i, i + 1] = M.A[i, i + 1] end
    end

    B = (BandedMatrix(zeros(m, l), (1, 0)), BandedMatrix(zeros(m, l), (1, 0)))
    for j in 1 : 2
        for i in 1 : l
            B[j][i, i] = M.B[j][i, i]
            B[j][i + 1, i] = M.B[j][i + 1, i]
        end
    end

    C = (BandedMatrix(zeros(l, n), (0, 1)), BandedMatrix(zeros(l, n), (0, 1)), BandedMatrix(zeros(l, n), (0, 1)), BandedMatrix(zeros(l, n), (0, 1)))
    for j in 1 : 2
        for i in 1 : l
            C[j][i, i] = M.C[j][i, i]
            C[j][i, i + 1] = M.C[j][i, i + 1]
        end
    end

    D = [BandedMatrix(zeros(m2, n2), (min(4, m2), min(2, n2))) for _ in 1:l]
    for j in 1 : l
        for i in 1 : m2
            if i > 2 
                D[j][i, i - 2] = M.D[j][i, i - 2] 
            end
            D[j][i, i] = M.D[j][i, i]
            if i < m2 - 1 
                D[j][i, i + 2] = M.D[j][i, i + 2] 
            end
        end
    end

    Asub = (zeros(m - 2), zeros(n - 2))
    Asup = (zeros(m - 2, 2), zeros(n - 2, 2))
    Bsup = (zeros(m - 2), zeros(l - 1))
    Bsub = ((zeros(m - 2), zeros(l - 1)), (zeros(m - 2), zeros(l - 1)))

    Csup = ((zeros(l - 1), zeros(n - 2)), (zeros(l - 1), zeros(n - 2)))
    Csub = (zeros(l - 1), zeros(n - 2))

    A22sub = (zeros(l - 1), zeros(l - 1))
    A32sup = (zeros(l - 1), zeros(l - 1))

    A31extra = zeros(l - 1)
    A32extra = zeros(l - 1)
    A33extra = zeros(l - 1)

    SemiseparableBBBArrowheadMatrix(A, B, C, Array{BandedMatrix{Float64}, 1}(D), Asub, Asup, Bsup, Bsub, Csup, Csub, A22sub, A32sup, A31extra, A32extra, A33extra)

end

end

