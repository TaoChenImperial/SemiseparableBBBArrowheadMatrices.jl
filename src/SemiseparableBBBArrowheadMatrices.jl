module SemiseparableBBBArrowheadMatrices
using LinearAlgebra, BlockArrays, BlockBandedMatrices, BandedMatrices, MatrixFactorizations, LazyBandedMatrices, LazyArrays
using PiecewiseOrthogonalPolynomials
#import ArrayLayouts: MemoryLayout, sublayout, sub_materialize, symmetriclayout, transposelayout, SymmetricLayout, HermitianLayout, TriangularLayout, layout_getindex, materialize!, MatLdivVec, AbstractStridedLayout, triangulardata, MatMulMatAdd, MatMulVecAdd, _fill_lmul!, layout_replace_in_print_matrix
import BandedMatrices: isbanded, bandwidths
import BlockArrays: BlockSlice, block, blockindex, blockvec, viewblock
import BlockBandedMatrices: blockbandwidths, AbstractBlockBandedLayout, AbstractBlockBandedMatrix
import Base: size, axes, getindex, +, -, *, /, ==, \, OneTo, oneto, replace_in_print_matrix, copy, diff, getproperty, adjoint, transpose, tail, _sum, inv, show, summary
using SemiseparableMatrices: LowRankMatrix, LayoutMatrix

export SemiseparableBBBArrowheadMatrix, copyBBBArrowheadMatrices, fast_ql


struct BandedPlusSemiseparableMatrix{T,D,A,B,R} <: LayoutMatrix{T}
    bands::BandedMatrix{T,D,R}
    upperfill::LowRankMatrix{T,A,B}
    lowerfill::LowRankMatrix{T,A,B}
    #BandedPlusSemiseparableMatrix{T,D,A,B,R}(bands, upperfill, lowerfill) where {T,D,A,B,R} = new{T,D,A,B,R}(bands, upperfill, lowerfill)
end


size(A::BandedPlusSemiseparableMatrix) = size(A.bands)

function getindex(A::BandedPlusSemiseparableMatrix, k::Integer, j::Integer)
    b1 = bandwidth(A.bands,1)
    b2 = bandwidth(A.bands,2)
    if j > k + b2
        A.upperfill[k,j-b2-1]
    elseif k > j + b1
        A.lowerfill[k-b1-1,j]
    else
        A.bands[k,j]
    end
end

struct SemiseparableBBBArrowheadMatrix{T} <: AbstractBlockBandedMatrix{T}
    # banded parts
    A::BandedMatrix{T}
    B::NTuple{2,BandedMatrix{T}} # first row blocks
    C::NTuple{4,BandedMatrix{T}} # first col blocks
    D::Vector{BandedMatrix{T}} # these are interlaces

    # fill parts
    Asub::NTuple{2,Vector{T}}
    Asup::NTuple{2,Matrix{T}} # matrices are m × 2
    Asub_extra::NTuple{2,Vector{T}} #It turned out that the sub part of Block(1,1) has rank 2 at some stage

    Bsup::NTuple{2,Vector{T}}
    Bsub::NTuple{2,NTuple{2,Vector{T}}}
    Bsub_extra::NTuple{2,Vector{T}} #It turned out that the sub part of Block(1,2) has rank 2

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

blockbandwidths(L::SemiseparableBBBArrowheadMatrix) = (4,2)

function getindex(L::SemiseparableBBBArrowheadMatrix{T}, Kk::BlockIndex{1}, Jj::BlockIndex{1})::T where T
    K,k = block(Kk),blockindex(Kk)
    J,j = block(Jj),blockindex(Jj)

    if K == Block(1) && J == Block(1)
        if k == j || k == j + 1 || k == j - 1
            return L.A[k, j]
        elseif k > j
            return L.Asub[1][k - 2] * L.Asub[2][j] + L.Asub_extra[1][k - 2] * L.Asub_extra[2][j]
        else
            return L.Asup[1][k, 1] * L.Asup[2][j - 2, 1] + L.Asup[1][k, 2] * L.Asup[2][j - 2, 2]
        end
    end

    if K == Block(1)
        if J == Block(2)
            if k == j || k == j + 1
                return L.B[1][k, j]
            elseif k > j
                return L.Bsub[1][1][k-2] * L.Bsub[1][2][j] + L.Bsub_extra[1][k-2] * L.Bsub_extra[2][j]
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

function viewblock(L::SemiseparableBBBArrowheadMatrix, KJ::Block{2})
    K,J = KJ.n
    ξ,n = size(L.A)
    m = length(L.D)
    μ,ν = size(L.D[1])
    if K == 1 && J == 1
        upperfill = LowRankMatrix(Matrix(L.Asup[1]), Matrix(L.Asup[2]'))
        lowerfill = LowRankMatrix(reshape(L.Asub[1],:,1), Matrix(L.Asub[2]'))
        return BandedPlusSemiseparableMatrix(L.A, upperfill, lowerfill)
    elseif K == 1 && J == 2
        upperfill = LowRankMatrix(reshape(L.Bsup[1],:,1), Matrix(L.Bsup[2]'))
        lowerfill = LowRankMatrix([L.Bsub[1][1] L.Bsub_extra[1]], Matrix([L.Bsub[1][2] L.Bsub_extra[2]]') )
        return BandedPlusSemiseparableMatrix(L.B[1], upperfill, lowerfill)
    elseif K == 1 && J == 3
        upperfill = LowRankMatrix(reshape(zeros(m-1),:,1), Matrix(zeros(ξ-2)'))
        lowerfill = LowRankMatrix(reshape(L.Bsub[2][1],:,1), Matrix(L.Bsub[2][2]'))
        return BandedPlusSemiseparableMatrix(L.B[2], upperfill, lowerfill)
    elseif K == 2 && J == 1
        upperfill = LowRankMatrix(reshape(L.Csup[1][1],:,1), Matrix(L.Csup[1][2]'))
        lowerfill = LowRankMatrix(reshape(L.Csub[1],:,1), Matrix(L.Csub[2]'))
        return BandedPlusSemiseparableMatrix(L.C[1], upperfill, lowerfill)
    elseif K == 2 && J == 2
        upperfill = LowRankMatrix(reshape(zeros(m-1),:,1), Matrix(zeros(m-1)'))
        lowerfill = LowRankMatrix(reshape(L.A22sub[1],:,1), Matrix(L.A22sub[2]'))
        return BandedPlusSemiseparableMatrix(BandedMatrix(Diagonal([L.D[x][1,1] for x in 1:m ])), upperfill, lowerfill)
    elseif K == 2 && J == 3
        upperfill = LowRankMatrix(reshape(zeros(m-1),:,1), Matrix(zeros(m-1)'))
        lowerfill = LowRankMatrix(reshape(zeros(m-1),:,1), Matrix(zeros(m-1)'))
        return BandedPlusSemiseparableMatrix(BandedMatrix(zeros(m,m), (0,0)), upperfill, lowerfill)
    elseif K == 3 && J == 1
        bands = BandedMatrix(zeros(m,n),(1,1))
        for i in 1:m
            bands[i, i] = L.C[2][i, i]
            bands[i, i+1] = L.C[2][i, i+1]
            if i > 1
                bands[i, i-1] = L.A31extra[i-1]
            end
        end
        upperfill = LowRankMatrix(reshape(L.Csup[2][1],:,1), Matrix(L.Csup[2][2]'))
        lowerfill = LowRankMatrix(reshape(zeros(m-2),:,1), Matrix(zeros(n-2)'))
        return BandedPlusSemiseparableMatrix(bands, upperfill, lowerfill)
    elseif K == 3 && J == 2
        bands = BandedMatrix(zeros(m,m),(1,0))
        for i in 1:m
            bands[i, i] = L.D[i][2, 1]
            if i > 1
                bands[i, i-1] = L.A32extra[i-1]
            end
        end
        upperfill = LowRankMatrix(reshape(L.A32sup[1],:,1), Matrix(L.A32sup[2]'))
        lowerfill = LowRankMatrix(reshape(zeros(m-2),:,1), Matrix(zeros(m-2)'))
        return BandedPlusSemiseparableMatrix(bands, upperfill, lowerfill)
    elseif K == 3 && J == 3
        bands = BandedMatrix(zeros(m,m),(1,0))
        for i in 1:m
            bands[i, i] = L.D[i][2, 2]
            if i > 1
                bands[i, i-1] = L.A33extra[i-1]
            end
        end
        upperfill = LowRankMatrix(reshape(zeros(m-1),:,1), Matrix(zeros(m-1)'))
        lowerfill = LowRankMatrix(reshape(zeros(m-2),:,1), Matrix(zeros(m-2)'))
        return BandedPlusSemiseparableMatrix(bands, upperfill, lowerfill)
    elseif 4 <= K <= 5 && J == 1
        upperfill = LowRankMatrix(reshape(zeros(n-2),:,1), Matrix(zeros(n-2)'))
        lowerfill = LowRankMatrix(reshape(zeros(m-1),:,1), Matrix(zeros(m-1)'))
        return BandedPlusSemiseparableMatrix(L.C[K-1], upperfill, lowerfill)
    elseif K == J || K == J-2 || K == J+2 || K == J+4
        upperfill = LowRankMatrix(reshape(zeros(m-1),:,1), Matrix(zeros(m-1)'))
        lowerfill = LowRankMatrix(reshape(zeros(m-1),:,1), Matrix(zeros(m-1)'))
        return BandedPlusSemiseparableMatrix(BandedMatrix(Diagonal([L.D[x][K-1,J-1] for x in 1:m ])), upperfill, lowerfill)
    elseif K <= μ + 1 && J == 1
        upperfill = LowRankMatrix(reshape(zeros(m-1),:,1), Matrix(zeros(n-2)'))
        lowerfill = LowRankMatrix(reshape(zeros(m-1),:,1), Matrix(zeros(n-2)'))
        return BandedPlusSemiseparableMatrix(BandedMatrix(zeros(m,n), (0,1)), upperfill, lowerfill)
    elseif K == 1 && J <= ν + 1
        upperfill = LowRankMatrix(reshape(zeros(ξ-2),:,1), Matrix(zeros(m-1)'))
        lowerfill = LowRankMatrix(reshape(zeros(ξ-2),:,1), Matrix(zeros(m-1)'))
        return BandedPlusSemiseparableMatrix(BandedMatrix(zeros(ξ,m), (1,0)), upperfill, lowerfill)
    elseif K <= μ + 1 && J <= ν + 1
        upperfill = LowRankMatrix(reshape(zeros(m-1),:,1), Matrix(zeros(m-1)'))
        lowerfill = LowRankMatrix(reshape(zeros(m-1),:,1), Matrix(zeros(m-1)'))
        return BandedPlusSemiseparableMatrix(BandedMatrix(zeros(m,m), (0,0)), upperfill, lowerfill)
    else
        error("Index not valid")
    end
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
    HT_column_block_2(L, τ)
    HT_column_block_1(L, τ)
    L, τ
end

#Householder transformations for different blocks of columns
function HT_column_block_above3(L::SemiseparableBBBArrowheadMatrix, τ::Vector{T}) where T
    m,n = size(L.A)
    l = length(L.D)
    m2, n2 = size(L.D[1])
    for j in m2 : -1 : 3
        for i in l : -1 : 1
            upper_entry = L.D[i][j-2,j] #L[Block(j - 1)[i],Block(j + 1)[i]]
            dia_entry = L.D[i][j,j] #L[Block(j + 1)[i],Block(j + 1)[i]]
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
            current_upper_entry = L.D[i][j-2,j-2] #L[Block(j - 1)[i],Block(j - 1)[i]] 
            current_lower_entry = L.D[i][j,j-2] #L[Block(j + 1)[i],Block(j - 1)[i]]
            L.D[i][j-2,j-2] = c1 * current_upper_entry + s1 * current_lower_entry #update L[Block(j - 1)[i],Block(j - 1)[i]]
            L.D[i][j,j-2] = c2 * current_upper_entry + s2 * current_lower_entry #update L[Block(j + 1)[i],Block(j - 1)[i]]
            if j >= 5
                #Deal with L.D blocks which do not share common rows with A.C
                current_entry = L.D[i][j-2,j-4] #L[Block(j - 1)[i],Block(j - 3)[i]]
                L.D[i][j-2,j-4] = c1 * current_entry #update L[Block(j - 1)[i],Block(j - 3)[i]] 
                L.D[i][j,j-4] = c2 * current_entry #update L[Block(j + 1)[i],Block(j - 3)[i]]
            else
                #Deal with L.D blocks which share common rows with L.C
                current_entry = L.C[j-2][i,i] #L[Block(j - 1)[i],Block(1)[i]]
                L.C[j-2][i,i] = c1 * current_entry #update L[Block(j - 1)[i],Block(1)[i]] 
                L.C[j][i,i] = c2 * current_entry #update L[Block(j + 1)[i],Block(1)[i]]

                current_entry = L.C[j-2][i,i+1] #L[Block(j - 1)[i],Block(1)[i + 1]]
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

    L.Bsub[1][1][m - 2] = 1.0 # it can be choosen arbitrarily.
    L.Bsup[2][l - 1] = 1.0 # it can be choosen arbitrarily.
    L.A32sup[2][l - 1] = 1.0 # it can be chose arbotrarily.
    L.Asup[2][n - 2, 1] = 1.0 # it can be choosen arbitrarily.
    L.Asub[1][m - 2] = 1.0 # it can be choosen arbitrarily.
    L.Csup[2][2][n - 2] = 1.0 # it can be choosen arbitrarily.
    for i in l:-1:1
        a = L.B[2][i, i] #L[Block(1)[i],Block(3)[i]] 
        b = L.B[2][i+1, i] #L[Block(1)[i + 1],Block(3)[i]] 
        c = L.D[i][2, 2] #L[Block(3)[i],Block(3)[i]]
        λ = i < l ? L.Bsub[2][2][i] : 0
        v_last = c + sign(c) * sqrt(a^2 + b^2 + λ^2 * x_len^2 + c^2)
        v_len = sqrt(a^2 + b^2 + λ^2 * x_len^2 + v_last^2)
        L.D[i][2, 2] = -sign(c) * sqrt(a^2 + b^2 + λ^2 * x_len^2 + c^2)
        if 1 < i < l
            temp = L.B[2][i, i-1] # L[Block(1)[i],Block(3)[i - 1]]
            L.Bsub[2][2][i - 1] = -2 / v_len^2 * a * temp * L.Bsub[2][2][i]
        end
        if i > 1
            temp = L.B[2][i, i-1] #L[Block(1)[i],Block(3)[i - 1]] 
            L.Bsub[2][1][i - 1] = -2 / v_len^2 * b * a * temp / L.Bsub[2][2][i - 1]
            L.A33extra[i - 1] = -2 / v_len^2 * v_last * a * temp
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
        HT_3to2(L, τ, i)
        HT_3to1(L, τ, i)
    end
end

function HT_3to2(L::SemiseparableBBBArrowheadMatrix, τ::Vector{T}, i) where T
    m,n = size(L.A)
    l = length(L.D)
    coef = τ[m + l + i]
    v = L[1:m, n + l + i]#L[Block(1,3)][:, i]

    # compute v'A[Block(1,2)]
    vTA_sub = zeros(l)
    vTA_dia = zeros(l)
    vTA_sup = zeros(l)
    sub_dotv = 0.0 # to compute sum(L.Bsub[1][1] .* v) iteratively
    sup_dotv = 0.0 # to compute sum(L.Bsup[1] .* v) iteratively 
    for k in 1 : l
        tempvar = L.B[1][k, k] #L[Block(1,2)][k, k]
        tempvar2 = L.B[1][k+1, k] #L[Block(1,2)][k + 1, k]
        vTA_dia[k] = v[k] * tempvar + v[k + 1] * tempvar2
    end
    for k in 2 : l
        sup_dotv = sup_dotv + L.Bsup[1][k - 1] * v[k - 1]
        vTA_sup[k] = L.Bsup[2][k-1] * sup_dotv
    end
    for k in l-1 : -1 : 1
        sub_dotv = sub_dotv + L.Bsub[1][1][k] * v[k + 2]
        vTA_sub[k] = L.Bsub[1][2][k] * sub_dotv
    end
    vTA = vTA_sub + vTA_dia + vTA_sup

    #update the banded part in L[Block(1,2)]
    for k in 1 : l
        L.B[1][k, k] = L.B[1][k, k] - coef * v[k] * vTA[k]
        L.B[1][k + 1, k] = L.B[1][k + 1, k] - coef * v[k + 1] * vTA[k]
    end

    #update the sub part in L[Block(1,2)]
    for k in 1 : l-1
        L.Bsub[1][2][k] = L.Bsub[1][2][k] - coef * v[m] * vTA[k] / L.Bsub[1][1][m-2]
    end
    if 1 < i < l
        L.Bsub[1][1][i - 1] = - coef * v[i + 1] * vTA[i - 1] / L.Bsub[1][2][i - 1]
    end

    #update the sup part in L[Block(1,2)]
    for k in 1 : m-2
        L.Bsup[1][k] = L.Bsup[1][k] - coef * v[k] * vTA[l] / L.Bsup[2][l-1]
    end
    if i < l - 1
        L.Bsup[2][i] = - coef * v[i] * vTA[i + 1] / L.Bsup[1][i]
    end

    #update L[Block(3,2)]
    L.D[i][2,1] = - coef * vTA[i]
    if i > 1
        L.A32extra[i - 1] = - coef * vTA[i - 1]
    end
    if i < l
        L.A32sup[1][i] = - coef * vTA[l] / L.A32sup[2][l-1]
    end
    if i < l - 1
        L.A32sup[2][i] = - coef * vTA[i + 1] / L.A32sup[1][i]
    end
end

function HT_3to1(L::SemiseparableBBBArrowheadMatrix, τ::Vector{T}, i) where T
    m,n = size(L.A)
    l = length(L.D)
    coef = τ[m + l + i]
    v = L[1:m, n + l + i]#L[Block(1,3)][:, i]

    # compute v'A[Block(1,2)]
    vTA_sub = zeros(n)
    vTA_dia = zeros(n)
    vTA_sup = zeros(n)
    sub_dotv = 0.0 # to compute sum(L.Bsub[1][1] .* v) iteratively
    sup_dotv = 0.0 # to compute sum(L.Bsup[1] .* v) iteratively 
    for k in 1 : n
        tempvar = L.A[k, k] #L[Block(1,1)][k, k]
        vTA_dia[k] = v[k] * tempvar
        if k > 1
            tempvar1 = L.A[k-1, k] #L[Block(1,1)][k - 1, k]
            vTA_dia[k] = vTA_dia[k] + v[k - 1] * tempvar1
        end
        if k < n
            tempvar2 = L.A[k+1, k] #L[Block(1,1)][k + 1, k]
            vTA_dia[k] = vTA_dia[k] + v[k + 1] * tempvar2
        end
    end
    for k in 3 : n
        sup_dotv = sup_dotv + L.Asup[1][k - 2, 1] * v[k - 2]
        vTA_sup[k] = L.Asup[2][k-2, 1] * sup_dotv
    end
    for k in n - 2 : -1 : 1
        sub_dotv = sub_dotv + L.Asub[1][k] * v[k + 2]
        vTA_sub[k] = L.Asub[2][k] * sub_dotv
    end
    vTA = vTA_sub + vTA_dia + vTA_sup
    # There are also some nonzero entries in A31
    vTA[i] = vTA[i] + L.C[2][i, i]
    vTA[i + 1] = vTA[i + 1] + L.C[2][i, i + 1]

    #update the banded part in L[Block(1,1)]
    for k in 1 : n
        L.A[k, k] = L.A[k, k] - coef * v[k] * vTA[k]
        if k > 1
            L.A[k - 1, k] = L.A[k - 1, k] - coef * v[k - 1] * vTA[k]
        end
        if k < n
            L.A[k + 1, k] = L.A[k + 1, k] - coef * v[k + 1] * vTA[k]
        end
    end

    #update the sub part in L[Block(1,1)]
    for k in 1 : n - 2
        L.Asub[2][k] = L.Asub[2][k] - coef * v[m] * vTA[k] / L.Asub[1][m-2, 1]
    end
    if 1 < i < l
        L.Asub[1][i - 1] = - coef * v[i + 1] * vTA[i - 1] / L.Asub[2][i - 1]
    end

    #update the sup part in L[Block(1,1)]
    for k in 1 : m - 2
        L.Asup[1][k, 1] = L.Asup[1][k, 1] - coef * v[k] * vTA[n] / L.Asup[2][n-2, 1]
    end
    if i < l - 1
        L.Asup[2][i] = - coef * v[i] * vTA[i + 2] / L.Asup[1][i]
    end

    #update L[Block(3,1)]
    L.C[2][i, i] = L.C[2][i, i] - coef * vTA[i]
    L.C[2][i, i + 1] = L.C[2][i, i + 1] - coef * vTA[i + 1]
    if i > 1
        L.A31extra[i - 1] = - coef * vTA[i - 1]
    end
    if i < l
        L.Csup[2][1][i] = - coef * vTA[n] / L.Csup[2][2][n-2]
    end
    if i < l - 1
        L.Csup[2][2][i] = - coef * vTA[i + 2] / L.Csup[2][1][i]
    end
end

function HT_column_block_2(L::SemiseparableBBBArrowheadMatrix, τ::Vector{T}) where T
    m,n = size(L.A)
    l = length(L.D)
    m2, n2 = size(L.D[1])

    # record the factors in sub Block(1,2) temporarily which is rank 2
    Bsub_factor = (zeros(m - 2), zeros(l - 1))
    Bsub_factor_extra = (zeros(m - 2), zeros(l - 1))

    # record additional Block(1,1) info when doing HT_2to1
    Asub_factor = (zeros(m - 2), zeros(l - 1))
    Asub_factor_extra = (zeros(m - 2), zeros(l - 1))

    Asup_factor = (zeros(m - 2), zeros(l - 1))
    Asup_factor_extra = (zeros(m - 2), zeros(l - 1))


    u1_square = 0 #suppose the sup part of Block(1,2) is u1*u2'
    for x in L.Bsup[1]
        u1_square = u1_square + x^2
    end

    for i in l:-1:1
        if i < l
            u1_square = u1_square - L.Bsup[1][i]^2
        end
        v_square = L.B[1][i,i]^2 + L.B[1][i+1, i]^2 + L.D[i][1,1]^2
        if i > 1
            v_square = v_square + L.Bsup[2][i-1]^2 * u1_square
        end
        if i < l
            for k in i:l-1
                v_square = v_square + (Bsub_factor[1][k] * Bsub_factor[2][i] + Bsub_factor_extra[1][k] * Bsub_factor_extra[2][i])^2 #L[Block(1,2)][k+2,i]^2 
            end
        end
        v_last = L.D[i][1,1] + sign(L.D[i][1,1]) * sqrt(v_square)
        L.D[i][1,1] = -sign(L.D[i][1,1]) * sqrt(v_square)
        #record the info of v
        if i > 1
            L.Bsup[2][i-1] = L.Bsup[2][i-1] / v_last
        end
        L.B[1][i,i] = L.B[1][i,i] / v_last
        L.B[1][i+1, i] = L.B[1][i+1, i] / v_last
        if i < l
            Bsub_factor[2][i] = Bsub_factor[2][i] / v_last
            Bsub_factor_extra[2][i] = Bsub_factor_extra[2][i] / v_last
        end

        v_new_square = 0
        if i > 1
            v_new_square = v_new_square + L.Bsup[2][i-1]^2 * u1_square
        end
        v_new_square = v_new_square + L.B[1][i,i]^2 + L.B[1][i+1, i]^2 + 1^2
        if i < l
            for k in i:l-1
                v_new_square = v_new_square + (Bsub_factor[1][k] * Bsub_factor[2][i] + Bsub_factor_extra[1][k] * Bsub_factor_extra[2][i])^2
            end
        end
        τ[n+i] = 2 / v_new_square
        
        #act HT on other columns in column block(2)
        if i > 1
            HT_2to2(L, τ, i, Bsub_factor, Bsub_factor_extra)
        end

        HT_2to1(L, τ, i, Asub_factor, Asub_factor_extra, Asup_factor, Asup_factor_extra, Bsub_factor, Bsub_factor_extra)
    end

    #copy v's in sub Block(1,2)
    L.Bsub[1][1][1:end] = Bsub_factor[1][1:end]
    L.Bsub[1][2][1:end] = Bsub_factor[2][1:end]
    L.Bsub_extra[1][1:end] = Bsub_factor_extra[1][1:end]
    L.Bsub_extra[2][1:end] = Bsub_factor_extra[2][1:end]

    #copy additional data into sub Block(1,1)
    L.Asub[1][1:end] = Asub_factor[1][1:end]
    L.Asub[2][1:end] = Asub_factor[2][1:end]
    L.Asub_extra[1][1:end] = Asub_factor_extra[1][1:end]
    L.Asub_extra[2][1:end] = Asub_factor_extra[2][1:end]

    #copy additional data into sup Block(1,1)
    L.Asup[1][1:end, 1] = Asup_factor[1][1:end]
    L.Asup[2][1:end, 1] = Asup_factor[2][1:end]
    L.Asup[1][1:end, 2] = Asup_factor_extra[1][1:end]
    L.Asup[2][1:end, 2] = Asup_factor_extra[2][1:end]
end

function HT_2to2(L::SemiseparableBBBArrowheadMatrix, τ::Vector{T}, i, Bsub_factor, Bsub_factor_extra) where T
    m,n = size(L.A)
    l = length(L.D)
    v = zeros(m)
    v[1:i+1] = L[1:i+1, n+i]#L[Block(1,2)][1:i+1, i]
    if i < l
        v[i+2:end] = Bsub_factor[1][i:end] * Bsub_factor[2][i] + Bsub_factor_extra[1][i:end] * Bsub_factor_extra[2][i]
    end
    coef = τ[m + i]

    # compute v'A[Block(1,2)]
    vTA_sub = zeros(i-1)
    vTA_dia = zeros(i-1)
    vTA_sup = zeros(i-1)
    sup_dotv= 0.0 # to compute sum(L.Bsup[1] .* v) iteratively 
    sub_dotv_1 = 0.0 # to compute sum(L.Bsub[1][1] .* v) iteratively
    sub_dotv_2 = 0.0 # to compute sum(L.Bsub_extra[1] .* v) iteratively
    for k in 1 : i-1
        tempvar1 = L.B[1][k, k] #L[Block(1,2)][k, k]
        tempvar2 = L.B[1][k+1, k] #L[Block(1,2)][k + 1, k]
        vTA_dia[k] = v[k] * tempvar1 + v[k + 1] * tempvar2
    end
    for k in 2 : i-1
        sup_dotv = sup_dotv + L.Bsup[1][k - 1] * v[k - 1]
        vTA_sup[k] = L.Bsup[2][k-1] * sup_dotv
    end
    for k in l-1 : -1 : 1
        sub_dotv_1 = sub_dotv_1 + L.Bsub[1][1][k] * v[k + 2]
        sub_dotv_2 = sub_dotv_2 + L.Bsub_extra[1][k] * v[k + 2]
        if k < i
            vTA_sub[k] = L.Bsub[1][2][k] * sub_dotv_1 + L.Bsub_extra[2][k] * sub_dotv_2
        end
    end
    vTA = vTA_sub + vTA_dia + vTA_sup

    #update the banded part in L[Block(1,2)]
    for k in 1 : i-1
        L.B[1][k, k] = L.B[1][k, k] - coef * v[k] * vTA[k]
        L.B[1][k + 1, k] = L.B[1][k + 1, k] - coef * v[k + 1] * vTA[k]
    end

    #update the sup part in L[Block(1,2)]
    for k in 2 : i-1
        L.Bsup[2][k-1] = L.Bsup[2][k-1] - coef * L.Bsup[2][i-1] * vTA[k]
    end

    #update L[Block(2,2)]
    if i == l
        L.A22sub[2][1:l-1] = vTA[1:l-1]
        L.A22sub[1][l-1] = -coef #That is a degree of freedom
    else
        L.A22sub[1][i-1] = -coef * vTA[1] / L.A22sub[2][1]
    end

    #update the sub part in L[Block(1,2)]
    if i == l
        L.Bsub_extra[2][1:l-1] = vTA[1:l-1]
        L.Bsub_extra[1][1:l-1] = -coef * v[3:end]
        Bsub_factor[1][1:end] = L.Bsub[1][1][1:end]
        Bsub_factor[2][1:end] = L.Bsub[1][2][1:end]
        Bsub_factor_extra[1][end] = L.Bsub_extra[1][end]
        Bsub_factor_extra[2][1:end] = L.Bsub_extra[2][1:end]
    else
        for j in 1:l-1
            L.Bsub_extra[1][j] = L.Bsub_extra[1][j] - coef * v[j+2] * vTA[1] / L.Bsub_extra[2][1]
        end
        L.Bsub_extra[2][i:end] .= 0
        # update Bsub_factor and Bsub_factor_extra
        for j in 1:i-1
            Bsub_factor[2][j] = Bsub_factor[2][j] - coef * Bsub_factor[2][i] * vTA[j]
            Bsub_factor_extra[2][j] = Bsub_factor_extra[2][j] - coef * Bsub_factor_extra[2][i] * vTA[j]
        end
        Bsub_factor_extra[1][i-1] = (L.Bsub[1][1][i-1] * L.Bsub[1][2][i-1] + L.Bsub_extra[1][i-1] * L.Bsub_extra[2][i-1] - Bsub_factor[1][i-1] * Bsub_factor[2][i-1]) / Bsub_factor_extra[2][i-1]
    end
end

function HT_2to1(L::SemiseparableBBBArrowheadMatrix, τ::Vector{T}, i, Asub_factor, Asub_factor_extra, Asup_factor, Asup_factor_extra, Bsub_factor, Bsub_factor_extra) where T
    m,n = size(L.A)
    l = length(L.D)
    v = zeros(m)
    v[1:i+1] = L[1:i+1, n+i]#L[Block(1,2)][1:i+1, i]
    if i < l
        v[i+2:end] = Bsub_factor[1][i:end] * Bsub_factor[2][i] + Bsub_factor_extra[1][i:end] * Bsub_factor_extra[2][i]
    end
    coef = τ[m + i]

    # compute v'A[Block(1,1)]
    vTA_sub = zeros(n)
    vTA_dia = zeros(n)
    vTA_sup = zeros(n)
    sup_dotv_1= 0.0 # to compute sum(L.Asup[1][1] .* v) iteratively 
    sup_dotv_2= 0.0 # to compute sum(L.Asup[2][1] .* v) iteratively 
    sup_dotv_3 = 0.0 # to compute sum(Asup_factor[1] .* v) iteratively
    sup_dotv_4 = 0.0  # to compute sum(Asup_factor_extra[1] .* v) iteratively
    sub_dotv_1 = 0.0 # to compute sum(L.Asub[1] .* v) iteratively
    sub_dotv_2 = 0.0 # to compute sum(L.Asub_extra[1] .* v) iteratively
    sub_dotv_3 = 0.0 # to compute sum(Asub_factor[1] .* v) iteratively
    sub_dotv_4 = 0.0 # to compute sum(Asub_factor_extra[1] .* v) iteratively

    for k in 1 : n
        vTA_dia[k] = v[k] * L.A[k, k]
        if k > 1
            vTA_dia[k] = vTA_dia[k] + v[k - 1] * L.A[k - 1, k]
        end
        if k < n
            vTA_dia[k] = vTA_dia[k] + v[k + 1] * L.A[k + 1, k]
        end
    end
    # compute vTA_sup
    for k in 3 : n
        if k <= i + 2
            sup_dotv_1 = sup_dotv_1 + L.Asup[1][k - 2, 1] * v[k - 2]
            sup_dotv_2 = sup_dotv_2 + L.Asup[1][k - 2, 2] * v[k - 2]
        end
        vTA_sup[k] = L.Asup[2][k-2, 1] * sup_dotv_1 + L.Asup[2][k-2, 2] * sup_dotv_2
    end

    for k in i + 3 : n
        sup_dotv_3 = sup_dotv_3 + Asup_factor[1][k - 2] * v[k - 2]
        sup_dotv_4 = sup_dotv_4 + Asup_factor_extra[1][k - 2] * v[k - 2]
        vTA_sup[k] = vTA_sup[k] + Asup_factor[2][k-2] * sup_dotv_3 + Asup_factor_extra[2][k-2] * sup_dotv_4
    end
    #compute vTA_sub
    for k in l-1 : -1 : 1
        sub_dotv_1 = sub_dotv_1 + L.Asub[1][k] * v[k + 2]
        sub_dotv_2 = sub_dotv_2 + L.Asub_extra[1][k] * v[k + 2]
        if k <= i
            vTA_sub[k] = L.Asub[2][k] * sub_dotv_1 + L.Asub_extra[2][k] * sub_dotv_2
        end
    end

    for k in l-1 : -1 : 1
        sub_dotv_3 = sub_dotv_3 + Asub_factor[1][k] * v[k + 2]
        sub_dotv_4 = sub_dotv_4 + Asub_factor_extra[1][k] * v[k + 2]
        if k > i
            vTA_sub[k] = Asub_factor[2][k] * sub_dotv_3 + Asub_factor_extra[2][k] * sub_dotv_4
        end
    end

    vTA = vTA_sub + vTA_dia + vTA_sup
    # There are elements in Block(2,1) that needs to be included in vTA:
    vTA[i] = vTA[i] + L.C[1][i, i]
    vTA[i + 1] = vTA[i + 1] + L.C[1][i, i + 1]


    #update the banded part in L[Block(1,2)]
    for k in 1 : n
        if k > 1
            L.A[k - 1, k] = L.A[k - 1, k] - coef * v[k - 1] * vTA[k]
        end
        L.A[k, k] = L.A[k, k] - coef * v[k] * vTA[k]
        if k < n
            L.A[k + 1, k] = L.A[k + 1, k] - coef * v[k + 1] * vTA[k]
        end
    end

    #update the banded part of L[Block(2,1)]
    L.C[1][i, i] = L.C[1][i, i] - coef * vTA[i]
    L.C[1][i, i + 1] = L.C[1][i, i + 1] - coef * vTA[i + 1]

    #update the sub part of L[Block(2,1)]
    if i == l
        L.Csub[2][1:l-1] = vTA[1:l-1]
        L.Csub[1][l-1] = -coef #That is a degree of freedom
    elseif i > 1
        L.Csub[1][i-1] = -coef * vTA[1] / L.Csub[2][1]
    end

    #update the sup part of L[Block(2,1)]
    if i == l - 1
        L.Csup[1][2][n-2] = -coef * vTA[n]
        L.Csup[1][1][l-1] = 1.0 #That is a degree of freedom
    elseif i < l
        L.Csup[1][1][i] = - coef * vTA[n] / L.Csup[1][2][n-2]
        L.Csup[1][2][i] = - coef * vTA[i + 2] / L.Csup[1][1][i]
    end

    #update the sub part of L[Block(1,1)]
    if i == l
        L.Asub_extra[2][1:l-1] = vTA[1:l-1]
        L.Asub_extra[1][1:l-1] = -coef * v[3:end]
        Asub_factor[1][end] = L.Bsub[1][1][end]
        Asub_factor[2][1:end] = L.Asub[2][1:end] * L.Asub[1][end] / Asub_factor[1][end]
        Asub_factor_extra[1][end] = Bsub_factor_extra[1][end]
        Asub_factor_extra[2][1:end] = L.Asub_extra[2][1:end] * L.Asub_extra[1][end] / Asub_factor_extra[1][end]
    else
        for j in 1:l-1
            L.Asub_extra[1][j] = L.Asub_extra[1][j] - coef * v[j+2] * vTA[1] / L.Asub_extra[2][1]
        end
        L.Asub_extra[2][i:end] .= 0
        L.Asub[2][i:end] .= 0
        Asub_factor[1][i] = Bsub_factor[1][i]
        Asub_factor_extra[1][i] = Bsub_factor_extra[1][i]
        # update Asub_factor and Asub_factor_extra
        for j in 1:l-1
            Asub_factor[2][j] = Asub_factor[2][j] - coef * Bsub_factor[2][i] * vTA[j]
            Asub_factor_extra[2][j] = Asub_factor_extra[2][j] - coef * Bsub_factor_extra[2][i] * vTA[j]
        end
        # display(Asub_factor[1] * Asub_factor[2]' + Asub_factor_extra[1] * Asub_factor_extra[2]')
    end
    
    #update the sup part of L[Block(1,1)]
    if i == l
        L.Asup[1][1:l-1, 2] = L.Bsup[1][1:l-1]
        L.Asup[2][1:l-1, 2] = -coef * L.Bsup[2][l-1] * vTA[3:end]
        Asup_factor[1][1:l-1] = L.Asup[1][1:l-1, 1] * L.Asup[2][l-1, 1] + L.Asup[1][1:l-1, 2] * L.Asup[2][l-1, 2]
        Asup_factor[2][l-1] = 1.0 # this is a degree of freedom
    else
        A_i_ip2 = L.Asup[1][i, 1] * L.Asup[2][i, 1] + L.Asup[1][i, 2] * L.Asup[2][i, 2] - coef * v[i] * vTA[i+2] #used to update Asup_factor[2]
        for j in 1 : l-1
            L.Asup[2][j, 2] = L.Asup[2][j, 2] - coef * v[1] * vTA[j+2] / L.Asup[1][1, 2]
        end
        L.Asup[1][i:end, 2] .= 0 
        L.Asup[1][i:end, 1] .= 0 

        if i == l-1
            Asup_factor_extra[2][l-1] = -coef * vTA[n]
            Asup_factor_extra[1][1:l-1] = v[1:l-1] # this is a degree of freedom
        else
            for j in 1 : l-1
                Asup_factor_extra[1][j] = Asup_factor_extra[1][j] - coef * v[j] * vTA[end] / Asup_factor_extra[2][end]
            end
            Asup_factor_extra[2][i] = Asup_factor_extra[2][l-1] * vTA[i+2] / vTA[l+1]
            Asup_factor[2][i] = (A_i_ip2 - Asup_factor_extra[2][i] * Asup_factor_extra[1][i]) / Asup_factor[1][i]
        end

        #display(Asup_factor[1] * Asup_factor[2]' + Asup_factor_extra[1] * Asup_factor_extra[2]')
    end
end

function HT_column_block_1(L::SemiseparableBBBArrowheadMatrix, τ::Vector{T}) where T
    m,n = size(L.A)
    l = length(L.D)
    m2, n2 = size(L.D[1])

    # record the final result of sub Block(1,1) which is rank 1
    Asub_final = (zeros(m - 2), zeros(l - 1))

    # record additional sub Block(1,1) info during the HT process
    Asub_extra2 = (zeros(m - 2), zeros(l - 1))


    for i in n:-1:2
        v_square = 0
        if i > 2
            for j in 1:i-2
                v_square = v_square + (L.Asup[1][j,1] * L.Asup[2][i-2, 1] + L.Asup[1][j,2] * L.Asup[2][i-2, 2])^2
            end
        end
        v_square = v_square + L.A[i, i]^2 + L.A[i-1, i]^2
        v_last = L.A[i,i] + sign(L.A[i,i]) * sqrt(v_square)
        L.A[i,i] = -sign(L.A[i,i]) * sqrt(v_square)

        #record the info of v
        if i > 2
            L.Asup[2][i-2, 1] = L.Asup[2][i-2, 1] / v_last
            L.Asup[2][i-2, 2] = L.Asup[2][i-2, 2] / v_last
        end
        L.A[i-1,i] = L.A[i-1,i] / v_last

        v_new_square = L.A[i-1, i]^2 + 1^2
        if i > 2
            for j in 1:i-2
                v_new_square = v_new_square + (L.Asup[1][j,1] * L.Asup[2][i-2, 1] + L.Asup[1][j,2] * L.Asup[2][i-2, 2])^2
            end
        end
        τ[i] = 2 / v_new_square

        HT_1to1(L, τ, i, Asub_final, Asub_extra2)
    end

    L.Asub[1][1:end] = Asub_final[1][1:end]
    L.Asub[2][1:end] = Asub_final[2][1:end]
    L.Asub_extra[1][1:end] .= 0
    L.Asub_extra[2][1:end] .= 0

end

function HT_1to1(L::SemiseparableBBBArrowheadMatrix, τ::Vector{T}, i, Asub_final, Asub_extra2) where T
    m,n = size(L.A)
    l = length(L.D)
    v = zeros(i)
    v[1:i-1] = L[1:i-1, i] #L[Block(1,1)][1:i-1, i]
    v[end] = 1
    coef = τ[i]

    # compute v'A[Block(1,1)]
    vTA_sub = zeros(i-1)
    vTA_dia = zeros(i-1)
    vTA_sup = zeros(i-1)
    sup_dotv_1 = 0.0 # to compute sum(L.Asup[1][:,1] .* v) iteratively
    sup_dotv_2 = 0.0 # to compute sum(L.Asup[1][:,2] .* v) iteratively  
    sub_dotv_1 = 0.0 # to compute sum(L.Asub[1] .* v) iteratively
    sub_dotv_2 = 0.0 # to compute sum(L.Asub_extra[1] .* v) iteratively
    sub_dotv_3 = 0.0 # to compute sum(Asub_extra2[1] .* v) iteratively
    for k in 1 : i-1
        vTA_dia[k] = v[k] * L.A[k, k]
        if k > 1
            vTA_dia[k] = vTA_dia[k] + v[k - 1] * L.A[k - 1, k]
        end
        vTA_dia[k] = vTA_dia[k] + v[k + 1] * L.A[k + 1, k]
    end
    for k in 3 : i-1
        sup_dotv_1 = sup_dotv_1 + L.Asup[1][k - 2, 1] * v[k - 2]
        sup_dotv_2 = sup_dotv_2 + L.Asup[1][k - 2, 2] * v[k - 2]
        vTA_sup[k] = L.Asup[2][k-2, 1] * sup_dotv_1 + L.Asup[2][k-2, 2] * sup_dotv_2
    end
    for k in i-2 : -1 : 1
        sub_dotv_1 = sub_dotv_1 + L.Asub[1][k] * v[k + 2]
        sub_dotv_2 = sub_dotv_2 + L.Asub_extra[1][k] * v[k + 2]
        sub_dotv_3 = sub_dotv_3 + Asub_extra2[1][k] * v[k + 2]
        if k < i
            vTA_sub[k] = L.Asub[2][k] * sub_dotv_1 + L.Asub_extra[2][k] * sub_dotv_2 + Asub_extra2[2][k] * sub_dotv_3
        end
    end
    vTA = vTA_sub + vTA_dia + vTA_sup

    #update the banded part in L[Block(1,1)]
    for k in 1 : i-1
        if k > 1
            L.A[k - 1, k] = L.A[k - 1, k] - coef * v[k - 1] * vTA[k]
        end
        L.A[k, k] = L.A[k, k] - coef * v[k] * vTA[k]
        L.A[k + 1, k] = L.A[k + 1, k] - coef * v[k + 1] * vTA[k]
    end

    #update the sup part in L[Block(1,1)]
    for k in 3 : i-1
        L.Asup[2][k-2, 1] = L.Asup[2][k-2, 1] - coef * L.Asup[2][i-2, 1] * vTA[k]
        L.Asup[2][k-2, 2] = L.Asup[2][k-2, 2] - coef * L.Asup[2][i-2, 2] * vTA[k]
    end

    #update the sub part in L[Block(1,1)]
    if i == n
        for j in 1 : n-2
            Asub_final[2][j] = L.Asub[1][end] * L.Asub[2][j] + L.Asub_extra[1][end] * L.Asub_extra[2][j] - coef * v[end] * vTA[j]
        end
        Asub_final[1][end] = 1.0
        
        Asub_extra2[2][1:end] = Asub_final[2][1:end]
        L.Asub[1][1:end-1] = L.Asub[1][1:end-1] - L.Asub[1][end]*v[3:end-1]
        L.Asub_extra[1][1:end-1] = L.Asub_extra[1][1:end-1] - L.Asub_extra[1][end]*v[3:end-1]
        Asub_extra2[1][1:end-1] = v[3:end-1]

        L.Asub[1][end] = 0
        L.Asub_extra[1][end] = 0
        Asub_extra2[1][end] = 0
    elseif i > 2
        Asub_final[1][i-2] = (L.Asub[1][i-2] * L.Asub[2][1] + L.Asub_extra[1][i-2] * L.Asub_extra[2][1] + Asub_extra2[1][i-2] * Asub_extra2[2][1] - coef * vTA[1]) / Asub_final[2][1]
        #display(Asub_final[1] * Asub_final[2]')
        if i > 3
            L.Asub[1][1:i-3] = L.Asub[1][1:i-3] - L.Asub[1][i-2] * v[3:i-1]
            L.Asub_extra[1][1:i-3] = L.Asub_extra[1][1:i-3] - L.Asub_extra[1][i-2] * v[3:i-1]
            Asub_extra2[1][1:i-3] = Asub_extra2[1][1:i-3] - (Asub_extra2[1][i-2] - Asub_final[1][i-2]) * v[3:i-1]

            L.Asub[1][i-2:end] .= 0
            L.Asub_extra[1][i-2:end] .= 0
            Asub_extra2[1][i-2:end] .= 0
        end
        #display(L.Asub[1] * L.Asub[2]' + L.Asub_extra[1] * L.Asub_extra[2]' + Asub_extra2[1] * Asub_extra2[2]')
    end
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
    Asub_extra = (zeros(m - 2), zeros(n - 2))
    Bsup = (zeros(m - 2), zeros(l - 1))
    Bsub = ((zeros(m - 2), zeros(l - 1)), (zeros(m - 2), zeros(l - 1)))
    Bsub_extra = (zeros(m - 2), zeros(l - 1))

    Csup = ((zeros(l - 1), zeros(n - 2)), (zeros(l - 1), zeros(n - 2)))
    Csub = (zeros(l - 1), zeros(n - 2))

    A22sub = (zeros(l - 1), zeros(l - 1))
    A32sup = (zeros(l - 1), zeros(l - 1))

    A31extra = zeros(l - 1)
    A32extra = zeros(l - 1)
    A33extra = zeros(l - 1)

    SemiseparableBBBArrowheadMatrix(A, B, C, Array{BandedMatrix{Float64}, 1}(D), Asub, Asup, Asub_extra, Bsup, Bsub, Bsub_extra, Csup, Csub, A22sub, A32sup, A31extra, A32extra, A33extra)
end

end