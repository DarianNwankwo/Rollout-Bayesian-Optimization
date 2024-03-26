using LinearAlgebra

"""
The PreallocatedSymmetricMatrix concerns itself with iteratively updating a covariance matrix and
the associated Cholesky factorization.

Let's concern ourselves with matrices, particularly covariance matrices. Suppose we have a data generating process that produces \$m\$ samples with no gradient observations and \$h+1\$ samples with gradient observations of \$d\$ dimensions. Let's assign symbols to the aforementioned.

If we consider the problem of computing mixed covariances, it proves useful to distinguish the covariance measures into several categorizations:
- A will denote the covariances between function values strictly.
- B will denote the covariances between fantasized function values and known samples.
- C will denote the covariances between fantasized function values against themselves.
- D will denote the covariances between fantasized gradients and known function values.
- E will denote the covariances between fantasized gradients against fantasized function values.
- G will denote the covariances between fantasized gradients against themselves.
"""
struct PreallocatedSymmetricMatrix{T <: Number}
    K::Base.RefValue{Matrix{T}}
    fK::Base.RefValue{LowerTriangular{T, Matrix{T}}}
    active_rows_cols::Vector{Int}
    m::Int # Number of function observations
    h::Int # Number of function and gradient observations
    d::Int # Dimensionality of feature vector
end

function PreallocatedSymmetricMatrix{T}(m::Int, h::Int, d::Int) where T <: Number
    n = m + (h + 1)*(d + 1)
    K = zeros(n, n)
    fK = LowerTriangular(zeros(n, n))
    active_rows_cols = Int[]
    
    return PreallocatedSymmetricMatrix{T}(Ref(K), Ref(fK), active_rows_cols, m, h, d)
end

function clear_fantasized!(PSM::PreallocatedSymmetricMatrix)
    m = PSM.m
    PSM.K[][m+1:end, :] .= 0
    PSM.K[][:, m+1:end] .= 0
    
    PSM.fK[][m+1:end, :] .= 0
    PSM.fK[][:, m+1:end] .= 0
    
    nothing
end

"""
We'll use the notation in function definitions of F to denote function observation covariances
and G to denote gradient observation covariances.
"""

# Subroutine 1: Covariance Matrix (A)
function update_knowns!(PSM::PreallocatedSymmetricMatrix, Kupdate)
    @assert size(Kupdate, 1) == PSM.m "Covariance matrix dimension is incorrect" 
    PSM.K[][1:PSM.m, 1:PSM.m] = Kupdate
    
    [push!(PSM.active_rows_cols, rc) for rc in 1:PSM.m]
    
    nothing
end

# Subroutine 1: Cholesky Factorization (L11)
function cholesky_update_knowns!(PSM::PreallocatedSymmetricMatrix, fKUpdate)
    @assert size(Kupdate, 1) == PSM.m "Cholesky factorization dimension is incorrect"
    PSM.fK[][1:PSM.m, 1:PSM.m] = fKupdate
    
    nothing
end

# Subroutine 2: Covariance Matrix (B)
function update_fantasized_vs_knowns!(PSM::PreallocatedSymmetricMatrix, Kvec_update, row_ndx)
    @assert length(Kvec_update) == PSM.m "Covariance vector length != PSM.m (m = $(PSM.m))"
    
    PSM.K[][row_ndx, 1:m] = Kvec_update
    PSM.K[][1:m, row_ndx] = Kvec_update
        
    nothing
end

# Subroutine 2: Cholesky Factorization (L21)
function cholesky_update_fantasized_vs_knowns!(PSM::PreallocatedSymmetricMatrix, row_ndx)
    m = PSM.m
    
    B = @view PSM.K[][row_ndx:row_ndx, 1:m]
    L11 = @view PSM.fK[][1:m, 1:m]
    PSM.fK[][row_ndx:row_ndx, 1:m] .= B / L11' # L21
        
    nothing
end

# Subroutine 3: Covariance Matrix (C)
function update_fantasized_vs_fantasized!(PSM::PreallocatedSymmetricMatrix, Kupdate, row_ndx)
    @assert (row_ndx - PSM.m) == size(Kupdate, 2)
    ustride = PSM.m+1:row_ndx
    
    PSM.K[][ustride, ustride] = Kupdate
    push!(PSM.active_rows_cols, row_ndx)
    
    nothing
end

# Subroutine 3: Cholesky Factorization (L22)
function cholesky_update_fantasized_vs_fantasized!(PSM::PreallocatedSymmetricMatrix, row_ndx)
    @assert (row_ndx - PSM.m) == size(Kupdate, 2)
    ustride = PSM.m+1:row_ndx
    
    C = @view PSM.K[][ustride, ustride]
    L21 = @view PSM.fK[][PSM.m+1:row_ndx, 1:m]
    PSM.fK[][ustride, ustride] .= cholesky(C - L21*L21').L # L22
    
    nothing
end

# Subroutine 4+5: Covariance Matrix (D)
# function update_gradfantasized_vs_fantasized_and_known!(PSM::PreallocatedSymmetricMatrix, Kupdate, grad_ndx)
#     m, h, d = PSM.m, PSM.h, PSM.d
#     grad_start = m+h+2
#     num_cols = PSM.active_rows_cols[end]
#     row_ustride = grad_start+(grad_ndx - 1)*d:grad_start+grad_ndx*d-1
#     col_ustride = 1:num_cols

#     PSM.K[][row_ustride, col_ustride] = Kupdate
#     PSM.K[][col_ustride, row_ustride] = Kupdate'
    
#     nothing
# end

# Subroutine 4: Covariance Matrix (D)
function update_gradfantasized_vs_known!(PSM::PreallocatedSymmetricMatrix, Kupdate, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    grad_start = m+h+2
    row_ustride = grad_start+(grad_ndx - 1)*d:grad_start+grad_ndx*d-1
    col_ustride = 1:m
    
    PSM.K[][row_ustride, col_ustride] = Kupdate
    PSM.K[][col_ustride, row_ustride] = Kupdate'
    
    nothing
end

# Subroutine 4: Cholesky Factorization (L31)
function cholesky_update_gradfantasized_vs_known!(PSM::PreallocatedSymmetricMatrix, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    grad_start = m+h+2
    row_ustride = grad_start+(grad_ndx - 1)*d:grad_start+grad_ndx*d-1
    col_ustride = 1:m
    
    D = @view PSM.K[][row_ustride, col_ustride]
    L11 = @view PSM.fK[][1:m, 1:m]
    PSM.fK[][row_ustride, col_ustride] .= D / L11' # L31
    
    nothing
end

# Subroutine 5: Covariance Matrix (E)
function update_gradfantasized_vs_fantasized!(PSM::PreallocatedSymmetricMatrix, Kupdate, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    grad_start = m+h+2
    num_cols = PSM.active_rows_cols[end]
    row_ustride = grad_start+(grad_ndx - 1)*d:grad_start+grad_ndx*d-1
    col_ustride = m+1:m+grad_ndx
    
    PSM.K[][row_ustride, col_ustride] = Kupdate
    PSM.K[][col_ustride, row_ustride] = Kupdate'
    
    nothing
end

# Subroutine 5: Cholesky Factorization (L32)
function cholesky_update_gradfantasized_vs_fantasized!(PSM::PreallocatedSymmetricMatrix, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    grad_start = m+h+2
    grad_cur = grad_start+(grad_ndx - 1)*d
    grad_cur_end = grad_cur + d - 1
    row_ustride = grad_start+(grad_ndx - 1)*d:grad_start+grad_ndx*d-1
    col_ustride = m+1:m+grad_ndx
    # println("5. Row Stride: $(row_ustride) -- Col Stride: $(col_ustride)")
    
    E = @view PSM.K[][row_ustride, col_ustride]
    L21 = @view PSM.fK[][m+1:m+grad_ndx, 1:m]
    L31 = @view PSM.fK[][grad_cur:grad_cur_end, 1:m]
    L22 = @view PSM.fK[][m+1:m+grad_ndx, col_ustride]
    # println("5. E = $(size(E)), L21 = $(size(L21)), L31 = $(size(L31)), L22 = $(size(L22))")

    PSM.fK[][row_ustride, col_ustride] .= (E - L31*L21') / L22' # L32
    
    nothing
end

# Subroutine 6: Covariance Matrix (G)
function update_gradfantasized_vs_self!(PSM::PreallocatedSymmetricMatrix, Kupdate, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    grad_start = m+h+2
    ustride = grad_start+(grad_ndx - 1)*d:grad_start+grad_ndx*d-1
    
    PSM.K[][ustride, ustride] = Kupdate
    
    nothing
end

# Subroutine 7: Covariance Matrix (G)
function update_gradfantasized_vs_gradfantasized!(PSM::PreallocatedSymmetricMatrix, Kupdate, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    if grad_ndx == 1 return nothing end
    
    grad_start = m+h+2
    row_ustride = grad_start+(grad_ndx - 1)*d:grad_start+grad_ndx*d-1
    col_ustride = grad_start:grad_start+(grad_ndx - 1)*d-1
    
    PSM.K[][row_ustride, col_ustride] = Kupdate
    PSM.K[][col_ustride, row_ustride] = Kupdate'
    
    nothing
end

# Subroutine 8: Covariance Matrix (E)
function update_fantasized_vs_allprev_gradfantasized!(PSM::PreallocatedSymmetricMatrix, Kupdate, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    if grad_ndx == 1 return nothing end
    
    grad_start = m+h+2
    row_ustride = grad_start:grad_start+(grad_ndx - 1)*d-1
    col_num = PSM.active_rows_cols[end]
    col_ustride = col_num:col_num
    
    # println("Subroutine: update_fantasized_vs_allprev_gradfantasized")
    # println("Column Update Stride: $col_ustride\nRow Update Stride: $row_ustride")
    
    PSM.K[][row_ustride, col_ustride] = Kupdate
    PSM.K[][col_ustride, row_ustride] = Kupdate'
    
    nothing
end

# # Subroutine 8: Cholesky Factorization (L32 top right portion)
# function cholesky_update_fantasized_vs_allprev_gradfantasized!(PSM::PreallocatedSymmetricMatrix, grad_ndx)
#     m, h, d = PSM.m, PSM.h, PSM.d
#     if grad_ndx == 1 return nothing end
    
#     grad_start = m+h+2
#     grad_cur = grad_start+(grad_ndx - 1)*d
#     grad_cur_end = grad_cur + d - 1
#     row_ustride = grad_start:grad_start+(grad_ndx - 1)*d-1
#     col_num = PSM.active_rows_cols[end]
#     col_ustride = col_num:col_num
    
#     E = @view PSM.K[][row_ustride, col_ustride] # sure
#     println("Row Stride: $row_ustride -- Col Stride: $col_ustride")
#     L21 = @view PSM.fK[][m+1:m+grad_ndx-1, 1:m]
#     L31 = @view PSM.fK[][grad_cur:grad_cur_end, 1:m]
#     L22 = @view PSM.fK[][col_ustride, col_ustride] # sure
#     println("E = $(size(E)), L21 = $(size(L21)), L31 = $(size(L31)), L22 = $(size(L22))")
#     PSM.fK[][row_ustride, col_ustride] .= (E - L31*L21') / L22' # L32
    
#     nothing
# end

# Subroutine 5+8: Cholesky Factorization (L32 total)
function cholesky_update_gradfantasized_vs_fantasized!(PSM::PreallocatedSymmetricMatrix, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    
    grad_start = m+h+2
    row_ustride = grad_start:grad_start+grad_ndx*d-1
    col_ustride = m+1:m+grad_ndx

    E = @view PSM.K[][row_ustride, col_ustride]
    L31 = @view PSM.fK[][row_ustride, 1:m]
    L21 = @view PSM.fK[][col_ustride, 1:m]
    L22 = @view PSM.fK[][col_ustride, col_ustride]
    # println("Row Stride: $row_ustride -- Col Stride: $col_ustride")
    # println("E = $(size(E)), L21 = $(size(L21)), L31 = $(size(L31)), L22 = $(size(L22))")
    PSM.fK[][row_ustride, col_ustride] .= (E - L31*L21') / L22' # L32

    nothing
end

# Subroutine 6+7: Cholesky Factorization (L33 total)
function cholesky_update_gradfantasized_vs_gradfantasized!(PSM::PreallocatedSymmetricMatrix, grad_ndx)
    m, h, d = PSM.m, PSM.h, PSM.d
    grad_start = m+h+2
    ustride = grad_start:grad_start+grad_ndx*d-1
    
    G = @view PSM.K[][ustride, ustride]
    L31 = @view PSM.fK[][ustride, 1:m]
    L32 = @view PSM.fK[][ustride, m+1:m+grad_ndx]
    PSM.fK[][ustride, ustride] .= cholesky(G - L31*L31' - L32*L32').L

    nothing
end


function get_KXX(PSM::PreallocatedSymmetricMatrix, row_ndx)
    return @view PSM.K[][1:row_ndx, 1:row_ndx]
end