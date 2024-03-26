"""
    kronecker_quasirand(d, N, start=0)

Return an `d`-by-`N` array of `N` quasi-random samples in [0, 1]^d generated
by an additive quasi-random low-discrepancy sample sequence.
"""
function kronecker_quasirand(d, N, start=0)
    
    # Compute the recommended constants ("generalized golden ratio")
    # See: http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    ϕ = 1.0+1.0/d
    for k = 1:10
        gϕ = ϕ^(d+1)-ϕ-1
        dgϕ= (d+1)*ϕ^d-1
        ϕ -= gϕ/dgϕ
    end
    αs = [mod(1.0/ϕ^j, 1.0) for j=1:d]
    
    # Compute the quasi-random sequence
    Z = zeros(d, N)
    for j = 1:N
        for i=1:d
            Z[i,j] = mod(0.5 + (start+j)*αs[i], 1.0)
        end
    end
    
    Z
end


function bkronecker_quasirand(d, N, bounds, start=0)
    db, _ = size(bounds) 
    Z = kronecker_quasirand(d, N, start)
    
    for j = 1:d
        r = db == d ? 0 : j-1
        width = bounds[j-r,2] - bounds[j-r,1]
        shift = bounds[j-r][1]
        Z[j,:] = (Z[j,:] .* width) .+ shift
    end
    
    Z
end