
const DeparalleliseTol = 1.0e-12

function _istwocolumnparallel(cola, colb, tol::Real)
    (length(cola) != length(colb)) && throw(DimensionMismatch())
    (length(cola) == 0) && throw(ArgumentError("the column should not be empty."))
    n = length(cola)
    idx = findall(x->abs(x)>tol, cola)
    (length(idx) == 0) && throw(ArgumentError("the column can not be all zeros."))
    factor = colb[idx[1]]/cola[idx[1]]
    dif = colb - factor*cola
    for i in dif
        if abs(i) > tol
        	return false, 0.
        end
    end
    return true, factor
end

function _getridofzerocol(m::AbstractArray{T, 2}, tol::Real, verbosity::Int=0) where {T}
    s1, s2 = size(m)
    zerocols = Vector{Int}(undef, 0)
    for j = 1:s2
    	allzero = true
    	for i=1:s1
    	    if abs(m[i, j]) > tol
    	        allzero = false
    	        break
    	    end
    	end
    	if allzero
    	    (verbosity > 3) && println("all elements of column $j are zero.")
    	    Base.push!(zerocols, j)
    	end
    end
    ns = s2 - length(zerocols)
    (ns == 0 && verbosity > 3) && println("all the columns are zero.")
	mout = zeros(T, (s1, ns))
	j = 1
	for i=1:s2
		if !(i in zerocols)
		    mout[:, j] .= view(m, :, i)
		    j += 1
	    end
	end
	return mout, zerocols
end


function _matrixdeparallelisenozerocols(m::AbstractArray{T, 2}, tol::Real, verbosity::Int=0) where {T}
    s1, s2 = size(m)
    K = []
    Tm = zeros(T, (s2, s2))
    for j = 1:s2
    	exist = false
    	for i=1:length(K)
    	    p, factor = _istwocolumnparallel(K[i], m[:, j], tol)
    	    if p
    	       (verbosity > 3) && println("column $i is in parallel with column $j.")
    	       Tm[i, j] = factor
    	       exist = true
    	       break
    	    end
    	end
    	if !exist
    	    Base.push!(K, m[:, j])
    	    nK = length(K)
    	    Tm[nK, j] = 1
    	end
    end
    nK = length(K)
    M = zeros(T, (s1, nK))
    for j=1:nK
        M[:, j] = K[j]
    end
    return M, Tm[1:nK, :]
end

function matrixdeparlise_col(m::AbstractArray{T, 2}, tol::Real; verbosity::Int=0) where {T}
    mnew, zerocols = _getridofzerocol(m, tol, verbosity)
    M, Tm = _matrixdeparallelisenozerocols(mnew, tol, verbosity)
    # isapprox(mnew, M*Tm) || error("matrixdeparallise error.")
    if isempty(M)
        (verbosity > 3) && println("all the elements of the matrix M are 0.")
        return M, Tm
    end
    Tnew = zeros(T, (size(Tm, 1), size(m, 2)))
    j = 1
    for i = 1:size(Tnew, 2)
        if !(i in zerocols)
            Tnew[:, i] .= Tm[:, j]
            j += 1
        end
    end
    # println("dim $(size(m, 2)) -> $(size(M, 2))")
    # isapprox(m, M*Tnew) || error("matrixdeparallise error.")
    return M, Tnew
end

function matrixdeparlise_row(m::AbstractMatrix, tol::Real; verbosity::Int=0)
    a, b = matrixdeparlise_col(transpose(m), tol, verbosity=verbosity)
    return transpose(b), transpose(a)
end

# matrixdeparlise(m::AbstractMatrix, row::Bool, tol::Real; verbosity::Int=0) = row ? matrixdeparlise_row(m, tol, verbosity=verbosity) : matrixdeparlise_col(m, tol, verbosity=verbosity)

"""
    deparallelise_util(a::AbstractArray{T, N}, axs::Tuple{NTuple{N1, Int}, NTuple{N2, Int}}, tol::Real=1.0e-12; verbose::Int=0) where {T, N, N1, N2}
Deparallelisation of QTensor a, by joining axs to be the second dimension
"""
function deparallelise_util(a::AbstractArray{T, N}, left::NTuple{N1, Int}, right::NTuple{N2, Int}; row_or_col::Symbol=:row,
    tol::Real=DeparalleliseTol, verbosity::Int=0) where {T, N, N1, N2}
    ((row_or_col == :row) || (row_or_col == :col)) || throw(ArgumentError("row_or_col must be row or col"))
    (N == N1+N2) || throw(DimensionMismatch())
    dim = (left..., right...)
    b = permute(a, dim)
    shape_b = size(b)
    ushape = shape_b[1:N1]
    vshape = shape_b[(N1+1):end]
    s1 = prod(ushape)
    s2 = prod(vshape)
    b = reshape(b, s1, s2)
    if row_or_col == :row
        u, v = matrixdeparlise_row(b, tol, verbosity=verbosity)
    else
        u, v = matrixdeparlise_col(b, tol, verbosity=verbosity)
    end
    (size(u,2)!=size(v,1)) && error("matrix deparallelise error.")
    md = size(u, 2)
    return reshape(u, ushape..., md), reshape(v, md, vshape...)
end

left_deparallelise(a::AbstractArray, left::Tuple, right::Tuple; kwargs...) = deparallelise_util(a, left, right; row_or_col=:col, kwargs...)
right_deparallelise(a::AbstractArray, left::Tuple, right::Tuple; kwargs...) = deparallelise_util(a, left, right; row_or_col=:row, kwargs...)



