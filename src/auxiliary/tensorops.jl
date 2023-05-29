using LinearAlgebra: BlasFloat, LAPACK
using LinearAlgebra.BLAS: gemm, gemm!


scalar(x::AbstractArray{T}) where {T<:Number} = only(x)

permute(m::AbstractArray, perm) = PermutedDimsArray(m, perm)
permute(m::AbstractArray, left, right) = permute(m, (left..., right...))

function random_hermitian(::Type{T}, n::Int) where {T <: Number}
	m = randn(T, n, n)
	return m + m'
end

random_unitary(::Type{T}, n::Int) where {T <: Number} = exp(im .* random_hermitian(T, n))

function _eye(::Type{T}, m::Int, n::Int) where {T<:Number}
	r = zeros(T, m, n)
	for i in 1:min(m, n)
		r[i, i] = 1
	end
	return r
end
_eye(::Type{T}, d::Int) where {T<:Number} = _eye(T, d, d)
_eye(d::Int) = _eye(Float64, d)

diag(m::AbstractVector{<: Number}) = LinearAlgebra.diagm(0=>m)

coerce_scalar_type(::Type{T}, ::Type{S}) where {T <: Union{Real, Complex}, S <: Real} = T
coerce_scalar_type(::Type{T}, ::Type{S}) where {T <: Complex, S <: Complex} = T
coerce_scalar_type(::Type{T}, ::Type{S}) where {T <: Real, S <: Complex} = Complex{real(S)}


"""	
	move_selected_index_forward(a, I)
	move the indexes specified by I to the front of a
	# Arguments
	@ a::NTuple{N, Int}: the input tensor.
	@ I: tuple or vector of integer.
"""
function move_selected_index_forward(a::Vector{T}, I) where {T}
    na = length(a)
    nI = length(I)
    b = Vector{T}(undef, na)
    k1 = 0
    k2 = nI
    for i=1:na
        s = 0
        while s != nI
        	if i == I[s+1]
        		b[s+1] = a[k1+1]
        	    k1 += 1
        	    break
        	end
        	s += 1
        end
        if s == nI
        	b[k2+1]=a[k1+1]
        	k1 += 1
            k2 += 1
        end
    end
    return b
end

function move_selected_index_forward(a::NTuple{N, T}, I) where {N, T}
    return NTuple{N, T}(move_selected_index_forward([a...], I))
end

"""	
	move_selected_index_backward(a, I)
	move the indexes specified by I to the back of a
	# Arguments
	@ a::NTuple{N, Int}: the input tensor.
	@ I: tuple or vector of integer.
"""
function move_selected_index_backward(a::Vector{T}, I) where {T}
	na = length(a)
	nI = length(I)
	nr = na - nI
	b = Vector{T}(undef, na)
	k1 = 0
	k2 = 0
	for i = 1:na
	    s = 0
	    while s != nI
	    	if i == I[s+1]
	    		b[nr+s+1] = a[k1+1]
	    		k1 += 1
	    		break
	    	end
	    	s += 1
	    end
	    if s == nI
	        b[k2+1] = a[k1+1]
	        k2 += 1
	        k1 += 1
	    end
	end
	return b
end

function move_selected_index_backward(a::NTuple{N, T}, I) where {N, T}
	return NTuple{N, T}(move_selected_index_backward([a...], I))
end

function _group_extent(extent::NTuple{N, Int}, idx::NTuple{N1, Int}) where {N, N1}
    ext = Vector{Int}(undef, N1)
    l = 0
    for i=1:N1
        ext[i] = prod(extent[(l+1):(l+idx[i])])
        l += idx[i]
    end
    return NTuple{N1, Int}(ext)
end


function tie(a::AbstractArray{T, N}, axs::NTuple{N1, Int}) where {T, N, N1}
    (sum(axs) != N) && error("total number of axes should equal to tensor rank.")
    return reshape(a, _group_extent(size(a), axs))
end

function contract(a::AbstractArray{Ta, Na}, b::AbstractArray{Tb, Nb}, axs::Tuple{NTuple{N, Int}, NTuple{N, Int}}) where {Ta, Na, Tb, Nb, N}
    ia, ib = axs
    seqindex_a = move_selected_index_backward(collect(1:Na), ia)
    seqindex_b = move_selected_index_forward(collect(1:Nb), ib)
    ap = permute(a, seqindex_a)
    bp = permute(b, seqindex_b)
    return reshape(tie(ap, (Na-N, N)) * tie(bp, (N, Nb-N)), size(ap)[1:(Na-N)]..., size(bp)[(N+1):Nb]...)
end


function Base.kron(a::AbstractArray{Ta, N}, b::AbstractArray{Tb, N}) where {Ta<:Number, Tb<:Number, N}
    N == 0 && error("empty tensors.")
    sa = size(a)
    sb = size(b)
    sc = Tuple(sa[i]*sb[i] for i=1:N)
    c = Array{promote_type(Ta, Tb), N}(undef, sc)
    ranges = Vector{UnitRange{Int}}(undef, N)
    for index in CartesianIndices(a)
        # ranges[1] = (index[1]*sb[1]+1):(index[1]+1)*sb[1]
        for j = 1:N
            ranges[j] = ((index[j]-1)*sb[j]+1):(index[j]*sb[j])
        end
        c[ranges...] = a[index]*b
    end
    return c
end

function stable_svd!(a::StridedArray{T, 2}, workspace::AbstractVector{T}) where T
	if length(workspace) < length(a)
		resize!(workspace, length(a))
	end
	ac = reshape(view(workspace, 1:length(a)), size(a))
	copyto!(ac, a)
    try
        return LAPACK.gesdd!('S', ac)
    catch
        return LAPACK.gesvd!('S', 'S', a)
    end
end

function tsvd!(a::StridedArray{T, 2}, workspace::AbstractVector{T}=similar(a, length(a)); trunc::TruncationScheme=NoTruncation()) where {T}
	u, s, v = stable_svd!(a, workspace)
	d_old = length(s)
	s, err = _truncate!(s, trunc)
	d = length(s)
	if d == d_old
		return u, s, v, err
	else
		return u[:, 1:d], s, v[1:d, :], err
	end
end

function tsvd!(a::StridedArray{T, N}, left::NTuple{N1, Int}, right::NTuple{N2, Int}, workspace::AbstractVector{T}=similar(a, length(a)); 
	trunc::TruncationScheme=NoTruncation()) where {T <: Number, N, N1, N2}
	(N == N1 + N2) || throw(DimensionMismatch())
	if length(workspace) <= length(a)
		resize!(workspace, length(a))
	end
    dim = (left..., right...)
    b = permute(a, dim)
    shape_b = size(b)
    ushape = shape_b[1:N1]
    vshape = shape_b[(N1+1):end]
    s1 = prod(ushape)
    s2 = prod(vshape)
    # u, s, v = F.U, F.S, F.Vt
    bmat = copyto!(reshape(view(workspace, 1:length(a)), s1, s2), reshape(b, s1, s2))
    u, s, v, err = tsvd!(bmat, reshape(a, length(a)), trunc=trunc)
    md = length(s)
    return reshape(u, (ushape..., md)), s, reshape(v, (md, vshape...)), err
end

function tqr!(a::StridedMatrix) 
    q, r = LinearAlgebra.qr!(a)
    return Matrix(q), r
end

function tlq!(a::StridedMatrix) 
    l, q = LinearAlgebra.lq!(a)
    return l, Matrix(q)
end

"""
    qr(a::AbstractArray{T, N}, axs::Tuple{NTuple{N1, Int}, NTuple{N2, Int}}) where {T, N, N1, N2}
QR decomposition of QTensor a, by joining axs to be the second dimension
"""
function tqr!(a::AbstractArray{T, N}, left::NTuple{N1, Int}, right::NTuple{N2, Int}, workspace::AbstractVector{T}=similar(a, length(a))) where {T<:Number, N, N1, N2}
    (N == N1 + N2) || throw(DimensionMismatch())
	if length(workspace) <= length(a)
		resize!(workspace, length(a))
	end
    newindex = (left..., right...)
    a1 = permute(a, newindex)
    shape_a = size(a1)
    dimu = shape_a[1:N1]
    s1 = prod(dimu)
    dimv = shape_a[(N1+1):end]
    s2 = prod(dimv)
    bmat = copyto!(reshape(view(workspace, 1:length(a)), s1, s2), reshape(a1, s1, s2))
    # F = LinearAlgebra.qr!(bmat)
    # u = Base.Matrix(F.Q)
    # v = Base.Matrix(F.R)
    u, v = tqr!(bmat)
    s = size(v, 1)
    return reshape(u, dimu..., s), reshape(v, s, dimv...)
end

function tlq!(a::AbstractArray{T, N}, left::NTuple{N1, Int}, right::NTuple{N2, Int}, workspace::AbstractVector{T}=similar(a, length(a))) where {T<:Number, N, N1, N2}
    (N == N1 + N2) || throw(DimensionMismatch())
	if length(workspace) <= length(a)
		resize!(workspace, length(a))
	end
    newindex = (left..., right...)
    a1 = permute(a, newindex)
    shape_a = size(a1)
    dimu = shape_a[1:N1]
    s1 = prod(dimu)
    dimv = shape_a[(N1+1):end]
    s2 = prod(dimv)
    bmat = copyto!(reshape(view(workspace, 1:length(a)), s1, s2), reshape(a1, s1, s2))
    # F = LinearAlgebra.lq!(bmat)
    # u = Matrix(F.L)
    # v = Matrix(F.Q)
    u, v = tlq!(bmat)
    s = size(v, 1)
    return reshape(u, dimu..., s), reshape(v, s, dimv...)
end

function texp(a::AbstractArray{T, N}, left::NTuple{N1, Int}, right::NTuple{N1, Int}) where {T <:Number, N, N1}
    (N == 2*N1) || throw(DimensionMismatch())
    perm = (left..., right...)
    a1 = permute(a, perm)
    shape_a = size(a1)
    for i in 1:N1
        (shape_a[i] == shape_a[i+N1]) || throw(DimensionMismatch())
    end
    m = prod(shape_a[1:(N-N1)])
    n = prod(shape_a[(N-N1+1):end])
    (m != n) && throw(ArgumentError("square matrix required.")) 
    # println(typeof(reshape(a1, m, n)))
    t2 = exp(Matrix(reshape(a1, m, n)))
    return reshape(t2, shape_a)
end



function _rightnull_lq!(A::StridedMatrix{<:BlasFloat}, atol::Real)
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by LQ"))
    m, n = size(A)
    k = min(m, n)
    At = adjoint!(similar(A, n, m), A)
    At, T = LAPACK.geqrt!(At, min(k, 36))
    N = similar(A, max(n-m, 0), n);
    fill!(N, 0)
    for k = 1:n-m
        N[k,m+k] = 1
    end
    N = LAPACK.gemqrt!('R', eltype(At) <: Real ? 'T' : 'C', At, T, N)
end

function _rightnull_svd!(A::StridedMatrix{<:BlasFloat}, atol::Real)
    size(A, 1) == 0 && return _one!(similar(A, (size(A, 2), size(A, 2))))
    U, S, V = LAPACK.gesvd!('N', 'A', A) 
    indstart = count(>(atol), S) + 1
    return V[indstart:end, :]
end

function _leftnull_qr!(A::StridedMatrix{<:BlasFloat}, atol::Real)
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by QR"))
    m, n = size(A)
    m >= n || throw(ArgumentError("no null space if less rows than columns"))

    A, T = LAPACK.geqrt!(A, min(minimum(size(A)), 36))
    N = similar(A, m, max(0, m-n));
    fill!(N, 0)
    for k = 1:m-n
        N[n+k,k] = 1
    end
    N = LAPACK.gemqrt!('L', 'N', A, T, N)
end

function _leftnull_svd!(A::StridedMatrix{<:BlasFloat}, atol::Real)
    size(A, 2) == 0 && return _one!(similar(A, (size(A, 1), size(A, 1))))
    U, S, V = LAPACK.gesvd!('A', 'N', A) 
    indstart = count(>(atol), S) + 1
    return U[:, indstart:end]
end

function entropy(v::AbstractVector{<:Real}) 
    a = v.*v
    s = sum(a)
    a ./= s
    return -dot(a, log.(a))
end

function renyi_entropy(v::AbstractVector{<:Real}, n::Int) 
    if n==1
        return entropy(v)
    else
        v ./= norm(v)
        a = v.^(2*n)
        return (1/(1-n)) * log(sum(a))
    end
end

# # m must be a square matrix
# function apply_physical!(m::StridedMatrix{T}, v::StridedArray{T, 3}, workspace::AbstractVector{T}) where T
#     L = size(m, 1) * size(v, 1)
#     if length(workspace) < L
#         resize!(workspace, L)
#     end
#     mv = reshape(view(workspace, 1:L), size(m, 1), size(v, 1))
#     beta = zero(T)
#     alpha = one(T)
#     for i in 1:size(v, 3)
#         vj = view(v, :, :, i)
#         gemm!('N', 'T', alpha, vj, m, beta, mv)
#         copy!(mv, vj)
#     end
#     return v
# end

# function apply_physical!(m::StridedMatrix{T}, vi::StridedArray{T, 3}, vo::StridedArray{T, 3}) where {T <: Number}
#     beta = zero(T)
#     alpha = one(T)
#     for i in 1:size(vi, 3)
#         vij = view(vi, :, :, i)
#         voj = view(vo, :, :, i)
#         gemm!('N', 'T', alpha, vij, m, beta, voj)
#     end
#     return vo
# end
# apply_physical!(m::StridedArray{T, 4}, vi::StridedArray{T, 4}, vo::StridedArray{T, 4}) where T = apply_physical!(tie(m, (2,2)), tie(vi, (1,2,1)), tie(vo, (1,2,1)))



