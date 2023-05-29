# orthogonalize mps to be left-canonical or right-canonical
abstract type MatrixProductOrthogonalAlgorithm  end

"""
	struct MatrixProductOrthogonalize{A<:Union{QR, SVD}, T<:TruncationScheme}
"""
struct Orthogonalize{A<:Union{QR, SVD}, T<:TruncationScheme} <: MatrixProductOrthogonalAlgorithm
	orth::A
	trunc::T
	normalize::Bool
end
Orthogonalize(a::Union{QR, SVD}=SVD(), trunc::TruncationScheme=NoTruncation(); normalize::Bool=false) = Orthogonalize(a, trunc, normalize)


leftorth!(psi::MPS, workspace::AbstractVector=Vector{eltype(psi)}(); alg::Orthogonalize = Orthogonalize()) = _leftorth!(psi, alg.orth, alg.trunc, alg.normalize, workspace)
function _leftorth!(psi::MPS, alg::QR, trunc::TruncationScheme, normalize::Bool, workspace::AbstractVector=Vector{eltype(psi)}())
	!isa(trunc, NoTruncation) &&  @warn "truncation has no effect with QR"
	L = length(psi)
	for i in 1:L-1
		q, r = tqr!(psi[i], (1, 2), (3,), workspace)
		normalize && normalize!(r)
		psi[i] = q
		psi[i+1] = reshape(r * tie(psi[i+1], (1, 2)), size(r, 1), size(psi[i+1], 2), size(psi[i+1], 3))
	end
	normalize && normalize!(psi[L])
	return psi
end

function _leftorth!(psi::MPS, alg::SVD, trunc::TruncationScheme, normalize::Bool, workspace::AbstractVector=Vector{eltype(psi)}())
	L = length(psi)
	# errs = Float64[]
	for i in 1:L-1
		# u, s, v, err = stable_tsvd(psi[i], (1, 2), (3,), trunc=trunc)
		u, s, v, err = tsvd!(tie(psi[i], (2, 1)), workspace, trunc=trunc)
		normalize && normalize!(s)
		d = length(s)
		psi[i] = reshape(u, size(psi[i],1), size(psi[i],2),d)
		v = Diagonal(s) * v
		psi[i+1] = reshape(v * tie(psi[i+1], (1, 2)), d, size(psi[i+1], 2), size(psi[i+1], 3))
		psi.s[i+1] = s
		# push!(errs, err)
	end
	normalize && normalize!(psi[L])
	return psi
end

rightorth!(psi::MPS, workspace::AbstractVector=Vector{eltype(psi)}(); alg::Orthogonalize = Orthogonalize()) = _rightorth!(psi, alg.orth, alg.trunc, alg.normalize, workspace)
function _rightorth!(psi::MPS, alg::QR, trunc::TruncationScheme, normalize::Bool, workspace::AbstractVector=Vector{eltype(psi)}())
	!isa(trunc, NoTruncation) &&  @warn "truncation has no effect with QR"
	L = length(psi)
	for i in L:-1:2
		l, q = tlq!(psi[i], (1,), (2, 3), workspace)
		normalize && normalize!(l)
		psi[i] = q
		psi[i-1] = reshape(tie(psi[i-1], (2, 1)) * l, size(psi[i-1], 1), size(psi[i-1], 2), size(l, 2))
	end
	normalize && normalize!(psi[1])
	return psi
end

function _rightorth!(psi::MPS, alg::SVD, trunc::TruncationScheme, normalize::Bool, workspace::AbstractVector=Vector{eltype(psi)}())
	L = length(psi)
	# errs = Float64[]
	for i in L:-1:2
		# u, s, v, err = stable_tsvd(psi[i], (1,), (2, 3), trunc=trunc)
		u, s, v, err = tsvd!(tie(psi[i], (1, 2)), workspace, trunc=trunc)
		normalize && normalize!(s)
		d = length(s)
		psi[i] = reshape(v, d, size(psi[i], 2), size(psi[i], 3))
		u = u * Diagonal(s)
		psi[i-1] = reshape(tie(psi[i-1], (2, 1)) * u, size(psi[i-1], 1), size(psi[i-1], 2), d)
		psi.s[i] = s
		# push!(errs, err)
	end
	return psi
end

function rightcanonicalize!(psi::MPS, workspace::AbstractVector=Vector{eltype(psi)}(); alg::Orthogonalize = Orthogonalize(SVD(), DefaultTruncation, normalize=false))
	_leftorth!(psi, QR(), NoTruncation(), alg.normalize, workspace)
	return rightorth!(psi, workspace, alg=alg)
end

canonicalize!(psi::MPS, args...; kwargs...) = rightcanonicalize!(psi, args...; kwargs...)

