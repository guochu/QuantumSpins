# orthogonalize mps to be left-canonical or right-canonical

leftorth!(psi::MPO, workspace::AbstractVector=Vector{eltype(psi)}(); alg::Orthogonalize = Orthogonalize()) = _leftorth!(h, alg.orth, alg.trunc, alg.normalize, workspace)
function _leftorth!(psi::MPO, alg::QR, trunc::TruncationScheme, normalize::Bool, workspace::AbstractVector=Vector{eltype(psi)}())
	L = length(psi)
	for i in 1:L-1
		q, r = tqr!(psi[i], (1, 2, 4), (3,), workspace)
		normalize && normalize!(r)
		psi[i] = permute(q, (1,2,4,3))
		psi[i+1] = @tensor tmp[1,3,4,5] := r[1,2] * psi[i+1][2,3,4,5]
	end
	normalize && normalize!(psi[L])
	return psi
end

function _leftorth!(psi::MPO, alg::SVD, trunc::TruncationScheme, normalize::Bool, workspace::AbstractVector=Vector{eltype(psi)}())
	L = length(psi)
	# errs = Float64[]
	for i in 1:L-1
		# u, s, v, err = stable_tsvd(psi[i], (1, 2), (3,), trunc=trunc)
		u, s, v, err = tsvd!(psi[i], (1,2,4), (3), workspace, trunc=trunc)
		psi[i] = permute(u, (1,2,4,3))
		v = Diagonal(s) * v
		psi[i+1] = @tensor tmp[1,3,4,5] := v[1,2] * psi[i+1][2,3,4,5]
		# push!(errs, err)
	end
	normalize && normalize!(h[L])
	return psi
end

rightorth!(h::MPO, workspace::AbstractVector=Vector{eltype(psi)}(); alg::Orthogonalize = Orthogonalize()) = _rightorth!(h, alg.orth, alg.trunc, alg.normalize)
function _rightorth!(psi::MPO, alg::QR, trunc::TruncationScheme, normalize::Bool, workspace::AbstractVector=Vector{eltype(psi)}())
	L = length(psi)
	for i in L:-1:2
		l, q = tlq!(psi[i], (1,), (2, 3, 4), workspace)
		normalize && normalize!(l)
		psi[i] = q
		psi[i-1] = @tensor tmp[1,2,5,4] := psi[i-1][1,2,3,4] * l[3,5]
	end
	normalize && normalize!(psi[1])
	return psi
end
function _rightorth!(psi::MPO, alg::SVD, trunc::TruncationScheme, normalize::Bool, workspace::AbstractVector=Vector{eltype(psi)}())
	L = length(psi)
	# errs = Float64[]
	for i in L:-1:2
		u, s, v, err = tsvd!(psi[i], (1,), (2, 3, 4), workspace, trunc=trunc)
		psi[i] = v
		u = u * Diagonal(s)
		psi[i-1] = @tensor tmp[1,2,5,4] := psi[i-1][1,2,3,4] * u[3,5]
		# push!(errs, err)
	end
	normalize && normalize!(psi[1])
	return psi
end
