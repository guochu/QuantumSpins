# orthogonalize mps to be left-canonical or right-canonical

function _leftorth_qr!(psi::MPO, normalize::Bool, workspace::AbstractVector=Vector{scalar_type(psi)}())
	L = length(psi)
	for i in 1:L-1
		q, r = tqr!(psi[i], (1, 2, 4), (3,), workspace)
		normalize && normalize!(r)
		psi[i] = permute(q, (1,2,4,3))
		psi[i+1] = @tensor tmp[1,3,4,5] := r[1,2] * psi[i+1][2,3,4,5]
	end
	normalize && normalize!(psi[L])
end

function _leftorth_svd!(psi::MPO, workspace::AbstractVector=Vector{scalar_type(psi)}(), trunc::TruncationScheme = NoTruncation())
	L = length(psi)
	errs = Float64[]
	for i in 1:L-1
		# u, s, v, err = stable_tsvd(psi[i], (1, 2), (3,), trunc=trunc)
		u, s, v, err = tsvd!(psi[i], (1,2,4), (3), workspace, trunc=trunc)
		psi[i] = permute(u, (1,2,4,3))
		v = Diagonal(s) * v
		psi[i+1] = @tensor tmp[1,3,4,5] := v[1,2] * psi[i+1][2,3,4,5]
		push!(errs, err)
	end
	return errs
end

_leftorth!(psi::MPO, workspace::AbstractVector, alg::QRFact) = _leftorth_qr!(psi, alg.normalize, workspace)
_leftorth!(psi::MPO, workspace::AbstractVector, alg::SVDFact) = _leftorth_svd!(psi, workspace, alg.trunc)
leftorth!(psi::MPO, workspace::AbstractVector=Vector{scalar_type(psi)}(); alg::AbstractMatrixFactorization=SVDFact()) = _leftorth!(psi, workspace, alg)

function _rightorth_svd!(psi::MPO, workspace::AbstractVector=Vector{scalar_type(psi)}(), trunc::TruncationScheme = NoTruncation())
	L = length(psi)
	errs = Float64[]
	for i in L:-1:2
		u, s, v, err = tsvd!(psi[i], (1,), (2, 3, 4), workspace, trunc=trunc)
		psi[i] = v
		u = u * Diagonal(s)
		psi[i-1] = @tensor tmp[1,2,5,4] := psi[i-1][1,2,3,4] * u[3,5]
		push!(errs, err)
	end
	return errs
end

function _rightorth_qr!(psi::MPO, normalize::Bool, workspace::AbstractVector=Vector{scalar_type(psi)}())
	L = length(psi)
	for i in L:-1:2
		l, q = tlq!(psi[i], (1,), (2, 3, 4), workspace)
		normalize && normalize!(l)
		psi[i] = q
		psi[i-1] = @tensor tmp[1,2,5,4] := psi[i-1][1,2,3,4] * l[3,5]
	end
	normalize && normalize!(psi[1])
end
_rightorth!(psi::MPO, workspace::AbstractVector, alg::QRFact) = _rightorth_qr!(psi, alg.normalize, workspace)
_rightorth!(psi::MPO, workspace::AbstractVector, alg::SVDFact) = _rightorth_svd!(psi, workspace, alg.trunc)
rightorth!(psi::MPO, workspace::AbstractVector=Vector{scalar_type(psi)}(); alg::AbstractMatrixFactorization=SVDFact()) = _rightorth!(psi, workspace, alg)
