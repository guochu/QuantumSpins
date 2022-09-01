

@with_kw struct SVDArith <: AbstractMPSArith
	D::Int = 100
	ϵ::Float64 = 1.0e-8
	verbosity::Int = 1
end

changeD(x::SVDArith; D::Int) = SVDArith(D=D, ϵ=x.ϵ, verbosity=x.verbosity)


function svd_mult(mpo::AbstractMPO, mps::AbstractMPS, alg::SVDArith = SVDArith())
	(length(mpo) != length(mps)) && error("dot mps requires mpo and mps of same size.")
	isempty(mpo) && error("mpo is empty.")
	L = length(mps)
	# res = promote_type(typeof(mpo), typeof(mps))(L)
	T = promote_type(scalar_type(mpo), scalar_type(mps))
	trunc = MPSTruncation(D=alg.D, ϵ=alg.ϵ)
	res = MPS{T}(L)

	@tensor tmp[1,5,2,3,6] := mpo[1][1,2,3,4] * mps[1][5,4,6]
	tmp3 = tie(tmp, (2,1,2))
	workspace = T[]
	u, s, v, err = tsvd!(tmp3, (1,2), (3,), workspace, trunc=trunc)
	res[1] = u
	v = Diagonal(s) * v

	for i in 2:L-1
	    @tensor tmp[1,5,2,3,6] := mpo[i][1,2,3,4] * mps[i][5,4,6]
	    tmp3 = tie(tmp, (2,1,2))
	    @tensor tmp3c[1,3,4] := v[1,2] * tmp3[2,3,4]
	    u, s, v, err = tsvd!(tmp3c, (1,2), (3,), workspace, trunc=trunc)
	    res[i] = u
	    v = Diagonal(s) * v
	end
	i = L
	@tensor tmp[1,5,2,3,6] := mpo[i][1,2,3,4] * mps[i][5,4,6]
	tmp3 = tie(tmp, (2,1,2))
	@tensor tmp3c[1,3,4] := v[1,2] * tmp3[2,3,4]
	res[L] = tmp3c
	rightorth!(res, workspace, alg=SVD(trunc=trunc))
	return res, err
end

