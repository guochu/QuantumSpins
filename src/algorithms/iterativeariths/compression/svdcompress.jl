
function svdcompress(psi::MPS{T, R}, alg::SVDArith = SVDArith()) where {T, R}
	L = length(psi)
	(L <= 1) && return copy(psi)
	data = Vector{Array{T, 3}}(undef, L)

	errs = Float64[]
	data[1] = psi[1]
	workspace = T[]

	trunc = get_trunc(alg)
	for i in 1:L-1
		u, s, v, err = tsvd!(copy(data[i]), (1,2), (3,), workspace, trunc=trunc)
		data[i] = u
		v = Diagonal(s) * v
		data[i+1] = @tensor tmp[-1 -2; -3] := v[-1, 1] * psi[i+1][1,-2,-3]
		push!(errs, err)
	end

	r = MPS(data)
	append!(errs, rightorth!(r, alg=SVDFact(trunc=trunc)))
	return r
end

function svdcompress(h::MPO{T}, alg::SVDArith = SVDArith()) where T
	L = length(h)
	(L <= 1) && return copy(h)
	data = Vector{Array{T, 4}}(undef, L)

	errs = Float64[]
	data[1] = h[1]
	workspace = T[]

	trunc = get_trunc(alg)
	for i in 1:L-1
		u, s, v, err = tsvd!(copy(data[i]), (1,2,4), (3,), workspace, trunc=trunc)
		data[i] = permute(u, (1,2), (4,3))
		v = Diagonal(s) * v
		data[i+1] = @tensor tmp[1 3; 4 5] := v[1,2] * h[i+1][2,3,4,5]
		push!(errs, err)
	end
	for i in L:-1:2
		u, s, v, err = tsvd!(data[i], (1,), (2,3,4), workspace, trunc=trunc)
		data[i] = permute(v, (1,2), (3,4))
		u = u * Diagonal(s)
		data[i-1] = @tensor tmp[-1 -2; -3 -4] := data[i-1][-1, -2, 1, -4] * u[1, -3]
		push!(errs, err)
	end

	return MPO(data)
end