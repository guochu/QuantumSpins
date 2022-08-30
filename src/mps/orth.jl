# orthogonalize mps to be left-canonical or right-canonical

function leftorth_qr!(psi::MPS, workspace::AbstractVector=Vector{scalar_type(psi)}())
	L = length(psi)
	for i in 1:L-1
		q, r = tqr!(psi[i], (1, 2), (3,), workspace)
		psi[i] = q
		psi[i+1] = reshape(r * tie(psi[i+1], (1, 2)), size(r, 1), size(psi[i+1], 2), size(psi[i+1], 3))
	end
end

function leftorth!(psi::MPS, workspace::AbstractVector=Vector{scalar_type(psi)}(); trunc::TruncationScheme = NoTruncation())
	L = length(psi)
	maybe_init_boundary_s!(psi)
	errs = Float64[]
	for i in 1:L-1
		# u, s, v, err = stable_tsvd(psi[i], (1, 2), (3,), trunc=trunc)
		u, s, v, err = tsvd!(tie(psi[i], (2, 1)), workspace, trunc=trunc)
		d = length(s)
		psi[i] = reshape(u, size(psi[i],1), size(psi[i],2),d)
		v = Diagonal(s) * v
		psi[i+1] = reshape(v * tie(psi[i+1], (1, 2)), d, size(psi[i+1], 2), size(psi[i+1], 3))
		psi.s[i+1] = s
		push!(errs, err)
	end
	return errs
end

function rightorth!(psi::MPS, workspace::AbstractVector=Vector{scalar_type(psi)}(); trunc::TruncationScheme = NoTruncation())
	L = length(psi)
	maybe_init_boundary_s!(psi)
	errs = Float64[]
	for i in L:-1:2
		# u, s, v, err = stable_tsvd(psi[i], (1,), (2, 3), trunc=trunc)
		u, s, v, err = tsvd!(tie(psi[i], (1, 2)), workspace, trunc=trunc)
		d = length(s)
		psi[i] = reshape(v, d, size(psi[i], 2), size(psi[i], 3))
		u = u * Diagonal(s)
		psi[i-1] = reshape(tie(psi[i-1], (2, 1)) * u, size(psi[i-1], 1), size(psi[i-1], 2), d)
		psi.s[i] = s
		push!(errs, err)
	end
	return errs
end

function rightorth_qr!(psi::MPS, workspace::AbstractVector=Vector{scalar_type(psi)}(); trunc::TruncationScheme = NoTruncation())
	L = length(psi)
	for i in L:-1:2
		l, q = tlq!(psi[i], (1,), (2, 3), workspace)
		psi[i] = q
		psi[i-1] = reshape(tie(psi[i-1], (2, 1)) * l, size(psi[i-1], 1), size(psi[i-1], 2), size(l, 2))
	end
end

function right_canonicalize!(psi::MPS, workspace::AbstractVector=Vector{scalar_type(psi)}(); normalize::Bool=false, trunc::TruncationScheme = NoTruncation())
	leftorth_qr!(psi, workspace)
	if normalize
		psi[end] ./= norm(psi[end])
	end
	errs = rightorth!(psi, workspace, trunc=trunc)
	return errs
end

canonicalize!(psi::MPS, args...; kwargs...) = right_canonicalize!(psi, args...; kwargs...)
