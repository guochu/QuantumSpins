# orthogonalize mps to be left-canonical or right-canonical



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


function right_canonicalize!(psi::MPS, workspace::AbstractVector=Vector{scalar_type(psi)}(); normalize::Bool=false, trunc::TruncationScheme = NoTruncation())
	err1 = leftorth!(psi, workspace, trunc=trunc)
	if normalize
		psi[end] /= norm(psi[end])
	end
	err2 = rightorth!(psi, workspace, trunc=trunc)
	append!(err1, err2)
	return err1
end

canonicalize!(psi::MPS, args...; kwargs...) = right_canonicalize!(psi, args...; kwargs...)