

function prodmpo(physpaces::Vector{Int}, m::QTerm) 
	is_constant(m) || throw(ArgumentError("only constant term allowed."))
	return prodmpo(scalar_type(m), physpaces, positions(m), op(m)) * value(coeff(m))
end

function MPO(physpaces::Vector{Int}, h::QuantumOperator; alg::AbstractCompression=Deparallelise())
	local mpo
	compress_threshold = 20
	for m in qterms(h)
		if (@isdefined mpo) && (!isnothing(mpo))
			mpo += prodmpo(physpaces, m)
		else
			mpo = prodmpo(physpaces, m)
		end
		if bond_dimension(mpo) >= compress_threshold
			mpo = compress!(mpo, alg=alg)
			compress_threshold += 5
		end
	end
	if (!(@isdefined mpo)) || isnothing(mpo) 
		@warn "Hamiltonian is empty after compression."
		return nothing
	end
	mpo = compress!(mpo, alg=alg)
	if isnothing(mpo)
		@warn "Hamiltonian is empty after final compression."
		return nothing
	end
	return mpo
end
MPO(h::QuantumOperator; kwargs...) = MPO(physical_dimensions(h), h; kwargs...)

MPO(physpaces::Vector{Int}, h::SuperOperator; kwargs...) = MPO(physpaces, h.data; kwargs...)
MPO(h::SuperOperator; kwargs...) = MPO(h.data; kwargs...)

function simplify(h::QuantumOperator, alg::AbstractCompression)
	is_constant(h) || throw(ArgumentError("Simplifying non constant Hamiltonian not supported."))
	r = similar(h)
	for (k, v) in raw_data(h)
		tmp = _qterm_compress(k, v, alg)
		if !isnothing(tmp)
			add!(r, tmp)
		end
	end
	return r
end

"""
	simplify(h::QuantumOperator; alg::AbstractCompression=default_mpo_compression())

Simplify the QuantumOperator the terms with the same positions into a single QTerm, 
and compress the bond dimension of the resulting QTerm.
"""
simplify(h::QuantumOperator; alg::AbstractCompression=default_mpo_compression()) = simplify(h, alg)
simplify(h::SuperOperator; kwargs...) = SuperOperator(simplify(h.data))

function _qterm_compress(pos::Tuple, v::Vector{Tuple{Vector{M}, AbstractCoefficient}}, alg::AbstractCompression) where {M <: MPOTensor}
	(isempty(v) || isempty(pos)) && error("empty term.")
	if length(pos) == 1
		m = v[1][1][1] * value(v[1][2])
		for i in 2:length(v)
			m += v[i][1][1] * value(v[i][2])
		end
		return QTerm(pos, [m])
	end
	new_pos = collect(1:length(pos))
	physpaces = [size(item, 2) for item in v[1][1]]

	local mpo
	compress_threshold = 20
	for (m, alhpa) in v
		# is_constant(alhpa) || error("only constant terms are allowed.")
		if (@isdefined mpo) && (!isnothing(mpo))
			mpo += prodmpo(physpaces, new_pos, m) * value(alhpa)
		else
			mpo = prodmpo(physpaces, new_pos, m) * value(alhpa)
		end
		if bond_dimension(mpo) >= compress_threshold
			mpo = compress!(mpo, alg)
			compress_threshold += 5
		end
	end
	if (!(@isdefined mpo)) || isnothing(mpo) 
		(alg.verbosity>=1) && println("Hamiltonian is empty after compression.")
		return nothing
	end
	mpo = compress!(mpo, alg)
	if isnothing(mpo)
		(alg.verbosity>=1) && println("Hamiltonian is empty after final compression.")
		return nothing
	end
	return QTerm(pos, raw_data(mpo))
end
