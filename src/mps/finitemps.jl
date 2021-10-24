
function _check_mps_space(mpstensors::Vector)
	L = length(mpstensors)
	for i in 1:L-1
		(size(mpstensors[i], 3) == size(mpstensors[i+1], 1)) || throw(DimensionMismatch())
	end

	# just require the left boundary to be a single sector
	(size(mpstensors[1], 1) == 1) || throw(DimensionMismatch("left boundary should be size 1."))
	(size(mpstensors[L], 3) == 1) || throw(DimensionMismatch("right boundary should be size 1."))
	return true
end

struct MPS{T<:Number, R<:Real} <: AbstractMPS
	data::Vector{Array{T, 3}}
	svectors::Vector{Array{R, 1}}

function MPS{T, R}(data::Vector{Array{T, 3}}, svectors::Vector{Array{R, 1}}) where {T<:Number, R<:Number}
	(R == real(T)) || throw(ArgumentError("scalar type for singular vectors must be real."))
	(length(data)+1 == length(svectors)) || throw(DimensionMismatch("length of singular vectors must be length of site tensors+1"))
	_check_mps_space(data)
	new{T, R}(data, svectors)
end

function MPS{T, R}(L::Int) where {T<:Number, R<:Number}
	(R == real(T)) || throw(ArgumentError("scalar type for singular vectors must be real."))
	new{T, R}(Vector{Array{T, 3}}(undef, L), Vector{Array{R, 1}}(undef, L+1))
end 

end


function Base.getproperty(psi::MPS, s::Symbol)
	if s == :s
		return MPSBondView(psi)
	else
		return getfield(psi, s)
	end
end
MPS(data::Vector{Array{T, 3}}, svectors::Vector{Array{R, 1}}) where {T, R} = MPS{T, R}(data, svectors)
MPS{T}(data::Vector{<:MPSTensor{C}}, svectors::Vector{Array{R, 1}}) where {T, C, R} = MPS{T, real(T)}(
	convert(Vector{Array{T, 3}}, data), convert(Vector{Array{real(T), 1}}, svectors))
function MPS{T}(data::Vector{<:MPSTensor{C}}) where {T<:Number, C<:Number}
	L = length(data)
	R = real(T)
	svectors = Vector{Array{R, 1}}(undef, L+1)
	return MPS{T, R}(convert(Vector{Array{T, 3}}, data), svectors)
end
MPS(data::Vector{<:MPSTensor{T}}) where {T <: Number} = MPS{T}(data)
MPS{T}(L::Int) where {T<:Number} = MPS{T, real(T)}(L)

"""
	The raw mps data as a list of 3-dimension tensors
	This is not supposed to be directly used by users
"""
raw_data(psi::MPS) = psi.data

"""
	The singular vectors are stored anyway even if the mps is not unitary.
	The raw singular vectors may not correspond to the correct Schmidt numbers
"""
raw_singular_matrices(psi::MPS) = psi.svectors

Base.eltype(::Type{MPS{T, R}}) where {T, R} = Array{T, 3}
Base.eltype(psi::MPS) = eltype(typeof(psi))
Base.length(psi::MPS) = length(raw_data(psi))
Base.isempty(psi::MPS) = isempty(raw_data(psi))
Base.size(psi::MPS, i...) = size(raw_data(psi), i...)
Base.getindex(psi::MPS, i::Int) = getindex(raw_data(psi), i)
Base.lastindex(psi::MPS) = lastindex(raw_data(psi))
Base.firstindex(psi::MPS) = firstindex(raw_data(psi))

Base.getindex(psi::MPS,r::AbstractRange{Int64}) = [psi[ri] for ri in r]
function Base.setindex!(psi::MPS, v::MPSTensor, i::Int)
	return setindex!(raw_data(psi), v, i)
end 
function Base.copy(psi::MPS)
	if svectors_uninitialized(psi)
		return MPS(copy(raw_data(psi)))
	else
		return MPS(copy(raw_data(psi)), copy(raw_singular_matrices(psi)))
	end
end
Base.conj(psi::MPS) = MPS(conj.(raw_data(psi)), copy(raw_singular_matrices(psi)) )
Base.adjoint(psi::MPS) = conj(psi)

scalar_type(::Type{MPS{T, R}}) where {T, R} = T
scalar_type(x::MPS) = scalar_type(typeof(x))

space_l(psi::MPS) = size(psi[1], 1)
space_r(psi::MPS) = size(psi[end], 3)

r_RR(psiA::MPS, psiB::MPS) = _eye(promote_type(scalar_type(psiA), scalar_type(psiB)), space_r(psiA), space_r(psiB))
r_RR(psi::MPS) = r_RR(psi, psi)
l_LL(psiA::MPS, psiB::MPS) = _eye(promote_type(scalar_type(psiA), scalar_type(psiB)), space_l(psiA), space_l(psiB))
l_LL(psi::MPS) = l_LL(psi, psi)

Base.cat(psiA::MPS, psiB::MPS) = MPS(vcat(raw_data(psiA), raw_data(psiB)))
# Base.conj(psi::MPS) = MPS(conj.(raw_data(psi)), raw_singular_matrices(psi))

function Base.complex(psi::MPS)
	if scalar_type(psi) <: Real
		data = [complex(item) for item in raw_data(psi)]
		if svectors_uninitialized(psi)
			return MPS(data)
		else
			return MPS(data, raw_singular_matrices(psi))
		end
	end
end

function _svectors_uninitialized(x)
	isempty(x) && return true
	s = raw_singular_matrices(x)
	for i in 1:length(s)
		isassigned(s , i) || return true
	end
	return false
end
svectors_uninitialized(psi::MPS) = _svectors_uninitialized(psi)
svectors_initialized(psi::MPS) = !svectors_uninitialized(psi)



function maybe_init_boundary_s!(psi::MPS{T, R}) where {T, R}
	isempty(psi) && return
	L = length(psi)
	if !isassigned(raw_singular_matrices(psi), 1)
		# (dim(space(psi[1], 1)) == 1) || throw(SpaceMismatch())
		psi.s[1] = ones(R, 1) 
	end
	if !isassigned(raw_singular_matrices(psi), L+1)
		psi.s[L+1] = ones(R, 1) 
	end
end

function _is_left_canonical(psij::MPSTensor; kwargs...)
	@tensor r[-1, -2] := conj(psij[1,2,-1]) * psij[1,2,-2]
	return isapprox(r, one(r); kwargs...) 
end

function _is_right_canonical(psij::MPSTensor; kwargs...)
	@tensor r[-1, -2] := conj(psij[-1,1,2]) * psij[-2,1,2]
	return isapprox(r, one(r); kwargs...) 
end



function is_right_canonical(psi::MPS; kwargs...)
	all([_is_right_canonical(item; kwargs...) for item in raw_data(psi)]) || return false
	# we also check whether the singular vectors are the correct Schmidt numbers
	svectors_uninitialized(psi) && return false
	hold = l_LL(psi)
	for i in 1:length(psi)-1
		hold = updateleft(hold, psi[i], psi[i])
		tmp = psi.s[i+1]
		isapprox(hold, Diagonal(tmp.^2); kwargs...) || return false
	end
	return true
end

"""
	iscanonical(psi::MPS; kwargs...) = is_right_canonical(psi; kwargs...)
check if the state is right-canonical, the singular vectors are also checked that whether there are the correct Schmidt numbers or not
This form is useful for time evolution for stability issue and also efficient for computing observers of unitary systems
"""
iscanonical(psi::MPS; kwargs...) = is_right_canonical(psi; kwargs...)

"""
	bond_dimension(psi::MPS[, bond::Int])
	bond_dimension(h::MPO[, bond::Int])
return bond dimension at the given bond, or return the largest bond dimension of all bonds.
"""
bond_dimension(psi::MPS, bond::Int) = begin
	((bond >= 1) && (bond < length(psi))) || throw(BoundsError())
	size(psi[bond], 3)
end 
bond_dimensions(psi::MPS) = [bond_dimension(psi, i) for i in 1:length(psi)-1]
bond_dimension(psi::MPS) = maximum(bond_dimensions(psi))

"""
	physical_dimensions(psi::MPS)
	physical_dimensions(psi::MPO) 
Return all the physical spaces of MPS or MPO
"""
physical_dimensions(psi::MPS) = [size(item, 2) for item in raw_data(psi)]

function max_bond_dimensions(physpaces::Vector{Int}, D::Int) 
	L = length(physpaces)
	left = 1
	right = 1
	virtualpaces = Vector{Int}(undef, L+1)
	virtualpaces[1] = left
	for i in 2:L
		virtualpaces[i] = min(virtualpaces[i-1] * physpaces[i-1], D)
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		virtualpaces[i] = min(virtualpaces[i], physpaces[i] * virtualpaces[i+1])
	end
	return virtualpaces
end
max_bond_dimensions(psi::MPS, D::Int) = max_bond_dimensions(physical_dimensions(psi), D)


function increase_bond!(psi::MPS; D::Int)
	if bond_dimension(psi) < D
		virtualpaces = max_bond_dimensions(physical_dimensions(psi), D)
		for i in 1:length(psi)
			sl = max(min(virtualpaces[i], D), size(psi[i], 1))
			sr = max(min(virtualpaces[i+1], D), size(psi[i], 3))
			m = zeros(scalar_type(psi), sl, size(psi[i], 2), sr)
			m[1:size(psi[i], 1), :, 1:size(psi[i], 3)] .= psi[i]
			psi[i] = m
		end
	end
	return psi
end



