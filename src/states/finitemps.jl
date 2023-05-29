
struct MPS{T<:Number, R<:Real} <: AbstractPureMPS
	data::Vector{Array{T, 3}}
	svectors::Vector{Array{R, 1}}

function MPS{T, R}(data::Vector, svectors::Vector) where {T<:Number, R<:Number}
	(R == real(T)) || throw(ArgumentError("scalar type for singular vectors must be real"))
	(length(data)+1 == length(svectors)) || throw(DimensionMismatch("length of singular vectors must be length of site tensors+1"))
	_check_mps_space(data)
	new{T, R}(convert(Vector{Array{T, 3}}, data), convert(Vector{Vector{R}}, svectors))
end
function MPS{T, R}(data::Vector) where {T<:Number, R<:Number}
	(R == real(T)) || throw(ArgumentError("scalar type for singular vectors must be real"))
	_check_mps_space(data)
	svectors = Vector{Vector{R}}(undef, length(data)+1)
	svectors[1] = ones(space_l(data[1]))
	svectors[end] = ones(space_r(data[end]))
	return new{T, R}(convert(Vector{Array{T, 3}}, data), svectors)
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
MPS{T}(data::Vector{<:MPSTensor}) where {T} = MPS{T, real(T)}(data)
MPS(data::Vector{<:MPSTensor{T}}) where {T <: Number} = MPS{T}(data)

"""
	The singular vectors are stored anyway even if the mps is not unitary.
	The raw singular vectors may not correspond to the correct Schmidt numbers
"""
raw_singular_matrices(psi::MPS) = psi.svectors

Base.eltype(::Type{MPS{T, R}}) where {T, R} = T

function Base.copy(psi::MPS)
	if svectors_uninitialized(psi)
		return MPS(copy(raw_data(psi)))
	else
		return MPS(copy(raw_data(psi)), copy(raw_singular_matrices(psi)))
	end
end
Base.conj(psi::MPS) = MPS(conj.(raw_data(psi)), copy(raw_singular_matrices(psi)) )
Base.adjoint(psi::MPS) = conj(psi)


Base.vcat(psiA::MPS, psiB::MPS) = MPS(vcat(raw_data(psiA), raw_data(psiB)))
# Base.conj(psi::MPS) = MPS(conj.(raw_data(psi)), raw_singular_matrices(psi))

function Base.complex(psi::MPS)
	if eltype(psi) <: Real
		data = [complex(item) for item in raw_data(psi)]
		return MPS(data, psi.svectors)
	end
	return psi
end

svectors_uninitialized(psi::MPS) = any(x->!isassigned(psi.svectors, x), 1:length(psi)+1)

isleftcanonical(a::MPS; kwargs...) = all(x->isleftcanonical(x; kwargs...), a.data)
isrightcanonical(a::MPS; kwargs...) = all(x->isrightcanonical(x; kwargs...), a.data)

"""
	iscanonical(psi::MPS; kwargs...) = is_right_canonical(psi; kwargs...)
check if the state is right-canonical, the singular vectors are also checked that whether there are the correct Schmidt numbers or not
This form is useful for time evolution for stability issue and also efficient for computing observers of unitary systems
"""
function iscanonical(psi::MPS; kwargs...)
	isrightcanonical(psi) || return false
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
			m = zeros(eltype(psi), sl, size(psi[i], 2), sr)
			m[1:size(psi[i], 1), :, 1:size(psi[i], 3)] .= psi[i]
			psi[i] = m
		end
	end
	return psi
end


function _check_mps_space(mpstensors::Vector)
	L = length(mpstensors)
	for i in 1:L-1
		(space_r(mpstensors[i]) == space_l(mpstensors[i+1])) || throw(DimensionMismatch())
	end

	# just require the left boundary to be a single sector
	(space_l(mpstensors[1]) == 1) || throw(DimensionMismatch("left boundary should be size 1."))
	# (size(mpstensors[L], 3) == 1) || throw(DimensionMismatch("right boundary should be size 1."))
	return true
end
