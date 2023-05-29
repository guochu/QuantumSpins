const ValidIndices = Union{Integer,AbstractRange{Int64}, Colon}

const MPSTensor{T} = AbstractArray{T, 3} where {T<:Number}
const MPOTensor{T} = AbstractArray{T, 4} where {T<:Number}
const SingularVector{T} = AbstractVector{T} where {T <: Real}

abstract type AbstractMPS end
abstract type AbstractPureMPS <: AbstractMPS end
abstract type AbstractMixedMPS <: AbstractMPS end


"""
	The raw mps data as a list of 3-dimension tensors
	This is not supposed to be directly used by users
"""
raw_data(psi::AbstractPureMPS) = psi.data
raw_data(x::AbstractMixedMPS) = raw_data(x.data)

Base.eltype(psi::AbstractMPS) = eltype(typeof(psi))
Base.length(psi::AbstractMPS) = length(raw_data(psi))
Base.isempty(psi::AbstractMPS) = isempty(raw_data(psi))
Base.size(psi::AbstractMPS, i...) = size(raw_data(psi), i...)
Base.getindex(psi::AbstractMPS, i::Int) = getindex(raw_data(psi), i)
Base.lastindex(psi::AbstractMPS) = lastindex(raw_data(psi))
Base.firstindex(psi::AbstractMPS) = firstindex(raw_data(psi))

function Base.setindex!(psi::AbstractMPS, v::MPSTensor, i::Int)
	return setindex!(raw_data(psi), v, i)
end 

space_l(m::MPSTensor) = size(m, 1)
space_r(m::MPSTensor) = size(m, 3)
space_l(m::MPOTensor) = size(m, 1)
space_r(m::MPOTensor) = size(m, 3)
space_l(psi::AbstractMPS) = size(psi[1], 1)
space_r(psi::AbstractMPS) = size(psi[end], 3)

r_RR(psiA::AbstractMPS, psiB::AbstractMPS) = _eye(promote_type(eltype(psiA), eltype(psiB)), space_r(psiA), space_r(psiB))
r_RR(psi::AbstractMPS) = r_RR(psi, psi)
l_LL(psiA::AbstractMPS, psiB::AbstractMPS) = _eye(promote_type(eltype(psiA), eltype(psiB)), space_l(psiA), space_l(psiB))
l_LL(psi::AbstractMPS) = l_LL(psi, psi)

"""
	bond_dimension(psi::MPS[, bond::Int])
	bond_dimension(h::MPO[, bond::Int])
return bond dimension at the given bond, or return the largest bond dimension of all bonds.
"""
bond_dimension(psi::AbstractMPS, bond::Int) = begin
	((bond >= 1) && (bond <= length(psi))) || throw(BoundsError())
	space_r(psi[bond])
end 
bond_dimensions(psi::AbstractMPS) = [bond_dimension(psi, i) for i in 1:length(psi)]
bond_dimension(psi::AbstractMPS) = maximum(bond_dimensions(psi))

"""
	physical_dimensions(psi::MPS)
	physical_dimensions(psi::MPO) 
Return all the physical spaces of MPS or MPO
"""
physical_dimensions(psi::AbstractMPS) = [size(item, 2) for item in raw_data(psi)]

isstrict(psi::AbstractMPS) = (space_l(psi) == space_r(psi) == 1)


function isleftcanonical(psij::MPSTensor; kwargs...)
	@tensor r[-1, -2] := conj(psij[1,2,-1]) * psij[1,2,-2]
	return isapprox(r, one(r); kwargs...) 
end
function isrightcanonical(psij::MPSTensor; kwargs...)
	@tensor r[-1, -2] := conj(psij[-1,1,2]) * psij[-2,1,2]
	return isapprox(r, one(r); kwargs...) 
end
function isleftcanonical(psij::MPOTensor; kwargs...)
	@tensor r[-1; -2] := conj(psij[1,2,-1,3]) * psij[1,2,-2,3]
	return isapprox(r, one(r); kwargs...) 
end
function isrightcanonical(psij::MPOTensor; kwargs...)
	@tensor r[-1; -2] := conj(psij[-1,1,2,3]) * psij[-2,1,2,3]
	return isapprox(r, one(r); kwargs...) 
end