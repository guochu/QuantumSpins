
# density operator is treated as a specialized MPS by fusing the two physical indices
# a fuser is a 3-d tensor from two physical indices into a single index
# fuser is used to go back to an MPO 
# I is ⟨I| which is used to compute expectations
# Thus default constructor should never be directly used by users

function identity_mps(::Type{T}, ds::Vector{Int}) where {T <: Number}
	L = length(ds)
	v = Vector{Vector{T}}(undef, L)
	for i in 1:L
		d = ds[i]
		v[i] = reshape(_eye(T, d), d * d)
	end
	return prodmps(T, v)
end

"""
	struct DensityOperatorMPS{A<:MPSTensor, B<:MPSBondTensor}
"""
struct DensityOperatorMPS{T<:Number, R<:Real} <: AbstractMPS
	data::MPS{T, R}
	fusers::Vector{Array{T, 3}}
	I::MPS{T, R}
end

function Base.getproperty(psi::DensityOperatorMPS{T, R}, s::Symbol) where {T, R}
	if s == :s
		return psi.data.s
	else
		return getfield(psi, s)
	end
end


Base.eltype(psi::DensityOperatorMPS) = eltype(typeof(psi))
Base.length(psi::DensityOperatorMPS) = length(psi.data)
Base.isempty(psi::DensityOperatorMPS) = isempty(psi.data)
Base.size(psi::DensityOperatorMPS, i...) = size(psi.data, i...)
Base.getindex(psi::DensityOperatorMPS, i::Int) = getindex(psi.data, i)
Base.lastindex(psi::DensityOperatorMPS) = lastindex(psi.data)
Base.firstindex(psi::DensityOperatorMPS) = firstindex(psi.data)

Base.getindex(psi::DensityOperatorMPS,r::AbstractRange{Int64}) = [psi[ri] for ri in r]


scalar_type(psi::DensityOperatorMPS) = scalar_type(typeof(psi))


Base.eltype(::Type{DensityOperatorMPS{A, B}}) where {A, B} = A
scalar_type(::Type{DensityOperatorMPS{A, B}}) where {A, B} = eltype(A)
Base.setindex!(psi::DensityOperatorMPS, v, i::Int) = setindex!(psi.data, v, i)
Base.copy(psi::DensityOperatorMPS) = DensityOperatorMPS(copy(psi.data), psi.fusers, psi.I)
Base.isapprox(x::DensityOperatorMPS, y::DensityOperatorMPS; kwargs...) = isapprox(x.data, y.data; kwargs...)

bond_dimension(psi::DensityOperatorMPS, bond::Int) = bond_dimension(psi.data, bond)
bond_dimensions(psi::DensityOperatorMPS) = bond_dimensions(psi.data)
bond_dimension(psi::DensityOperatorMPS) = bond_dimension(psi.data)

physical_dimensions(psi::DensityOperatorMPS) = physical_dimensions(psi.data)

LinearAlgebra.tr(psi::DensityOperatorMPS) = dot(psi.I, psi.data)

increase_bond!(psi::DensityOperatorMPS; kwargs...) = begin
	increase_bond!(psi.data; kwargs...)
	return psi
end

canonicalize!(psi::DensityOperatorMPS; kwargs...) = canonicalize!(psi.data; kwargs...)
default_fusers(::Type{T}, ds::Vector{Int}) where {T<:Number} = [reshape(_eye(T, d*d), d, d, d*d) for d in ds]

