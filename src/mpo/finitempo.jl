
function _check_mpo_space(mpotensors::Vector)
	for i in 1:length(mpotensors)-1
		(space_r(mpotensors[i]) == space_l(mpotensors[i+1])) || throw(DimensionMismatch())
	end
	# boundaries should be dimension 
	(space_l(mpotensors[1]) == 1) || throw(DimensionMismatch())
	return true
end



"""
	MPO{A <: MPOTensor}
Finite Matrix Product Operator which stores a chain of rank-4 site tensors.
"""
struct MPO{T<:Number} <: AbstractMPO
	data::Vector{Array{T, 4}}

"""
	MPO{A}(mpotensors::Vector)
Constructor entrance for MPO, which only supports strictly quantum number conserving operators

site tensor convention:
i mean in arrow, o means out arrow
    o 
    |
    2
o-1   3-i
	4
	|
	i
The left and right boundaries are always vacuum.
The case that the right boundary is not vacuum corresponds to operators which do not conserve quantum number, 
such as a†, this case is implemented with another MPO object.
"""
function MPO{T}(mpotensors::Vector) where {T<:Number}
	isempty(mpotensors) && error("no input mpstensors.")
	_check_mpo_space(mpotensors)
	return new{T}(convert(Vector{Array{T, 4}}, mpotensors))
end
end

MPO(mpotensors::Vector{<:MPOTensor{T}}) where {T <: Number} = MPO{T}(mpotensors)

raw_data(h::MPO) = h.data

Base.eltype(::Type{MPO{T}}) where {T} = T
Base.eltype(h::MPO) = eltype(typeof(h))
Base.length(h::MPO) = length(raw_data(h))
Base.isempty(h::MPO) = isempty(raw_data(h))
Base.size(h::MPO, i...) = size(raw_data(h), i...)
Base.getindex(h::MPO, i::Int) = getindex(raw_data(h), i)
Base.lastindex(h::MPO) = lastindex(raw_data(h))
Base.firstindex(h::MPO) = firstindex(raw_data(h))

Base.transpose(mpo::MPO) = MPO([permute(s, (1,4,3,2)) for s in raw_data(mpo)])
Base.adjoint(mpo::MPO) =  MPO([mpo_tensor_adjoint(s) for s in raw_data(mpo)])


function Base.setindex!(h::MPO, v, i::Int)
	return setindex!(raw_data(h), v, i)
end 

Base.copy(h::MPO) = MPO(copy(raw_data(h)))


"""
	r_RR, right boundary 2-tensor
	i-1
	o-2
"""
r_RR(a::MPO, b::MPO) = _eye(promote_type(eltype(a), eltype(b)) , space_r(a), space_r(b)) 
r_RR(a::MPO) = r_RR(a, a)

"""
	l_LL, left boundary 2-tensor
	o-1
	i-2
"""
l_LL(a::MPO, b::MPO) = _eye(promote_type(eltype(a), eltype(b)) , space_l(a), space_l(b)) 
l_LL(a::MPO) = l_LL(a, a)

bond_dimension(h::MPO, bond::Int) = begin
	((bond >= 1) && (bond <= length(h))) || throw(BoundsError())
	return space_r(h[bond])
end 
bond_dimensions(h::MPO) = [bond_dimension(h, i) for i in 1:length(h)]
bond_dimension(h::MPO) = maximum(bond_dimensions(h))

ophysical_dimensions(psi::MPO) = [size(item, 2) for item in raw_data(psi)]
iphysical_dimensions(psi::MPO) = [size(item, 4) for item in raw_data(psi)]


function physical_dimensions(psi::MPO)
	xs = ophysical_dimensions(psi)
	(xs == iphysical_dimensions(psi)) || error("i and o physical dimension mismatch.")
	return xs
end


isstrict(psi::MPO) = (space_l(psi) == space_r(psi) == 1)

mpo_tensor_adjoint(m::MPOTensor) = conj(permute(m, (1,4,3,2)))

