




struct MPSBondView{M}
	parent::M
end


Base.length(m::MPSBondView) = length(raw_singular_matrices(m.parent))
Base.firstindex(m::MPSBondView) = firstindex(raw_singular_matrices(m.parent))
Base.lastindex(m::MPSBondView) = lastindex(raw_singular_matrices(m.parent))
# Base.getindex(m::MPSBondView, i::Integer) = getindex(raw_singular_matrices(m.parent), i)
Base.getindex(m::MPSBondView, i::Int) = raw_singular_matrices(m.parent)[i] 
function Base.setindex!(m::MPSBondView, v, i::Integer)
	L = length(m.parent)
	(1 < i <= L) || throw(BoundsError(m.parent.svectors, i))
	return setindex!(raw_singular_matrices(m.parent), v, i)
end
