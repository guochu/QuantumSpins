



struct OverlapCache{_A, _B, _C} <: AbstractCache
	A::_A
	B::_B
	cstorage::_C
end

Base.length(x::OverlapCache) = length(x.A)
scalar_type(x::OverlapCache) = promote_type(scalar_type(x.A), scalar_type(x.B))

bra(x::OverlapCache) = x.A
ket(x::OverlapCache) = x.B

function environments(psiA::M, psiB::M) where {M <: Union{AbstractMPS, MPO}}
	(length(psiA) == length(psiB)) || throw(DimensionMismatch())
	(space_r(psiA) == space_r(psiB)) || throw(SpaceMismatch())
	hold = r_RR(psiA, psiB)
	L = length(psiA)
	cstorage = Vector{typeof(hold)}(undef, L+1)
	cstorage[L+1] = hold
	cstorage[1] = l_LL(psiA, psiB)
	for i in L:-1:2
		cstorage[i] = updateright(cstorage[i+1], psiA[i], psiB[i])
	end
	return OverlapCache(psiA, psiB, cstorage)
end

environments(psi::DensityOperatorMPS) = environments(psi.I, psi.data)



