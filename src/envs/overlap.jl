



struct OverlapCache{_A<:MPS, _B<:MPS, _C} <: AbstractCache
	A::_A
	B::_B
	cstorage::_C
end


function environments(psiA::MPS, psiB::MPS)
	(length(psiA) == length(psiB)) || throw(DimensionMismatch())
	(space_r(psiA) == space_r(psiB)) || throw(SpaceMismatch())
	hold = r_RR(psiA, psiB)
	L = length(psiA)
	cstorage = Vector{typeof(hold)}(undef, L+1)
	cstorage[L+1] = hold
	for i in L:-1:1
		cstorage[i] = updateright(cstorage[i+1], psiA[i], psiB[i])
	end
	return OverlapCache(psiA, psiB, cstorage)
end

environments(psi::DensityOperatorMPS) = environments(psi.I, psi.data)



