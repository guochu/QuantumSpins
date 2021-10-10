
"""
	assume the underlying state is canonical
"""
function expectation_canonical(m::QTerm, psi::MPS)
	is_constant(m) || throw(ArgumentError("only constant QTerm allowed."))
	is_zero(m) && return 0.
	L = length(psi)
	pos = positions(m)
	ops = op(m)
	pos_end = pos[end]
	(pos_end <= L) || throw(BoundsError())
	util = ones(1)
	@tensor hold[-1; -3] := conj(psi[pos_end][-1, 1, 2]) * psi[pos_end][-3, 4, 2] * ops[end][1, 4] 
	for j in pos_end-1:-1:pos[1]
		pj = findfirst(x->x==j, pos)
		if isnothing(pj)
			hold = updateright(hold, psi[j], pj, psi[j])
		else
			hold = updateright(hold, psi[j], ops[pj], psi[j])
		end
	end	 
	s = psi.s[pos[1]]
	hnew = Diagonal(s) * hold * Diagonal(s)
 	return tr(hnew) * value(coeff(m))
end

function expectation(psiA::MPS, m::QTerm, psiB::MPS, envs::OverlapCache=environments(psiA, psiB))
	cstorage = envs.cstorage
	(length(psiA) == length(psiB) == length(cstorage)-1) || throw(DimensionMismatch())
	is_constant(m) || throw(ArgumentError("only constant QTerm allowed."))
	is_zero(m) && return 0.
	L = length(psiA)
	pos = positions(m)
	ops = op(m)
	pos_end = pos[end]
	(pos_end <= L) || throw(BoundsError())
	@tensor hold[-1; -3] := conj(psiA[pos_end][-1, 1, 2]) * cstorage[pos_end+1][2, 3] * psiB[pos_end][-3, 5, 3] * ops[end][1, 5]
	for j in pos_end-1:-1:1
		pj = findfirst(x->x==j, pos)
		if isnothing(pj)
			hold = updateright(hold, psiA[j], pj, psiB[j])
		else
			hold = updateright(hold, psiA[j], ops[pj], psiB[j])
		end
	end
	return scalar(hold) * value(coeff(m))
end

expectation(m::QTerm, psi::MPS; iscanonical::Bool=false) = iscanonical ? expectation_canonical(m, psi) : expectation(psi, m, psi)

function expectation(psiA::MPS, h::QuantumOperator, psiB::MPS)
	(length(h) <= length(psiA)) || throw(DimensionMismatch())
	envs = environments(psiA, psiB)
	r = 0.
	for m in qterms(h)
		r += expectation(psiA, m, psiB, envs)
	end
	return r
end
function expectation(h::QuantumOperator, psi::MPS; iscanonical::Bool=false)
	if iscanonical
		r = 0.
		for m in qterms(h)
			r += expectation_canonical(m, psi)
		end
		return r
	else
		return expectation(psi, h, psi)
	end
end

# density operator
"""
	expectation(m::QTerm, psi::FiniteDensityOperatorMPS) 
	⟨I|h|ρ⟩
"""
expectation(m::QTerm, psi::DensityOperatorMPS, envs=environments(psi)) = expectation(psi.I, m, psi.data, envs)

function expectation(m::QuantumOperator, psi::DensityOperatorMPS)
	envs = environments(psi)
	r = 0.
	for m in qterms(h)
		r += expectation(m, psi, envs)
	end
	return r
end
