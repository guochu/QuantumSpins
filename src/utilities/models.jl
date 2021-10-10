

"""
    heisenberg xxz chain
"""
function heisenberg_chain(L::Int; J::Real, Jzz::Real, hz::Real)
	p = spin_half_matrices()
    sp, sm, z = p["+"], p["-"], p["z"]
    terms = []
    # one site terms
    for i in 1:L
        push!(terms, QTerm(i=>z, coeff=hz))
    end
    # nearest-neighbour interactions
    for i in 1:L-1
        t = QTerm(i=>sp, i+1=>sm, coeff=2*J)
        push!(terms, t)
        push!(terms, t')
        push!(terms, QTerm(i=>z, i+1=>z, coeff=Jzz))
    end
    return QuantumOperator([terms...])
end


function boundary_driven_xxz(L::Int; J::Real, Jzz::Real, hz::Real, nl::Real, Λl::Real, nr::Real, Λr::Real, Λp::Real)
	p = spin_half_matrices()
	(nl >=0 && nl <=1) || error("nl must be between 0 and 1.")
	(nr >=0 && nr <=1) || error("nr must be between 0 and 1.")
	(Λl >=0 && Λr >= 0 && Λp>= 0) || error("Λ should not be negative.")
	sp, sm, sz = p["+"], p["-"], p["z"]
	lindblad = superoperator(heisenberg_chain(L; J=J, Jzz=Jzz, hz=hz))

	gammal_plus = Λl*nl
	gammal_minus = Λl*(1-nl)
	gammar_plus = Λr*nr
	gammar_minus = Λr*(1-nr)

	for i in 1:L
		add_dissipation!(lindblad, QTerm(i=>sz, coeff=sqrt(Λp)))
	end
	add_dissipation!(lindblad, QTerm(1=>sp, coeff=sqrt(gammal_plus)))
	add_dissipation!(lindblad, QTerm(1=>sm, coeff=sqrt(gammal_minus)))
	add_dissipation!(lindblad, QTerm(L=>sp, coeff=sqrt(gammar_plus)))
	add_dissipation!(lindblad, QTerm(L=>sm, coeff=sqrt(gammar_minus)))
	return lindblad
end


