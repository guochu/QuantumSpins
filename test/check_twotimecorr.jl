
function build_models_AB(h1s, h2s, p)
	sp, sm, sz = p["+"], p["-"], p["z"]
	terms = []
	L = length(h1s)
	for i in 1:L
		push!(terms, QTerm(i=>sz, coeff=h1s[i]) )
	end
	h2 = randn(L-1)
	for i in 1:(L-1)
		t = QTerm(i=>sp, i+1=>sm, coeff=h2s[i])
		push!(terms, t)
		push!(terms, t')
	end
	ham = QuantumOperator([terms...])

	sm_op = prodmpo(physical_dimensions(ham), QTerm(1=>sm)) 

	sp_op = prodmpo(physical_dimensions(ham), QTerm(1=>sm)') 

	return ham, sp_op, sm_op
end

function build_open_models_AB(h1s, h2s, p)
	sp, sm, sz = p["+"], p["-"], p["z"]
	terms = []
	L = length(h1s)
	h1 = randn(L)
	for i in 1:L
		push!(terms, QTerm(i=>sz, coeff=h1s[i]) )
	end
	h2 = randn(L-1)
	for i in 1:(L-1)
		t = QTerm(i=>sp, i+1=>sm, coeff=h2s[i])
		push!(terms, t)
		push!(terms, t')
	end
	ham = QuantumOperator([terms...])
	lindblad = superoperator(-im * ham)

	# add_dissipation!(lindblad, QTerm(2=>sz, coeff=0.9))

	sm_op = prodmpo(physical_dimensions(ham), QTerm(1=>sm)) 

	sp_op = prodmpo(physical_dimensions(ham), QTerm(1=>sm)') 

	return lindblad, sp_op, sm_op
end


function check_twotime_corr()
	L = 4
	h1s = [0.2508648902536, -0.6949580074772713, -0.41523333387180145, -0.5090894962618974]
	# h1s = zeros(2)
	h2s = [-0.7153395702446729, -0.08398485202495319, -0.3995503327930961]
	# h2s = ones(1)
	init_state = [0 for i in 1:L]
	init_state[1:2:L] .= 1
	init_state_2 = [0 for i in 1:L]
	init_state_2[2:2:L] .= 1
	state = prodmps(ComplexF64, [2 for i in 1:L], init_state) + prodmps(ComplexF64, [2 for i in 1:L], init_state_2)
	canonicalize!(state, normalize=true)

	p = spin_half_matrices()
	ts = [0., 0.05, 0.2, 0.56]
	stepsize = 0.01
	
	function compare_nsym_corr(reverse::Bool)
		h, sp_op, sm_op, = build_models_AB(h1s, h2s, p)
		corr = correlation_2op_1t(h, sp_op, sm_op, copy(state), ts, stepper=TEBDStepper(stepsize=stepsize), reverse=reverse)
		return corr
	end

	function compare_open_nsym_corr(reverse::Bool)
		h, sp_op, sm_op = build_open_models_AB(h1s, h2s, p)
		# rho = increase_bond!(DensityOperator(state), D=20)
		# canonicalize!(rho)
		rho = DensityOperator(state)
		corr = correlation_2op_1t(h, sp_op, sm_op, rho, ts, stepper=TDVPStepper(D=20, stepsize=stepsize), reverse=reverse)
		return corr
	end

	err1 = maximum(abs.(compare_nsym_corr(true) - compare_open_nsym_corr(true)))
	err2 = maximum(abs.(compare_nsym_corr(false) - compare_open_nsym_corr(false)))

	return err1 <= 1.0e-3 && err2 <= 1.0e-3
end


@testset "two-time correlation" begin
	@test check_twotime_corr()
end