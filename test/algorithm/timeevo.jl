


function check_unitary_timeevo()
	L = 5
	J = 1.
	Jzz = 1.2
	hz = 0.3
	stepsize = 0.02

	ham = heisenberg_chain(L,J=J,Jzz=Jzz,hz=hz)
	p = spin_half_matrices()
	observers = [QTerm(i=>p["z"]) for i in 1:L]

	init_state = [0 for i in 1:L]
	init_state[1:2:L] .= 1

	state = prodmps(ComplexF64, [2 for i in 1:L], init_state)
	obs1 = Vector{Float64}[]
	canonicalize!(state)

	delta = 0.1

	# TEBD
	cache = timeevo_cache(ham, TEBDStepper(stepsize=stepsize, tspan=(0, -im*delta), order=2), state)
	push!(obs1, [real(expectation(item, state, iscanonical=true)) for item in observers])
	for i in 1:10
		stepper = TEBDStepper(stepsize=stepsize, tspan=(-im*(i-1)*delta, -im*i*delta), order=4)
		state, cache = timeevo!(state, ham, stepper, cache)
		push!(obs1, [real(expectation(item, state, iscanonical=true)) for item in observers])
	end
	

	# TDVP
	# state = increase_bond!(prodmps(ComplexF64, [2 for i in 1:L], init_state), D=20)
	state = prodmps(ComplexF64, [2 for i in 1:L], init_state)
	canonicalize!(state)

	obs2 = Vector{Float64}[]
	cache = timeevo_cache(ham, TDVPStepper(D=20, stepsize=stepsize, tspan=(0, -im*delta), ishermitian=true), state)
	push!(obs2, [real(expectation(item, state, iscanonical=true)) for item in observers])
	for i in 1:10
		stepper = TDVPStepper(D=20, stepsize=stepsize, tspan=(-im*(i-1)*delta, -im*i*delta), ishermitian=true)
		state, cache = timeevo!(state, ham, stepper, cache)
		push!(obs2, [real(expectation(item, state, iscanonical=false)) for item in observers])
	end

	err = maximum(abs.( obs1[end] - obs2[end] ))
	return err <= 1.0e-3
end

function check_open_timeevo()
	L = 5
	J = 1.
	Jzz = 1.1
	hz = 0.3

	nl = 0.6
	gammal = 1.1

	nr = 0.3
	gammar = 1.3

	gammaphase = 0.1
	t = 0.5
	stepsize = 0.02


	p = spin_half_matrices()

	lindblad1 = boundary_driven_xxz(L, J=J, Jzz=Jzz, hz=hz, nl=nl, Λl=gammal, nr=nr, Λr=gammar, Λp=gammaphase)

	observers = [superterm(QTerm(i=>p["z"]), id(QTerm(i=>p["z"]))) for i in 1:L]

	init_state = [0 for i in 1:L]
	init_state[2:2:L] .= 1

	psi = prodmps(ComplexF64, [2 for i in 1:L], init_state)
	rho = DensityOperator(psi)
	rho, cache = timeevo!(rho, lindblad1, TEBDStepper(tspan=(0, t), stepsize=stepsize, order=4))
	obs1 = [real(expectation(item, rho)) for item in observers]


	rho = increase_bond!(DensityOperator(psi), D=20)
	canonicalize!(rho)
	rho, cache = timeevo!(rho, MPO(lindblad1), TDVPStepper(tspan=(0, t), stepsize=stepsize))
	obs2 = [real(expectation(item, rho)) for item in observers]

	err = maximum(abs.( (obs1 - obs2)))
	return err <= 1.0e-3
end

@testset "time evolution" begin
	@test check_unitary_timeevo()
	@test check_open_timeevo()
end

