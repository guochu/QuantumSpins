push!(LOAD_PATH, dirname(Base.@__DIR__) * "/src")

using Test
using QuantumSpins
using KrylovKit
using TensorOperations
using LinearAlgebra: Diagonal


println("-----------test elementary tensor operations-----------------")

function check_svd()
	m = randn(ComplexF64, 3,5)
	u, s, v, = tsvd!(copy(m))
	return u * Diagonal(s) * v ≈ m
end

function check_tsvd()
	m = randn(ComplexF64, 3,5, 7)
	u, s, v, = tsvd!(copy(m), (1,3), (2,))
	sm = Matrix(Diagonal(s))
	@tensor tmp[-1,-2,-3] := u[-1,-3,1] * sm[1,2] * v[2,-2]
	return tmp ≈ m
end

function check_qr()
	m = randn(ComplexF64, 3,5, 7)
	u, v, = tqr!(copy(m), (1,3), (2,))
	@tensor tmp[-1,-2,-3] := u[-1,-3,1] * v[1,-2]
	return tmp ≈ m	
end

function check_lq()
	m = randn(ComplexF64, 3,5, 7)
	u, v, = tlq!(copy(m), (1,3), (2,))
	@tensor tmp[-1,-2,-3] := u[-1,-3,1] * v[1,-2]
	return tmp ≈ m	
end

function check_exp()
	m = randn(ComplexF64, 3,3,2,2)
	m2 = texp(m, (1,3), (2,4))
	return tie(m2, (2,2)) ≈ exp( Matrix(tie(permute(m, (1,3,2,4)), (2,2))) )
end

@testset "elementary tensor operations" begin
	@test check_svd()
	@test check_tsvd()
	@test check_qr()
	@test check_lq()
	@test check_exp()
end


println("-----------test mps operations-----------------")

function check_canonical()
	mps = randommps(ComplexF64, 7, d=2, D=3)
	canonicalize!(mps, normalize=true)
	return iscanonical(mps)
end

function check_prodmps()
	mps = prodmps(ComplexF64, [2 for i in 1:5], rand(0:1, 5))
	return (norm(mps) ≈ 1) && bond_dimension(mps) == 1
end

function check_mps_add()
	a = prodmps(ComplexF64, [2 for i in 1:3], [1,0,1])
	b = prodmps(ComplexF64, [2 for i in 1:3], [0,1,1])

	m = a - b
	s1 = norm(m) ≈ sqrt(2)
	normalize!(m)
	return s1 && (dot(m, a) ≈ sqrt(2) / 2) && (dot(m, b) ≈ -sqrt(2) / 2) && (norm(m * 2) ≈ 2)
end

function check_densityop()
	mps = normalize!(randommps(ComplexF64, 5, d=2, D=2))
	rho = DensityOperator(mps)
	mpo = MPO(rho)
	rho2 = DensityOperator(mpo)
	# println(expectation(mpo, mps))
	return (tr(rho) ≈ 1) && (tr(mpo) ≈ 1) && (expectation(mpo, mps) ≈ 1) && (rho ≈ rho2)
end

function check_densityop2()
	mps = normalize!(randommps(ComplexF64, 5, d=2, D=2))
	mpo = randommpo(5, d=2, D=2)
	rho = DensityOperator(mps)
	return expectation(mpo, mps) ≈ expectation(kron(mpo, id(mpo)), rho)
end

@testset "mps operations" begin
	@test check_canonical()
	@test check_prodmps()
	@test check_mps_add()
	@test check_densityop()
	@test check_densityop2()
end


println("-----------test mps algorithms-----------------")

function check_gs()
	L = 5
	J = 1.
	Jzz = 1.2
	hz = 0.3

	ham = heisenberg_chain(L,J=J,Jzz=Jzz,hz=hz)

	mpo = MPO(ham)
	mps = randommps(L, d=2, D=10)
	canonicalize!(mps, normalize=true)
	energies, delta = ground_state!(mps, mpo, DMRG(verbosity=0))

	mat = matrix(ham)
	eigvals, vecs = eigsolve(mat, randn(eltype(mat), size(mat, 1)), 1, :SR, Lanczos())
	return isapprox(energies[end], eigvals[1], atol=1.0e-8)
end

@testset "ground state" begin
	@test check_gs()
end

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
	state = increase_bond!(prodmps(ComplexF64, [2 for i in 1:L], init_state), D=20)
	canonicalize!(state)

	obs2 = Vector{Float64}[]
	cache = timeevo_cache(ham, TDVPStepper(stepsize=stepsize, tspan=(0, -im*delta), ishermitian=true), state)
	push!(obs2, [real(expectation(item, state, iscanonical=true)) for item in observers])
	for i in 1:10
		stepper = TDVPStepper(stepsize=stepsize, tspan=(-im*(i-1)*delta, -im*i*delta), ishermitian=true)
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

	observers = [superoperator(QTerm(i=>p["z"]), id(QTerm(i=>p["z"]))) for i in 1:L]

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
		rho = increase_bond!(DensityOperator(state), D=20)
		canonicalize!(rho)
		corr = correlation_2op_1t(h, sp_op, sm_op, rho, ts, stepper=TDVPStepper(stepsize=stepsize), reverse=reverse)
		return corr
	end

	err1 = maximum(abs.(compare_nsym_corr(true) - compare_open_nsym_corr(true)))
	err2 = maximum(abs.(compare_nsym_corr(false) - compare_open_nsym_corr(false)))

	return err1 <= 1.0e-3 && err2 <= 1.0e-3
end


@testset "two-time correlation" begin
	@test check_twotime_corr()
end

