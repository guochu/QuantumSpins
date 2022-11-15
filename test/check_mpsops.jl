

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