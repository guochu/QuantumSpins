


println("-----------test mps algorithms-----------------")


function create_random_mpo_mps(::Type{T}, L, d, D1, D2) where T
    mpo = randommpo(T, L, d=d, D=D1)
    mps = randommps(T, L, d=d, D=D2)
    for i in 1:L
    	mpo[i] /= D1
    end
    for i in 1:L
    	mps[i] /= D2
    end    
    return mpo, mps
end

function test_iterative_mult(::Type{T}, L::Int) where T
	d = 2
	mpo, mps = create_random_mpo_mps(T, L, d, 2, 3)

	mps_exact = mpo * mps
	mps_iterative, err = iterative_mult(mpo, mps, IterativeArith(D=6, verbosity=0))

	return distance(mps_exact, mps_iterative) <= 1.0e-6
end

function test_svd_mult(::Type{T}, L::Int) where T
	d = 2
	mpo, mps = create_random_mpo_mps(T, L, d, 2, 3)

	mps_exact = mpo * mps
	mps_iterative, err = svd_mult(mpo, mps, SVDArith(verbosity=0))

	return distance(mps_exact, mps_iterative) <= 1.0e-6
end

function test_stable_mult(::Type{T}, L::Int) where T
	d = 2
	mpo, mps = create_random_mpo_mps(T, L, d, 2, 3)

	mps_exact = mpo * mps
	mps_iterative, err = stable_mult(mpo, mps, StableArith(D=6, verbosity=0))

	return distance(mps_exact, mps_iterative) <= 1.0e-6
end

function check_mpsadd(::Type{T}, L::Int) where T
	a = randommps(T, L, d=2, D=2)
	b = randommps(T, L, d=2, D=3)

	c1 = a + b
	c2, err = svd_add(a, b, SVDArith(D=5))
	c3, err = iterative_add(a, b, IterativeArith(D=5))

	# println(distance(c1, c2))
	# println(distance(c1, c3))
	return max(distance(c1, c2), distance(c1, c3)) < 1.0e-5

end

function check_mpsadd_2(::Type{T}, L::Int) where T
	a1 = randommps(T, L, d=2, D=2)
	a2 = randommps(T, L, d=2, D=3)
	a3 = randommps(T, L, d=2, D=2)

	c1, err = svd_add([a1,a2,a3], SVDArith(D=7))
	c2, err = iterative_add([a1,a2,a3], IterativeArith(D=7))
	c3, err = stable_add([a1,a2,a3], StableArith(D=7))

	return max(distance(c1, c2), distance(c1, c3)) < 1.0e-5
end

function check_mpompo_iterative_mult(::Type{T}, L::Int) where T
	dx = [2 for i in 1:L]
	dy = [3 for i in 1:L]

	mpo = randommpo(T, dy, dx, D=2)
	A = randommpo(T, dx, dx, D=3)

	mpo_exact = mpo * A
	mpo_iterative, err = iterative_mult(mpo, A, IterativeArith(D=6, verbosity=0))
	return distance(mpo_exact, mpo_iterative)  / norm(mpo_exact) <= 1.0e-6
end

# println("-----------test mpo mps iterative mult-----------------")

@testset "mpo mps/mpo iterative mult" begin
	@test test_iterative_mult(Float64, 7)
	@test test_iterative_mult(ComplexF64, 6)
	@test test_svd_mult(Float64, 5)
	@test test_svd_mult(ComplexF64, 4)
	@test test_stable_mult(Float64, 3)
	@test test_stable_mult(ComplexF64, 8)
	@test check_mpsadd(ComplexF64, 5)
	@test check_mpsadd_2(Float64, 7)
	@test check_mpompo_iterative_mult(ComplexF64, 6)
	@test check_mpompo_iterative_mult(Float64, 7)
end
