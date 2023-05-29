println("------------------------------------")
println("|       tensor operations          |")
println("------------------------------------")

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