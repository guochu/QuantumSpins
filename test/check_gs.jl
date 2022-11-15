

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