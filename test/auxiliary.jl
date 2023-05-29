println("------------------------------------")
println("|            auxiliary             |")
println("------------------------------------")


@testset "Coefficients" begin
	a = Coefficient(1.2)
	b = a * 3.5
	@test eltype(a) <: Number
	@test eltype(b) <: Number
	@test typeof(a + 1) == typeof(a)
	@test eltype(1.3 + a) <: Real
	@test eltype(a * 3.11) <: Real
	@test eltype(3.11 * a) <: Real
	@test eltype(a / 3.1) <: Real
	@test eltype(3.1 / a) <: Real
	@test value(b) == value(a) * 3.5
	@test value(-a) == -value(a)
	@test a(0.1) == 1.2
	@test value(a / 2) == value(a) / 2
	@test value(3.7 / a) == 3.7 / value(a)
	@test conj(Coefficient(3.1+5*im)) == Coefficient(3.1-5*im)
	@test iszero(Coefficient(0))
	@test isconstant(a)
	f = Coefficient(sin)
	@test Coefficient(sin) == f
	@test !(f == a)
	@test !isconstant(f)
	@test f(1.7) == sin(1.7)
	@test (1 / f)(2.3) == 1 / f(2.3)
	@test (f / 2.4)(1.5) == f(1.5) / 2.4
	@test sqrt(f)(1.6) ≈ sqrt(f(1.6))
	@test conj(f(3+4*im)) ≈ conj(f)(3+4*im)
	@test (2.4 / f)(1.7) == 2.4 / f(1.7)
	@test (-f)(1.4) == -(f(1.4))
	g = Coefficient(cos)
	@test (f + g)(1.3) == f(1.3) + g(1.3)
	@test (f / g)(2.7) ≈ f(2.7) / g(2.7)
	@test (f - g)(1.4) ≈ f(1.4) - g(1.4)
end

@testset "Deparallel" begin
	a = rand(20, 10) .* 1.0e-14
	b, c = leftdeparallel(a)
	@test size(b, 1) == size(a, 1)
	@test size(c, 2) == size(a, 2)
	@test isempty(b)
	@test isempty(c)

	a = rand(20, 10) .* 1.0e-14
	b, c = rightdeparallel(a)
	@test size(b, 1) == size(a, 1)
	@test size(c, 2) == size(a, 2)
	@test isempty(b)
	@test isempty(c)


	a = randn(8, 5)
	a[:, 3] .= a[:, 1]
	a[:, 5] .= a[:, 1]
	a[:, 4] .= a[:, 2] .+ 3.4
	b, c = leftdeparallel(a)
	@test size(b, 1) == size(a, 1)
	@test size(c, 2) == size(a, 2)
	@test size(b, 2) == size(c, 1)
	@test size(b, 2) <= 3
	@test b * c ≈ a

	a = randn(8, 5)
	b, c = rightdeparallel(a)
	@test size(b, 1) == size(a, 1)
	@test size(c, 2) == size(a, 2)
	@test size(b, 2) == size(c, 1)
	@test b * c ≈ a
	
	a = randn(5, 10)
	a[1, :] .= 1
	a[4, :] .= 1
	a[3, :] .= 1 + 1.0e-14
	a[5, :] .= 1	
	b, c = rightdeparallel(a)
	@test size(b, 1) == size(a, 1)
	@test size(c, 2) == size(a, 2)
	@test size(b, 2) == size(c, 1)
	@test size(b, 2) <= 2
	@test b * c ≈ a

end
