
function LinearAlgebra.dot(hA::MPO, hB::MPO) 
	(length(hA) == length(hB)) || throw(DimensionMismatch())
    hold = l_LL(hA)
    for i in 1:length(hA)
        hold = updateleft(hold, hA[i], hB[i])
    end
    return tr(hold)
end
LinearAlgebra.norm(h::MPO) = sqrt(real(dot(h, h)))
distance2(hA::MPO, hB::MPO) = _distance2(hA, hB)
distance(hA::MPO, hB::MPO) = _distance(hA, hB)


# get identity operator

"""
    id(m::MPO)
Retuen an identity MPO from a given MPO
"""
id(m::MPO) = MPO([reshape(_eye(eltype(m), size(item, 2)), 1, size(item, 2), 1, size(item, 4)) for item in raw_data(m)])


l_tr(h::Union{MPO, DensityOperatorMPS}) = ones(eltype(h), 1)

function LinearAlgebra.tr(h::MPO)
    isempty(h) && return 0.
    L = length(h)
    hold = l_tr(h)
    for i in 1:L
        hold = updatetraceleft(hold, h[i])
    end
    return scalar(hold)
end

function Base.:*(h::MPO, f::Number)
	T = coerce_scalar_type(eltype(h), typeof(f))
	r = MPO{T}(copy(raw_data(h)))
	r[1] *= convert(T, f)
	return r
end 
Base.:*(f::Number, h::MPO) = h * f
Base.:/(h::MPO, f::Number) = h * (1/f)



"""
    Base.:+(hA::M, hB::M) where {M <: MPO}
    addition of two MPOs
"""
function Base.:+(hA::MPO, hB::MPO)
    @assert !isempty(hA)
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    T = promote_type(eltype(hA), eltype(hB))
    L = length(hA)
    r = Vector{Array{T, 4}}(undef, L)
    r[1] = cat(hA[1], hB[1], dims=3)
    r[L] = cat(hA[L], hB[L], dims=1)
    for i in 2:L-1
    	r[i] = cat(hA[i], hB[i], dims=(1,3))
    end
    return MPO(r)
end
# adding mpo with adjoint mpo will return an normal mpo
Base.:-(hA::MPO, hB::MPO) = hA + (-1) * hB
Base.:-(h::MPO) = -1 * h

"""
    Base.:*(h::MPO, psi::MPS)
    Base.:*(h::MPO, psi::FiniteDensityOperatorMPS)
    Multiplication of mps by an mpo.
"""
function Base.:*(h::MPO, psi::MPS)
    @assert !isempty(h)
    (length(h) == length(psi)) || throw(DimensionMismatch())
    r = [@tensor tmp[-1 -2; -3 -4 -5] := a[-1, -3, -4, 1] * b[-2, 1, -5] for (a, b) in zip(raw_data(h), raw_data(psi))]
    return MPS([tie(item,(2,1,2)) for item in r])
end
Base.:*(h::MPO, psi::DensityOperatorMPS) = DensityOperatorMPS(h * psi.data, psi.fusers, psi.I)


"""
    Base.:*(hA::M, hB::M) where {M <: MPO}
    a * b
"""
Base.:*(hA::MPO, hB::MPO) = MPO(_mult_n_n(raw_data(hA), raw_data(hB)))

function _mult_n_n(hA::Vector{<:MPOTensor}, hB::Vector{<:MPOTensor})
    @assert !isempty(hA)
    (length(hA) == length(hB)) || throw(DimensionMismatch())
    r = [@tensor tmp[-1 -2 -3; -4 -5 -6] := aj[-1, -3, -4, 1] * bj[-2, 1, -5, -6] for (aj, bj) in zip(hA, hB)]
    return [tie(item, (2,1,2,1)) for item in r]
end

const MPO_APPROX_EQUAL_ATOL = 1.0e-12

"""
    Base.isapprox(a::M, b::M) where {M <: MPO} 
    Check is two MPOs are approximated equal 
"""
Base.isapprox(a::MPO, b::MPO; atol=MPO_APPROX_EQUAL_ATOL) = distance2(a, b) <= atol

r_RR(psiA::MPS, h::MPO, psiB::MPS) = reshape(_eye(promote_type(eltype(psiA), eltype(h), eltype(psiB)), space_r(psiA), space_r(h) * space_r(psiB)),
	space_r(psiA), space_r(h), space_r(psiB))


l_LL(psiA::MPS, h::MPO, psiB::MPS) = reshape(_eye(promote_type(eltype(psiA), eltype(h), eltype(psiB)), space_l(psiA), space_l(h) * space_l(psiB)),
	space_l(psiA), space_l(h), space_l(psiB))


"""
    expectation(psiA::MPS, h::MPO, psiB::MPS)
    expectation(h::MPO, psi::MPS) = expectation(psi, h, psi)
    expectation(h::MPO, psi::FiniteDensityOperatorMPS) = expectation(psi.I, h, psi.data)
compute < psiA | h | psiB >
"""
function expectation(psiA::MPS, h::MPO, psiB::MPS) 
    (length(psiA) == length(h) == length(psiB)) || throw(DimensionMismatch())
    hold = r_RR(psiA, h, psiB)
    for i in length(psiA):-1:1
        hold = updateright(hold, psiA[i], h[i], psiB[i])
    end
    return scalar(hold)
end
expectation(h::MPO, psi::MPS) = expectation(psi, h, psi)


function LinearAlgebra.ishermitian(h::MPO)
    @assert !isempty(h)
    return isapprox(h, h', atol=1.0e-10) 
end


MPO(psi::DensityOperatorMPS) = MPO([@tensor o[-1 -2; -3 -4] := psi[i][-1,1,-3]*psi.fusers[i][-2,-4,1] for i in 1:length(psi)])
function DensityOperator(h::MPO)
    T = eltype(h)
    # fusers = [reshape(_eye(T, size(m, 2) * size(m, 4)), size(m, 2), size(m, 4), size(m, 2) * size(m, 4)) for m in raw_data(h)]
    ds = physical_dimensions(h)
    fusers = default_fusers(T, ds)
    mps = MPS([@tensor o[-1 -2; -3] := m[-1,1,-3,2] * conj(fj[1,2,-2])  for (fj, m) in zip(fusers, raw_data(h))])
    return DensityOperatorMPS(mps, fusers, identity_mps(T, ds))
end

expectation(h::MPO, psi::DensityOperatorMPS) = expectation(psi.I, h, psi.data)

# kronecker product between MPOs
Base.kron(x::MPO, y::MPO) = MPO([rkron(a, b) for (a, b) in zip(raw_data(x), raw_data(y))])
