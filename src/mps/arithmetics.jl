

function LinearAlgebra.dot(psiA::AbstractMPS, psiB::AbstractMPS) 
	(length(psiA) == length(psiB)) || throw(ArgumentError("dimension mismatch."))
    hold = l_LL(psiA)
    for i in 1:length(psiA)
        hold = updateleft(hold, psiA[i], psiB[i])
    end
    return tr(hold)
end

function LinearAlgebra.norm(psi::AbstractMPS; iscanonical::Bool=false) 
	iscanonical ? norm(psi[1]) : sqrt(real(dot(psi, psi)))
end



"""
    distance(a, b)
Square of Euclidean distance between a and b.
"""
distance2(a::AbstractMPS, b::AbstractMPS) = _distance2(a, b)

"""
    distance(a, b)
Euclidean distance between a and b.
"""
distance(a::AbstractMPS, b::AbstractMPS) = _distance(a, b)


function LinearAlgebra.normalize!(psi::AbstractPureStateMPS; iscanonical::Bool=false)
    n = norm(psi, iscanonical=iscanonical)
    (n ≈ zero(n)) && @warn "quantum state has zero norm."
    if n != one(n)
        if iscanonical
            psi[1] = psi[1] / n
        else
            factor = n^(1 / length(psi))
            for i in 1:length(psi)
                psi[i] = psi[i] / factor
            end
        end  
    end
    return psi
end
LinearAlgebra.normalize(psi::AbstractPureStateMPS; iscanonical::Bool=false) = normalize!(copy(psi); iscanonical=iscanonical)

function Base.:*(psi::MPS, f::Number)
	T = coerce_scalar_type(scalar_type(psi), typeof(f)) 
	r = MPS{T}(copy(raw_data(psi)), copy(raw_singular_matrices(psi)))
	r[1] *= convert(T, f)
	return r
end 
Base.:*(f::Number, psi::MPS) = psi * f
Base.:/(psi::MPS, f::Number) = psi * (1/f)

function LinearAlgebra.normalize!(psi::DensityOperatorMPS)
    n = tr(psi)
    (n ≈ zero(n)) && @warn "density operator has zero trace."
    if n != one(n)
        factor = n^(1 / length(psi))
        for i in 1:length(psi)
            psi[i] = psi[i] / factor
        end
    end
    return psi
end


"""
    Base.:+(psiA::MPS, psiB::MPS) 
Addition of two MPSs
"""
function Base.:+(psiA::MPS, psiB::MPS) 
    (length(psiA) == length(psiB)) || throw(DimensionMismatch())
    (isempty(psiA)) && error("input mps is empty.")
    if length(psiA) == 1
        return MPS([psiA[1] + psiB[1]])
    end
    L = length(psiA)
    T = promote_type(scalar_type(psiA), scalar_type(psiB))
    r = Vector{Array{T, 3}}(undef, L)
    r[1] = cat(psiA[1], psiB[1], dims=3)
    r[L] = cat(psiA[L], psiB[L], dims=1)
    for i in 2:L-1
    	r[i] = cat(psiA[i], psiB[i], dims=(1,3))
    end
    return MPS(r)
end
Base.:-(psiA::MPS, psiB::MPS) = psiA + (-1) * psiB
Base.:-(psi::MPS) = -1 * psi

const MPS_APPROX_EQUAL_ATOL = 1.0e-14
Base.isapprox(psiA::MPS, psiB::MPS; atol=MPS_APPROX_EQUAL_ATOL) = distance2(psiA, psiB) <= atol

function infinite_temperature_state(::Type{T}, ds::Vector{Int}) where {T <: Number}
	iden = identity_mps(T, ds)
	state = DensityOperatorMPS(copy(iden), default_fusers(T, ds), iden)
	return normalize!(state)
end

rkron(x::AbstractArray, y::AbstractArray) = kron(y, x)

function Base.kron(x::MPS, y::MPS; trunc::TruncationScheme=DefaultTruncation)
    (length(x) == length(y)) || throw(DimensionMismatch())
    isempty(x) && throw(ArgumentError("input is empty."))
    T = promote_type(scalar_type(x), scalar_type(y))
    L = length(x)
    workspace = T[]

    # first site
    m = rkron(x[1], y[1])
    # errs = Float64[]
    u, s, v, err = tsvd!(tie(m, (2, 1)), workspace, trunc=trunc)
    # push!(errs, err)
    r = Vector{typeof(m)}(undef, L)
    r[1] = reshape(u, size(m, 1), size(m, 2), length(s))
    v = Diagonal(s) * v
    # middle sites
    for i in 2:L-1
        m = rkron(x[i], y[i])
        m = reshape(v * tie(m, (1, 2)), size(v, 1), size(m, 2), size(m, 3))
        u, s, v, err = tsvd!(tie(m, (2, 1)), workspace, trunc=trunc)
        # push!(errs, err)
        r[i] = reshape(u, size(m, 1), size(m, 2), length(s))
        v = Diagonal(s) * v
    end

    # last site
    m = rkron(x[L], y[L])
    m = reshape(v * tie(m, (1, 2)), size(v, 1), size(m, 2), size(m, 3))
    r[L] = m
    mpsout = MPS(r)
    rightorth!(mpsout, workspace, alg=SVDFact(trunc=trunc))
    return mpsout
end

function DensityOperator(psi::MPS; kwargs...)
    T = scalar_type(psi)
    rho = kron(psi, psi'; kwargs...)
    ds = physical_dimensions(psi)
    return DensityOperatorMPS(rho, default_fusers(T, ds), identity_mps(T, ds))
end 

