
abstract type AbstractMatrixFactorization end
struct QRFact <: AbstractMatrixFactorization end
struct SVDFact{T<:TruncationScheme} <: AbstractMatrixFactorization 
	trunc::T
	normalize::Bool
end
SVDFact(; trunc::TruncationScheme=NoTruncation(), normalize::Bool=false) = SVDFact(trunc, normalize)
