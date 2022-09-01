
abstract type AbstractMatrixFactorization end
struct QRFact <: AbstractMatrixFactorization end
struct SVDFact{T<:TruncationScheme} <: AbstractMatrixFactorization 
	trunc::T
end
SVDFact(; trunc::TruncationScheme=NoTruncation()) = SVDFact(trunc)
