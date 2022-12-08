
abstract type AbstractMatrixFactorization end
struct QRFact <: AbstractMatrixFactorization 
	normalize::Bool
end
QRFact(; normalize::Bool=false) = QRFact(normalize)
struct SVDFact{T<:TruncationScheme} <: AbstractMatrixFactorization 
	trunc::T
	normalize::Bool
end
SVDFact(; trunc::TruncationScheme=NoTruncation(), normalize::Bool=false) = SVDFact(trunc, normalize)
