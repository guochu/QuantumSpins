
abstract type AbstractMatrixFactorization end
struct QR <: AbstractMatrixFactorization end
struct SVD{T<:TruncationScheme} <: AbstractMatrixFactorization 
	trunc::T
end
SVD(; trunc::TruncationScheme=NoTruncation()) = SVD(trunc)
