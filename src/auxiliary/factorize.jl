
abstract type AbstractMatrixFactorization end
struct QR <: AbstractMatrixFactorization end
struct SVD <: AbstractMatrixFactorization end