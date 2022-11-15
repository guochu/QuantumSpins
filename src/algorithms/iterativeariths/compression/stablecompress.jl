



function stable_compress(mps::Union{MPO, MPS}, alg::OneSiteStableArith=OneSiteStableArith())
	svd_alg = get_svd_alg(alg)
	mpsout = svdcompress(mps, svd_alg)
	return iterative_compress!(mpsout, mps, get_iterative_alg(alg))
end
