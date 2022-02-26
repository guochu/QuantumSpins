


function _distance2(x, y)
	sA = real(dot(x, x))
	sB = real(dot(y, y))
	c = dot(x, y)
	r = sA+sB-2*real(c)
	return abs(r)
end

_distance(x, y) = sqrt(_distance2(x, y))


distance2(x::AbstractArray{<:Number, N}, y::AbstractArray{<:Number, N}) where N = _distance2(x, y)
distance(x::AbstractArray{<:Number, N}, y::AbstractArray{<:Number, N}) where N = _distance(x, y)
