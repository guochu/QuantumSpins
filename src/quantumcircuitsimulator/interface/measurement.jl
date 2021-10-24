


struct QMeasure 
	position::Int
	auto_reset::Bool
	keep::Bool

	QMeasure(key::Int; auto_reset::Bool=true, keep::Bool=true) = new(key, auto_reset, keep)
end

name(s::QMeasure) = "Q:Z$(s.key)"

