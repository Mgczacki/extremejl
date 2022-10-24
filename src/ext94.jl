module ext94

include("wfn.jl")
include("calculations.jl")

export read_wfn, read_wfx, cpu, gpu
export find_critical_ρ_points, get_electronic_density, get_electronic_density_laplacian

end # module
