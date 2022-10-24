using Tullio
using LinearAlgebra
using CUDA, CUDAKernels, KernelAbstractions

"""
    rep(a::Number, b::Number, needs_replacing::Integer)

A function that selects one of two numbers given a binary mask.
Args:
    a[Number]: Number to output if needs_replacing is 0.
    b[Number]: Number to output if needs_replacing is 1 (or anything other than zero).
    needs_replacing[Integer]: Boolean mask.
"""
function rep(a::Number, b::Number, needs_replacing::Integer)
    if needs_replacing == 0
       return a 
    else
        return b
    end
end

"""
    get_gaussian_constant(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32

Obtains the gaussian constant relative to the type of molecular orbital.
Args:
    t[Integer]: Type of molecular orbital.
    ΔX[Float32]
    ΔY[Float32]
    ΔZ[Float32]
Returns:
    c[Float32]: Gaussian constant
"""
function get_gaussian_constant(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    t == 1 ? 1.0 :
    t == 2 ? ΔX :
    t == 3 ? ΔY :
    t == 4 ? ΔZ :
    t == 5 ? ΔX^2 :
    t == 6 ? ΔY^2 :
    t == 7 ? ΔZ^2 :
    t == 8 ? ΔX*ΔY :
    t == 9 ? ΔX*ΔZ :
    t == 10 ? ΔY*ΔZ :
    t == 11 ? ΔX^3 :
    t == 12 ? ΔY^3 :
    t == 13 ? ΔZ^3 :
    t == 14 ? ΔX^2*ΔY :
    t == 15 ? ΔX^2*ΔZ :
    t == 16 ? ΔY^2*ΔZ :
    t == 17 ? ΔX*ΔY^2 :
    t == 18 ? ΔX*ΔZ^2 :
    t == 19 ? ΔY*ΔZ^2 :
    t == 20 ? ΔX*ΔY*ΔZ :
    1.0 #Any other type (should be unreachable)
end

"""
    get_∂gc∂X(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32 
    
Obtains the derivative relative to X for the Gaussian constant associated to the type.
Args:
    t[Integer]: Type of molecular orbital.
    ΔX[Float32]
    ΔY[Float32]
    ΔZ[Float32]
Returns:
    c[Float32]: Partial derivative with respect to X of Gaussian constant
"""
function get_∂gc∂X(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32 
    t == 2 ? 1.0 :
    t == 5 ? 2*ΔX :
    t == 8 ? ΔY :
    t == 9 ? ΔZ :
    t == 11 ? 3*ΔX^2 :
    t == 14 ? 2*ΔX*ΔY :
    t == 15 ? 2*ΔX*ΔZ :
    t == 17 ? ΔY^2 :
    t == 18 ? ΔZ^2 :
    t == 20 ? ΔY*ΔZ :
    0.0 #Any other type
end

"""
    get_∂gc∂Y(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32 
    
Obtains the derivative relative to Y for the Gaussian constant associated to the type.
Args:
    t[Integer]: Type of molecular orbital.
    ΔX[Float32]
    ΔY[Float32]
    ΔZ[Float32]
Returns:
    c[Float32]: Partial derivative with respect to Y of Gaussian constant
"""
function get_∂gc∂Y(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    t == 3 ? 1.0 :
    t == 6 ? 2*ΔY :
    t == 8 ? ΔX :
    t == 10 ? ΔZ :
    t == 12 ? 3*ΔY^2 :
    t == 14 ? ΔX^2 :
    t == 16 ? 2*ΔY*ΔZ :
    t == 17 ? 2*ΔX*ΔY :
    t == 19 ? ΔZ^2 :
    t == 20 ? ΔX*ΔZ :
    0.0 #Any other type
end

"""
    get_∂gc∂Z(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32 
    
Obtains the derivative relative to Z for the Gaussian constant associated to the type.
Args:
    t[Integer]: Type of molecular orbital.
    ΔX[Float32]
    ΔY[Float32]
    ΔZ[Float32]
Returns:
    c[Float32]: Partial derivative with respect to Z of Gaussian constant
"""
function get_∂gc∂Z(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    t == 4 ? 1.0 :
    t == 7 ? 2*ΔZ :
    t == 9 ? ΔX :
    t == 10 ? ΔY :
    t == 13 ? 3*ΔZ^2 :
    t == 15 ? ΔX^2 :
    t == 16 ? ΔY^2 :
    t == 18 ? 2*ΔX*ΔZ :
    t == 19 ? 2*ΔY*ΔZ :
    t == 20 ? ΔX*ΔY :
    0.0 #Any other type
end

#Second derivatives, for calculating hessian
"""
    get_∂²gc∂X²(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    
Obtains the derivative relative to X² for the Gaussian constant associated to the type.
Args:
    t[Integer]: Type of molecular orbital.
    ΔX[Float32]
    ΔY[Float32]
    ΔZ[Float32]
Returns:
    c[Float32]: Partial derivative with respect to X² of Gaussian constant
"""
function get_∂²gc∂X²(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    t == 5 ? 2.0 :
    t == 11 ? 6*ΔX :
    t == 14 ? 2*ΔY :
    t == 15 ? 2*ΔZ :
    0.0 #Any other type
end

"""
    get_∂²gc∂XY(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    
Obtains the derivative relative to XY for the Gaussian constant associated to the type.
Args:
    t[Integer]: Type of molecular orbital.
    ΔX[Float32]
    ΔY[Float32]
    ΔZ[Float32]
Returns:
    c[Float32]: Partial derivative with respect to XY of Gaussian constant
"""
function get_∂²gc∂XY(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32 
    t == 8 ? 1.0 :
    t == 14 ? 2*ΔX :
    t == 17 ? 2*ΔY :
    t == 20 ? ΔZ :
    0.0 #Any other type
end

"""
    get_∂²gc∂XZ(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    
Obtains the derivative relative to XZ for the Gaussian constant associated to the type.
Args:
    t[Integer]: Type of molecular orbital.
    ΔX[Float32]
    ΔY[Float32]
    ΔZ[Float32]
Returns:
    c[Float32]: Partial derivative with respect to XZ of Gaussian constant
"""
function get_∂²gc∂XZ(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32 
    t == 9 ? 1.0 :
    t == 15 ? 2*ΔX :
    t == 18 ? 2*ΔZ :
    t == 20 ? ΔY :
    0.0 #Any other type
end

"""
    get_∂²gc∂Y²(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    
Obtains the derivative relative to Y² for the Gaussian constant associated to the type.
Args:
    t[Integer]: Type of molecular orbital.
    ΔX[Float32]
    ΔY[Float32]
    ΔZ[Float32]
Returns:
    c[Float32]: Partial derivative with respect to Y² of Gaussian constant
"""
function get_∂²gc∂Y²(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    t == 6 ? 2.0 :
    t == 12 ? 6*ΔY :
    t == 16 ? 2*ΔZ :
    t == 17 ? 2*ΔY :
    0.0 #Any other type
end

"""
    get_∂²gc∂YZ(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    
Obtains the derivative relative to YZ for the Gaussian constant associated to the type.
Args:
    t[Integer]: Type of molecular orbital.
    ΔX[Float32]
    ΔY[Float32]
    ΔZ[Float32]
Returns:
    c[Float32]: Partial derivative with respect to YZ of Gaussian constant
"""
function get_∂²gc∂YZ(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    t == 10 ? 1.0 :
    t == 16 ? 2*ΔY :
    t == 19 ? 2*ΔZ :
    t == 20 ? ΔX :
    0.0 #Any other type
end

"""
    get_∂²gc∂Z²(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    
Obtains the derivative relative to Z² for the Gaussian constant associated to the type.
Args:
    t[Integer]: Type of molecular orbital.
    ΔX[Float32]
    ΔY[Float32]
    ΔZ[Float32]
Returns:
    c[Float32]: Partial derivative with respect to Z² of Gaussian constant
"""
function get_∂²gc∂Z²(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    t == 7 ? 2.0 :
    t == 13 ? 6*ΔZ :
    t == 18 ? 2*ΔX :
    t == 19 ? 2*ΔY :
    0.0 #Any other type
end

"""
    generate_Y_matrix_el(m::Integer,
                         n::Integer,
                         X₁₁::Number,
                         X₁₂::Number,
                         X₁₃::Number,
                         X₂₂::Number,
                         X₂₃::Number,
                         X₃₃::Number)::Float32
    
Obtains the element at position m,n of the transposed matrix of cofactors for a 3x3 symmetric real matrix.
This enables the calculation of an arbitrary number of 3x3 inverse Hessians in parallel.
Args:
    m::Integer
    n::Integer
    X₁₁::Number
    X₁₂::Number
    X₁₃::Number
    X₂₂::Number
    X₂₃::Number
    X₃₃::Number
Returns:
    c[Number]: Element of the transposed matrix of cofactors at position m,n.
"""
function generate_Y_matrix_el(m::Integer,
                           n::Integer,
                           X₁₁::Number,
                           X₁₂::Number,
                           X₁₃::Number,
                           X₂₂::Number,
                           X₂₃::Number,
                           X₃₃::Number)::Float32
    X₂₁ = X₁₂
    X₃₁ = X₁₃
    X₃₂ = X₂₃
    
    m == n == 1 ? X₂₂*X₃₃ - X₂₃*X₃₂ :
    m == 2 && n == 1 ? X₂₃*X₃₁ - X₂₁*X₃₃ :
    m == 3 && n == 1 ? X₂₁*X₃₂ - X₂₂*X₃₁ :
    m == 1 && n == 2 ? X₁₃*X₃₂ - X₁₂*X₃₃ :
    m == n == 2 ? X₁₁*X₃₃ - X₁₃*X₃₁ :
    m == 3 && n == 2 ? X₁₂*X₃₁ - X₁₁*X₃₂ :
    m == 1 && n == 3 ? X₁₂*X₂₃ - X₁₃*X₂₂ :
    m == 2 && n == 3 ? X₁₃*X₂₁ - X₁₁*X₂₃ :
    m == n == 3 ? X₁₁*X₂₂ - X₁₂*X₂₁ :
    0.0
end

"""
    get_electronic_density(r⃗::AbstractArray, f::AtomicInformationFile)::AbstractArray   
    
Obtains the electronic density for n points in 3D space given an AtomicInformationFile.
Args:
    r⃗::AbstractArray
    f::AtomicInformationFile
Returns:
    ρ[AbstractArray]: Array of electronic densities.
"""
function get_electronic_density(r⃗::AbstractArray, f::AtomicInformationFile)::AbstractArray
    @tullio r⃗_μ[prim,dim] := f.nuclei_pos[f.center_assignments[prim],dim] grad=false
    @tullio Δr⃗[p,dim,r] := r⃗[r,dim] - r⃗_μ[p,dim] grad=false
    sq_dist = dropdims(sum(Δr⃗.^2, dims=2), dims=2)
    @tullio c_g[p,r] := get_gaussian_constant(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
    Ψ_μ = c_g .* exp.(-f.exponents .* sq_dist)
    Φ_r = f.mo * Ψ_μ
    ρ = transpose(Φ_r.^2) * f.occ_no
end

"""
    get_electronic_density_laplacian(r⃗::AbstractArray, f::AtomicInformationFile)::AbstractArray   
    
Obtains the electronic density laplacian for n points in 3D space given an AtomicInformationFile.
Args:
    r⃗::AbstractArray
    f::AtomicInformationFile
Returns:
    ∇²[AbstractArray]: Array of electronic density laplacians.
"""
function get_electronic_density_laplacian(r⃗::AbstractArray, f::AtomicInformationFile)::AbstractArray
    @tullio r⃗_μ[prim,dim] := f.nuclei_pos[f.center_assignments[prim],dim] grad=false
    @tullio Δr⃗[p,dim,r] := r⃗[r,dim] - r⃗_μ[p,dim] grad=false
    sq_dist = dropdims(sum(Δr⃗.^2, dims=2), dims=2)
    @tullio c_g[p,r] := get_gaussian_constant(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
    
    @tullio ∂gc∂X[p,r] := get_∂gc∂X(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
    @tullio ∂gc∂Y[p,r] := get_∂gc∂Y(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
    @tullio ∂gc∂Z[p,r] := get_∂gc∂Z(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false

    @tullio ∂²gc∂X²[p,r] := get_∂²gc∂X²(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
    @tullio ∂²gc∂Y²[p,r] := get_∂²gc∂Y²(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
    @tullio ∂²gc∂Z²[p,r] := get_∂²gc∂Z²(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
    
    F0 = exp.(-f.exponents .* sq_dist)
    toalp = -2.0 * f.exponents
    toalpe = toalp .* F0
    ∂F∂X = toalpe .* Δr⃗[:,1,:]
    ∂F∂Y = toalpe .* Δr⃗[:,2,:]
    ∂F∂Z = toalpe .* Δr⃗[:,3,:]
    ∂²F∂X² = (toalp .* Δr⃗[:,1,:] .* ∂F∂X) + toalpe
    ∂²F∂Y² = (toalp .* Δr⃗[:,2,:] .* ∂F∂Y) + toalpe
    ∂²F∂Z² = (toalp .* Δr⃗[:,3,:] .* ∂F∂Z) + toalpe
    
    Ψ_μ = c_g .* F0
    ∂Ψ_μ∂X = ∂gc∂X .* F0 + c_g .* ∂F∂X
    ∂Ψ_μ∂Y = ∂gc∂Y .* F0 + c_g .* ∂F∂Y
    ∂Ψ_μ∂Z = ∂gc∂Z .* F0 + c_g .* ∂F∂Z
    ∂²Ψ_μ∂X² = (∂²gc∂X² .* F0) + (2 * ∂gc∂X .* ∂F∂X) + (c_g .* ∂²F∂X²)
    ∂²Ψ_μ∂Y² = (∂²gc∂Y² .* F0) + (2 * ∂gc∂Y .* ∂F∂Y) + (c_g .* ∂²F∂Y²)
    ∂²Ψ_μ∂Z² = (∂²gc∂Z² .* F0) + (2 * ∂gc∂Z .* ∂F∂Z) + (c_g .* ∂²F∂Z²)

    Φ_r = f.mo * Ψ_μ
    ∂Φ∂X = f.mo * ∂Ψ_μ∂X
    ∂Φ∂Y = f.mo * ∂Ψ_μ∂Y
    ∂Φ∂Z = f.mo * ∂Ψ_μ∂Z
    ∂²Φ∂X² = f.mo * ∂²Ψ_μ∂X²
    ∂²Φ∂Y² = f.mo * ∂²Ψ_μ∂Y²
    ∂²Φ∂Z² = f.mo * ∂²Ψ_μ∂Z²

    ∂²ρ∂X² = 2 * (transpose(∂Φ∂X.^2)  + transpose(∂²Φ∂X² .* Φ_r)) * f.occ_no
    ∂²ρ∂Y² = 2 * (transpose(∂Φ∂Y.^2) + transpose(∂²Φ∂Y² .* Φ_r)) * f.occ_no
    ∂²ρ∂Z² = 2 * (transpose(∂Φ∂Z.^2) + transpose(∂²Φ∂Z² .* Φ_r)) * f.occ_no
    
    ∂²ρ∂X²+∂²ρ∂Y²+∂²ρ∂Z²
end


"""
    find_critical_ρ_points(r⃗, f::AtomicInformationFile; iters = 15, conv_check= 1e-7)
    
Uses Newton-Raphson to try to find critical points of the electronic density given the initial guesses r⃗.
Args:
    r⃗
    f::AtomicInformationFile
    iters = 15
    conv_check = 1e-7
Returns:
    r⃗[AbstractArray]: Array of the found coordinates for critical points.
    ρ⃗[AbstractArray]: Array of the associated electronic densities for the critical points.
    iterations_done[AbstractArray]: Array containing the number of iterations done for each point.
"""
function find_critical_ρ_points(r⃗, f::WFN; iters = 15, conv_check = 1e-7)
    #To allow us to acces Φ_r outside the loop's scope
    Φ_r = NaN
    iterations_done = NaN
    if r⃗ isa CuArray
        iterations_done = CUDA.zeros(size(r⃗,1))
    else
        iterations_done = zeros(size(r⃗,1))
    end
    for i in 1:iters
    #Assignation of center per primitive
        @tullio r⃗_μ[prim,dim] := f.nuclei_pos[f.center_assignments[prim],dim] grad=false
        #Difference between each proposed point and each nuclei center assigned to a MO
        @tullio Δr⃗[p,dim,r] := r⃗[r,dim] - r⃗_μ[p,dim] grad=false
        #Squared distances
        sq_dist = dropdims(sum(Δr⃗.^2, dims=2), dims=2)
        #Gaussian constant
        @tullio c_g[p,r] := get_gaussian_constant(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false

        @tullio ∂gc∂X[p,r] := get_∂gc∂X(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
        @tullio ∂gc∂Y[p,r] := get_∂gc∂Y(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
        @tullio ∂gc∂Z[p,r] := get_∂gc∂Z(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false

        @tullio ∂²gc∂X²[p,r] := get_∂²gc∂X²(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
        @tullio ∂²gc∂XY[p,r] := get_∂²gc∂XY(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
        @tullio ∂²gc∂XZ[p,r] := get_∂²gc∂XZ(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
        @tullio ∂²gc∂Y²[p,r] := get_∂²gc∂Y²(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
        @tullio ∂²gc∂YZ[p,r] := get_∂²gc∂YZ(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
        @tullio ∂²gc∂Z²[p,r] := get_∂²gc∂Z²(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false

        F0 = exp.(-f.exponents .* sq_dist)
        toalp = -2.0 * f.exponents
        toalpe = toalp .* F0
        ∂F∂X = toalpe .* Δr⃗[:,1,:]
        ∂F∂Y = toalpe .* Δr⃗[:,2,:]
        ∂F∂Z = toalpe .* Δr⃗[:,3,:]
        ∂²F∂X² = (toalp .* Δr⃗[:,1,:] .* ∂F∂X) + toalpe
        ∂²F∂XY = toalp .* Δr⃗[:,2,:] .* ∂F∂X
        ∂²F∂XZ = toalp .* Δr⃗[:,3,:] .* ∂F∂X
        ∂²F∂Y² = (toalp .* Δr⃗[:,2,:] .* ∂F∂Y) + toalpe
        ∂²F∂YZ = toalp .* Δr⃗[:,3,:] .* ∂F∂Y
        ∂²F∂Z² = (toalp .* Δr⃗[:,3,:] .* ∂F∂Z) + toalpe

        Ψ_μ = c_g .* F0
        ∂Ψ_μ∂X = ∂gc∂X .* F0 + c_g .* ∂F∂X
        ∂Ψ_μ∂Y = ∂gc∂Y .* F0 + c_g .* ∂F∂Y
        ∂Ψ_μ∂Z = ∂gc∂Z .* F0 + c_g .* ∂F∂Z
        ∂²Ψ_μ∂X² = (∂²gc∂X² .* F0) + (2 * ∂gc∂X .* ∂F∂X) + (c_g .* ∂²F∂X²)
        ∂²Ψ_μ∂XY = (∂²gc∂XY .* F0) + (∂gc∂X .* ∂F∂Y) + (∂gc∂Y .* ∂F∂X) + (c_g .* ∂²F∂XY)
        ∂²Ψ_μ∂XZ = (∂²gc∂XZ .* F0) + (∂gc∂X .* ∂F∂Z) + (∂gc∂Z .* ∂F∂X) + (c_g .* ∂²F∂XZ)
        ∂²Ψ_μ∂Y² = (∂²gc∂Y² .* F0) + (2 * ∂gc∂Y .* ∂F∂Y) + (c_g .* ∂²F∂Y²)
        ∂²Ψ_μ∂YZ = (∂²gc∂YZ .* F0) + (∂gc∂Y .* ∂F∂Z) + (∂gc∂Z .* ∂F∂Y) + (c_g .* ∂²F∂YZ)
        ∂²Ψ_μ∂Z² = (∂²gc∂Z² .* F0) + (2 * ∂gc∂Z .* ∂F∂Z) + (c_g .* ∂²F∂Z²)
        
        Φ_r = f.mo * Ψ_μ
        ∂Φ∂X = f.mo * ∂Ψ_μ∂X
        ∂Φ∂Y = f.mo * ∂Ψ_μ∂Y
        ∂Φ∂Z = f.mo * ∂Ψ_μ∂Z
        ∂²Φ∂X² = f.mo * ∂²Ψ_μ∂X²
        ∂²Φ∂XY = f.mo * ∂²Ψ_μ∂XY
        ∂²Φ∂XZ = f.mo * ∂²Ψ_μ∂XZ
        ∂²Φ∂Y² = f.mo * ∂²Ψ_μ∂Y²
        ∂²Φ∂YZ = f.mo * ∂²Ψ_μ∂YZ
        ∂²Φ∂Z² = f.mo * ∂²Ψ_μ∂Z²

        #ρ = electronic density for proposed points
        #ρ = transpose(Φ_r.^2) * f.occ_no
        
        #Newton/pseudo-Newton methods optimize 3 equations for 3 variables:
        #Variables: X, Y, Z of each point
        #Equations: Gradient of ρ at X, Y, Z
        ∂ρ∂X = transpose(∂Φ∂X .* Φ_r) * f.occ_no
        ∂ρ∂Y = transpose(∂Φ∂Y .* Φ_r) * f.occ_no
        ∂ρ∂Z = transpose(∂Φ∂Z .* Φ_r) * f.occ_no
        #Newton/pseudo-Newton method need the inverse jacobian of the function to optimize.
        #Calculating second derivatives of ρ to obtain the Hessian
        ∂²ρ∂X² = 2 * (transpose(∂Φ∂X.^2)  + transpose(∂²Φ∂X² .* Φ_r)) * f.occ_no
        ∂²ρ∂XY = 2 * (transpose(∂Φ∂X .* ∂Φ∂Y)  + transpose(∂²Φ∂XY .* Φ_r)) * f.occ_no
        ∂²ρ∂XZ = 2 * (transpose(∂Φ∂X .* ∂Φ∂Z)  + transpose(∂²Φ∂XZ .* Φ_r)) * f.occ_no
        ∂²ρ∂Y² = 2 * (transpose(∂Φ∂Y.^2) + transpose(∂²Φ∂Y² .* Φ_r)) * f.occ_no
        ∂²ρ∂YZ = 2 * (transpose(∂Φ∂Y .* ∂Φ∂Z)  + transpose(∂²Φ∂YZ .* Φ_r)) * f.occ_no
        ∂²ρ∂Z² = 2 * (transpose(∂Φ∂Z.^2) + transpose(∂²Φ∂Z² .* Φ_r)) * f.occ_no

        #Generating the inverse of the Hessian of ρ by determinants/cofactors
        #Evaluating as a single expression because of speed.
        #Otherwise launches multiple kernels, which is slower than simply repeating the operation
        @tullio inv_H[m,n,p] := generate_Y_matrix_el(m, n, ∂²ρ∂X²[p],
            ∂²ρ∂XY[p],
            ∂²ρ∂XZ[p],
            ∂²ρ∂Y²[p],
            ∂²ρ∂YZ[p],
            ∂²ρ∂Z²[p]) / (∂²ρ∂X²[p] * generate_Y_matrix_el(1, 1, ∂²ρ∂X²[p],
            ∂²ρ∂XY[p],
            ∂²ρ∂XZ[p],
            ∂²ρ∂Y²[p],
            ∂²ρ∂YZ[p],
            ∂²ρ∂Z²[p]) + ∂²ρ∂XY[p] * generate_Y_matrix_el(2, 1, ∂²ρ∂X²[p],
            ∂²ρ∂XY[p],
            ∂²ρ∂XZ[p],
            ∂²ρ∂Y²[p],
            ∂²ρ∂YZ[p],
            ∂²ρ∂Z²[p]) + ∂²ρ∂XZ[p] * generate_Y_matrix_el(3, 1, ∂²ρ∂X²[p],
            ∂²ρ∂XY[p],
            ∂²ρ∂XZ[p],
            ∂²ρ∂Y²[p],
            ∂²ρ∂YZ[p],
            ∂²ρ∂Z²[p])) (m in 1:3, n in 1:3) grad=false
        #Update step for multidimensional Newton-Raphson
        @tullio r⃗_update[p,dim] := -inv_H[dim,1,p] * ∂ρ∂X[p] - inv_H[dim,2,p] * ∂ρ∂Y[p] - inv_H[dim,3,p] * ∂ρ∂Z[p]
        #Check if any numerical instability has happened. In such a case, updates aren't applied.
        null_results = reduce(|,isnan.(r⃗_update); dims=2)
        @tullio r⃗[r,dim] += rep(r⃗_update[r,dim], 0, null_results[r])
        iterations_done = iterations_done + (1 .- null_results)
    end
    r⃗, transpose(Φ_r.^2) * f.occ_no, iterations_done
end