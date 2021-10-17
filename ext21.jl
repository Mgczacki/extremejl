using Tullio
using LinearAlgebra
using CUDA, CUDAKernels, KernelAbstractions

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

#First derivatives, for gradient

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

function get_∂²gc∂X²(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    t == 5 ? 2.0 :
    t == 11 ? 6*ΔX :
    t == 14 ? 2*ΔY :
    t == 15 ? 2*ΔZ :
    0.0 #Any other type
end

function get_∂²gc∂XY(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32 
    t == 8 ? 1.0 :
    t == 14 ? 2*ΔX :
    t == 17 ? 2*ΔY :
    t == 20 ? ΔZ :
    0.0 #Any other type
end

function get_∂²gc∂XZ(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32 
    t == 9 ? 1.0 :
    t == 15 ? 2*ΔX :
    t == 18 ? 2*ΔZ :
    t == 20 ? ΔY :
    0.0 #Any other type
end

function get_∂²gc∂Y²(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    t == 6 ? 2.0 :
    t == 12 ? 6*ΔY :
    t == 16 ? 2*ΔZ :
    t == 17 ? 2*ΔY :
    0.0 #Any other type
end

function get_∂²gc∂YZ(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    t == 10 ? 1.0 :
    t == 16 ? 2*ΔY :
    t == 19 ? 2*ΔZ :
    t == 20 ? ΔX :
    0.0 #Any other type
end

function get_∂²gc∂Z²(t::Integer, ΔX::Float32, ΔY::Float32, ΔZ::Float32)::Float32
    t == 7 ? 2.0 :
    t == 13 ? 6*ΔZ :
    t == 18 ? 2*ΔX :
    t == 19 ? 2*ΔY :
    0.0 #Any other type
end

function generate_Y_matrix_el(m::Integer,
                           n::Integer,
                           ∂²ρ∂X²::Number,
                           ∂²ρ∂XY::Number,
                           ∂²ρ∂XZ::Number,
                           ∂²ρ∂Y²::Number,
                           ∂²ρ∂YZ::Number,
                           ∂²ρ∂Z²::Number)::Float32
    X₁₁ = ∂²ρ∂X²
    X₁₂ = X₂₁ = ∂²ρ∂XY
    X₁₃ = X₃₁ = ∂²ρ∂XZ
    X₂₂ = ∂²ρ∂Y²
    X₂₃ = X₃₂ = ∂²ρ∂YZ
    X₃₃ = ∂²ρ∂Z²
    
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

function get_electronic_density(r⃗::AbstractArray, f::WFN)::AbstractArray
    @tullio r⃗_μ[prim,dim] := f.nuclei_pos[f.center_assignments[prim],dim] grad=false
    @tullio Δr⃗[p,dim,r] := r⃗[r,dim] - r⃗_μ[p,dim] grad=false
    sq_dist = dropdims(sum(Δr⃗.^2, dims=2), dims=2)
    @tullio c_g[p,r] := get_gaussian_constant(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=false
    Ψ_μ = c_g .* exp.(-f.exponents .* sq_dist)
    Φ_r = f.mo * Ψ_μ
    ρ = transpose(Φ_r.^2) * f.occ_no
end

function get_sum_squared_gradients(r⃗::AbstractArray, f::WFN)::Number
    @tullio r⃗_μ[prim,dim] := f.nuclei_pos[f.center_assignments[prim],dim] grad=Dual
    @tullio Δr⃗[p,dim,r] := r⃗[r,dim] - r⃗_μ[p,dim] grad=Dual
    sq_dist = dropdims(sum(Δr⃗.^2, dims=2), dims=2)
    @tullio c_g[p,r] := get_gaussian_constant(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=Dual
    @tullio ∂gc∂X[p,r] := get_∂gc∂X(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=Dual
    @tullio ∂gc∂Y[p,r] := get_∂gc∂Y(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=Dual
    @tullio ∂gc∂Z[p,r] := get_∂gc∂Z(f.type_assignments[p], Δr⃗[p,1,r], Δr⃗[p,2,r], Δr⃗[p,3,r]) grad=Dual
    F0 = exp.(-f.exponents .* sq_dist)
    ∂F∂X = -2.0*f.exponents .* F0 .* Δr⃗[:,1,:]
    ∂F∂Y = -2.0*f.exponents .* F0 .* Δr⃗[:,2,:]
    ∂F∂Z = -2.0*f.exponents .* F0 .* Δr⃗[:,3,:]
    ∂Ψ_μ∂X = ∂gc∂X .* F0 + c_g .* ∂F∂X
    ∂Ψ_μ∂Y = ∂gc∂Y .* F0 + c_g .* ∂F∂Y
    ∂Ψ_μ∂Z = ∂gc∂Z .* F0 + c_g .* ∂F∂Z
    ∂Φ∂X = f.mo * ∂Ψ_μ∂X
    ∂Φ∂Y = f.mo * ∂Ψ_μ∂Y
    ∂Φ∂Z = f.mo * ∂Ψ_μ∂Z
    Ψ_μ = c_g .* exp.(-f.exponents .* sq_dist)
    Φ_r = f.mo * Ψ_μ
    Wx = transpose(∂Φ∂X .* Φ_r) * f.occ_no
    Wy = transpose(∂Φ∂Y .* Φ_r) * f.occ_no
    Wz = transpose(∂Φ∂Z .* Φ_r) * f.occ_no

    sum(Wx.^2) + sum(Wy.^2) + sum(Wz.^2)
end


function find_points(r⃗, f; iters = 100, η = 0.1)
    df(x) = Zygote.gradient(r⃗ -> get_sum_squared_gradients(r⃗, f), x)
    for iter in 1:iters
        grad = df(r⃗)[1]
        r⃗ -= η .* grad
    end
    r⃗, get_electronic_density(r⃗, f)
end
