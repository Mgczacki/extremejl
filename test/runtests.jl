using Test
using extreme

function get_results(filepath, i⃗, device)
    #Obtain the results when executing on a file for a given device, initialization.
    f = read_wfn(filepath, device = device);
    r⃗ = i⃗ .|> Float32 |> device
    r⃗_found, ρ⃗, iters = find_critical_ρ_points(r⃗, f, iters=100);
    r⃗_found |> cpu, ρ⃗ |> cpu
end;

function molecule_crit_point_test(name, filepath, i⃗, r⃗_real, ρ⃗_real)
    #Execute the tests for a filepath.
    @testset verbose = true "$name" begin
        #First obtain the calculations given same initialization for gpu, cpu
        r⃗_g, ρ⃗_g = get_results(filepath, i⃗, gpu)
        r⃗_c, ρ⃗_c = get_results(filepath, i⃗, cpu)
        #Equivalence test. Pass expected only if GPU results are the same as CPU results.
        @testset verbose = true "Compute Equivalence" begin
            @test r⃗_g ≈ r⃗_c atol=0.01
            @test ρ⃗_g ≈ ρ⃗_c atol=0.01
        end
        #Accuracy test. Pass expected if found values match real critical points and their ρ.
        @testset verbose = true "Accuracy" begin
            @test r⃗_g ≈ r⃗_real atol=0.01
            @test ρ⃗_g ≈ ρ⃗_real atol=0.01 
        end
    end
end;

@testset verbose = true "Critical Points Testset" begin
    #Test case for H2O
    filepath = "validation_data/h2o.wfn";
    i⃗ = [ 0.0   1.0   0.0
         -1.5   1.5   0.0];
    r⃗_real = [-0.106 0.777 0; -1.976 2.098 0];
    ρ⃗_real = [0.327; 0.327];
    molecule_crit_point_test("H2O", filepath, i⃗, r⃗_real, ρ⃗_real)
    #Test case for H2O2
    filepath = "validation_data/h2o2.wfn";
    i⃗ = [ -1.5  1.5  0.0
          -0.1  1.0  0.0
           1.2  0.0  1.0];
    r⃗_real = [-2.05 2.089 -0.153; -0.257 0.911 0.199; 1.53 0.108 1.13]
    ρ⃗_real = [0.322; 0.2046; 0.322];
    molecule_crit_point_test("H2O2", filepath, i⃗, r⃗_real, ρ⃗_real)
    #Test case for phen (a subset of possible critical points)
    filepath = "validation_data/phen.wfn";
    i⃗ = [ 25.8  26.5  14.0
          23.0  22.0  12.0
          27.0  28.0  15.0
          23.1  15.0   7.5];
    r⃗_real = [26.375  25.207  13.630; 23.296  22.072  12.390; 27.470  28.492  15.546; 23.556 15.263 7.683]
    ρ⃗_real = [0.02; 0.02; 0.26; 0.26];
    molecule_crit_point_test("phen", filepath, i⃗, r⃗_real, ρ⃗_real)
    #Test case for tih2o6 (a subset of possible critical points)
    filepath = "validation_data/tih2o6.wfn";
    i⃗ = [ -1.0  1.0  0.0
          -3.0  2.0  0.0
          -8.0  1.0  0.0];
    r⃗_real = [-1.178 1.157 -0.220; -3.008 2.898 0.122; -8.124 1.289 -0.664]
    ρ⃗_real = [0.055; 0.055; 0.325];
    molecule_crit_point_test("tih2o6", filepath, i⃗, r⃗_real, ρ⃗_real)
end;
