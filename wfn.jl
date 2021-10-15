using Flux

struct WFN
    mol_name::String
    n_mol_orbitals::Integer
    n_primitives::Integer
    n_nuclei::Integer
    nuclei_type::AbstractArray
    nuclei_pos::AbstractArray
    nuclei_charge::AbstractArray
    center_assignments::AbstractArray
    type_assignments::AbstractArray
    exponents::AbstractArray
    mo::AbstractArray
    occ_no::AbstractArray
    orb_energy::AbstractArray
    tot_energy::Float32
    virial::Float32
end

function read_wfn(filepath; device = cpu)
    io = open(filepath, "r");
    doc = read(io, String)
    close(io)
    
    #Filename
    name = match(r"(?<=\s)(.*?)(?=\n)"s, doc).match
    
    #Parse centers section
    centers = match(r"GAUSSIAN(.*?)(?=\nCENTRE)"s, doc)
    nuclei_list = split(centers.match, "\n")
    metadata = popfirst!(nuclei_list)
    
    #Get metadata
    n_mol_orbitals = match(r"[0-9]*(?= MOL ORBITALS)", metadata).match |> string
    n_mol_orbitals = parse(Int64, n_mol_orbitals)
    n_primitives = match(r"[0-9]*(?= PRIMITIVES)", metadata).match |> string
    n_primitives = parse(Int64, n_primitives)
    n_nuclei = match(r"[0-9]*(?= NUCLEI)", metadata).match |> string
    n_nuclei = parse(Int64, n_nuclei)
    
    #Parse nuclei section
    nuclei_parse = [split(n) for n in nuclei_list]
    nuclei_type_l = [string(i[1]) for i in nuclei_parse]
    nuclei_pos = [parse(Float32, nuclei_parse[y][x]) for  y in 1:n_nuclei, x in 5:7]
    nuclei_charge = [parse(Float32, i[end]) for i in nuclei_parse]
    
    #Parse center assignments
    c_assignments = match(r"CENTRE ASSIGNMENTS(.*?)(?=\nTYPE)"s, doc)
    c_assignments = replace(c_assignments.match, "CENTRE ASSIGNMENTS"=>"")
    c_assignments = replace(c_assignments, r"\s+"=>" ")
    center_assignments = [parse(Int64, i) for i in split(c_assignments)]
    
    #Parse type assignments
    t_assignments = match(r"TYPE ASSIGNMENTS(.*?)(?=\nEXPONENTS)"s, doc)
    t_assignments = replace(t_assignments.match, "TYPE ASSIGNMENTS"=>"")
    t_assignments = replace(t_assignments, r"\s+"=>" ")
    type_assignments = [parse(Int64, i) for i in split(t_assignments)]

    function parse_exponent(t, s)
        a=parse.(t, split(s,'D'))
        length(a)>1 && return a[1]*10^a[2]
        a[1] 
    end

    #Parse exponents
    exps = match(r"EXPONENTS(.*?)(?=\nMO)"s, doc)
    exps = replace(exps.match, "EXPONENTS "=>"")
    exps = replace(exps, r"\s+"=>" ")
    exponents = [parse_exponent(Float32, i) for i in split(exps)]
    
    #Parse MOs section
    MO_matches = collect(eachmatch(r"MO (.*?)(?=(\nMO |\nEND DATA))"s, doc));

    function get_MOs(a)
        a = replace(a, r"MO.*\n"=>"")
        a = replace(a, r"\s+"=>" ")
        [parse_exponent(Float32, i) for i in split(a)]
    end
    
    MO = zeros(n_mol_orbitals, n_primitives)
    OCC_NO = zeros(n_mol_orbitals)
    ORB_ENERGY = zeros(n_mol_orbitals)
    
    for i in 1:n_mol_orbitals
        header = split(split(MO_matches[i].match, "\n")[1])
        MO[i,:] = get_MOs(MO_matches[i].match)
        OCC_NO[i] = parse_exponent(Float32, header[8])
        ORB_ENERGY[i] = parse_exponent(Float32, header[12])
    end
    
    #Parse last line
    tot_energy_l = match(r"TOTAL ENERGY .*"s, doc)
    last_line = split(tot_energy_l.match)
    t_e = parse(Float32, last_line[4])
    virial = parse(Float32, last_line[7])
    
    WFN(name,
    n_mol_orbitals,
    n_primitives,
    n_nuclei,
    nuclei_type_l,
    nuclei_pos,
    nuclei_charge |> device,
    center_assignments,
    type_assignments |> device,
    exponents |> device,
    MO |> device,
    OCC_NO |> device,
    ORB_ENERGY,
    t_e,
    virial)
end
