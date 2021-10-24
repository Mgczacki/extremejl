using Flux
using LightXML

struct WFN
    filename::String
    title::String
    n_mol_orbitals::Int32
    n_primitives::Int32
    n_nuclei::Int32
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
    virial_ratio::Float32
end

struct WFX
    filename::String
    title::String
    keywords::AbstractArray
    n_mol_orbitals::Int32
    n_primitives::Int32
    n_nuclei::Int32
    n_elec::Int32
    n_perturbations::Int32
    n_α_elec::Int32
    n_β_elec::Int32
    elec_spin_mult::Int32
    n_core_elec::Int32
    nuclear_names::AbstractArray
    atomic_numbers::AbstractArray
    nuclei_pos::AbstractArray
    nuclei_charge::AbstractArray
    center_assignments::AbstractArray
    type_assignments::AbstractArray
    exponents::AbstractArray
    mo::AbstractArray
    mo_spin_types::AbstractArray
    occ_no::AbstractArray
    orb_energy::AbstractArray
    tot_energy::Float32
    virial_ratio::Float32
end

AtomicInformationFile = Union{WFN,WFX}

function read_wfn(filepath; device = cpu)
    io = open(filepath, "r");
    doc = read(io, String)
    close(io)
    
    #title
    name = match(r"(?<=\s)(.*?)(?=\n)"s, doc).match
    
    #Parse centers section
    centers = match(r"GAUSSIAN(.*?)(?=\nCENTRE)"s, doc)
    nuclei_list = split(centers.match, "\n")
    metadata = popfirst!(nuclei_list)
    
    #Get metadata
    n_mol_orbitals = match(r"[0-9]*(?= MOL ORBITALS)", metadata).match |> string
    n_mol_orbitals = parse(Int32, n_mol_orbitals)
    n_primitives = match(r"[0-9]*(?= PRIMITIVES)", metadata).match |> string
    n_primitives = parse(Int32, n_primitives)
    n_nuclei = match(r"[0-9]*(?= NUCLEI)", metadata).match |> string
    n_nuclei = parse(Int32, n_nuclei)
    
    #Parse nuclei section
    nuclei_parse = [split(n) for n in nuclei_list]
    nuclei_type_l = [string(i[1]) for i in nuclei_parse]
    nuclei_pos = [parse(Float32, nuclei_parse[y][x]) for  y in 1:n_nuclei, x in 5:7]
    nuclei_charge = [parse(Float32, i[end]) for i in nuclei_parse]
    
    #Parse center assignments
    c_assignments = match(r"CENTRE ASSIGNMENTS(.*?)(?=\nTYPE)"s, doc)
    c_assignments = replace(c_assignments.match, "CENTRE ASSIGNMENTS"=>"")
    c_assignments = replace(c_assignments, r"\s+"=>" ")
    center_assignments = [parse(Int32, i) for i in split(c_assignments)]
    
    #Parse type assignments
    t_assignments = match(r"TYPE ASSIGNMENTS(.*?)(?=\nEXPONENTS)"s, doc)
    t_assignments = replace(t_assignments.match, "TYPE ASSIGNMENTS"=>"")
    t_assignments = replace(t_assignments, r"\s+"=>" ")
    type_assignments = [parse(Int32, i) for i in split(t_assignments)]

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
    
    MO = zeros(Float32, n_mol_orbitals, n_primitives)
    OCC_NO = zeros(Float32, n_mol_orbitals)
    ORB_ENERGY = zeros(Float32, n_mol_orbitals)
    
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
    
    WFN(
    filepath,
    name,
    n_mol_orbitals,
    n_primitives,
    n_nuclei,
    nuclei_type_l,
    nuclei_pos |> device,
    nuclei_charge |> device,
    center_assignments |> device,
    type_assignments |> device,
    exponents |> device,
    MO |> device,
    OCC_NO |> device,
    ORB_ENERGY,
    t_e,
    virial)
end

function read_wfx(filepath; device = cpu)
    io = open(filepath, "r");
    doc = read(io, String)
    close(io)
    doc = replace(doc, "Virial Ratio (-V/T)"=>"Virial Ratio")
    doc = sprint(sizehint=sizeof(doc)) do io
	    in_tag = false
	    invalid = [' ', '=', '+']
	    for c in doc
		if c == '<'
		    in_tag = true
		elseif c == '>'
		    in_tag = false
		end
		
		write(io, in_tag && c in invalid ? '_' : c)
	    end
	end
    xdoc = parse_string("<root>"*doc*"</root>")
    
    title = split(content(find_element(root(xdoc), "Title")))[1]
    keywords = split(content(find_element(root(xdoc), "Keywords"))) .|> String
    num_nuc = split(content(find_element(root(xdoc), "Number_of_Nuclei")))[1] |> x -> parse(Int32, x)
    num_mo = split(content(find_element(root(xdoc), "Number_of_Occupied_Molecular_Orbitals")))[1] |> x -> parse(Int32, x)
    num_pert = split(content(find_element(root(xdoc), "Number_of_Perturbations")))[1] |> x -> parse(Int32, x)
    num_elec = split(content(find_element(root(xdoc), "Number_of_Electrons")))[1] |> x -> parse(Int32, x)
    num_α_elec = split(content(find_element(root(xdoc), "Number_of_Alpha_Electrons")))[1] |> x -> parse(Int32, x)
    num_β_elec = split(content(find_element(root(xdoc), "Number_of_Beta_Electrons")))[1] |> x -> parse(Int32, x)
    elec_spin_mult = split(content(find_element(root(xdoc), "Electronic_Spin_Multiplicity")))[1] |> x -> parse(Int32, x)
    num_core_elec = split(content(find_element(root(xdoc), "Number_of_Core_Electrons")))[1] |> x -> parse(Int32, x)
    nuclear_names = split(content(find_element(root(xdoc), "Nuclear_Names"))) .|> String
    atomic_numbers = split(content(find_element(root(xdoc), "Atomic_Numbers"))) .|> x -> parse(Int32, x)
    nuclear_charges = split(content(find_element(root(xdoc), "Nuclear_Charges"))) .|> x -> parse(Float32, x)
    nuclear_coordinates = split(content(find_element(root(xdoc), "Nuclear_Cartesian_Coordinates"))) .|> x -> parse(Float32, x)
    nuclear_coordinates = permutedims(reshape(nuclear_coordinates, (3,:)), (2,1))
    num_primitives = split(content(find_element(root(xdoc), "Number_of_Primitives")))[1] |> x -> parse(Int32, x)
    primitive_centers = split(content(find_element(root(xdoc), "Primitive_Centers"))) .|> x -> parse(Int32, x)
    primitive_types = split(content(find_element(root(xdoc), "Primitive_Types"))) .|> x -> parse(Int32, x)
    primitive_exponents = split(content(find_element(root(xdoc), "Primitive_Exponents"))) .|> x -> parse(Float32, x)
    mo_occ_numbers = split(content(find_element(root(xdoc), "Molecular_Orbital_Occupation_Numbers"))) .|> x -> parse(Float32, x)
    mo_energies = split(content(find_element(root(xdoc), "Molecular_Orbital_Energies"))) .|> x -> parse(Float32, x)
    mo_spin_types = split(content(find_element(root(xdoc), "Molecular_Orbital_Spin_Types")))
    energy = split(content(find_element(root(xdoc), "Energy___T___Vne___Vee___Vnn")))[1] |> x -> parse(Float32, x)
    virial_ratio = split(content(find_element(root(xdoc), "Virial_Ratio")))[1] |> x -> parse(Float32, x)

    mo_l = find_element(root(xdoc), "Molecular_Orbital_Primitive_Coefficients")
    MOs = split(content(mo_l), "\n\n", keepempty=false)
    MO_matrix = zeros(Float32, num_mo ,num_primitives)

    for (idx,val) in enumerate(MOs)
       if idx %2 == 0
           MO_matrix[Integer(idx/2),:] = (split(val) .|> x -> parse(Float32, x))
        end
    end
    
    WFX(filepath,
    title,
    keywords,
    num_mo,
    num_primitives,
    num_nuc,
    num_elec,
    num_pert,
    num_α_elec,
    num_β_elec,
    elec_spin_mult,
    num_core_elec,
    nuclear_names,
    atomic_numbers,
    nuclear_coordinates |> device,
    nuclear_charges |> device,
    primitive_centers |> device,
    primitive_types |> device,
    primitive_exponents |> device,
    MO_matrix |> device,
    mo_spin_types,
    mo_occ_numbers |> device,
    mo_energies,
    energy,
    virial_ratio
    )
end
