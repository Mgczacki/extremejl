# Extreme.jl

This repository contains the code associated with the refactor of the ext94 (extreme) component for the [AIMPAC](https://www.chemistry.mcmaster.ca/aimpac/imagemap/imagemap.htm) suite of software applications. This was created by reverse-engineering the original Fortran code and porting it to Julia in order to harness its support for parallelism, GPU compute, and extensibility.

The future aim of this program is to generate a flexible and scalable library for molecule topological analysis following Richard Bader's Quantum Theory of Atoms in Molecules.

Currently the project has the necessary functionality in order to:

- Read WFN and WFX files.
- Calculate electronic densities and their laplacians, for n points in 3D space.
- Find criticial points in the electronic densitiy of molecules.
- Create plots/animations of electronic densities and their laplacians.

Install:
1. Open Julia
2. `using Pkg`
3. `Pkg.add(url="https://github.com/Mgczacki/extremejl")`

Test:
1. Clone this repository and navigate to the root folder.
2. Open Julia
3. `]activate .`
4. `]test`

For usage, see the Jupyter notebooks in the examples folder.

**This project was possible thanks to the program UNAM-PAPIIT IA-104720 / Investigación realizada gracias al programa UNAM-PAPIIT IA-104720.**
