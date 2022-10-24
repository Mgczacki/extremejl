# Extreme 94 refactor - Julia

This repository contains the code associated with the refactor of the ext94 component for the [AIMPAC](https://www.chemistry.mcmaster.ca/aimpac/imagemap/imagemap.htm) suite of software applications. This was created by reverse-engineering the original Fortran code and porting it to Julia in order to harness its support for parallelism, GPU compute, and extensibility.

The future aim of this program is to generate a flexible and scalable library for molecule topological analysis following Richard Bader's Quantum Theory of Atoms in Molecules.

Currently the project has the necessary functionality in order to:

- Read WFN and WFX files.
- Calculate electronic densities and their laplacians, for n points in 3D space.
- Find criticial points in the electronic densitiy of molecules.
- Create plots/animations of electronic densities and their laplacians.

Install:
1. Clone the repository.
2. Open Julia
3. `using Pkg`
4. `Pkg.add(path="<Download path>/ext94")`

For usage, see the Jupyter notebooks in the examples folder.

**This project was possible thanks to the program UNAM-PAPIIT IA-104720 / Investigación realizada gracias al programa UNAM-PAPIIT IA-104720.**
