---
title: 'Extreme.jl: A Julia package for Interacting Quantum Atoms Exploration' 
tags:
  - Julia
  - Quantum Chemical Topology 
  - Interacting Quantum Atoms
  - Cuda.jl
  - Tullio.jl
authors:
  - name: Mario H. Garrido-Czacki
    equal-contrib: true
    affiliation: "1"
  - name: Brandon Meza-González
    equal-contrib: true
    affiliation: "2"
  - name: Tomás Rocha-Rinza
    orcid: 0000-0003-1650-4150
    equal-contrib: true
    affiliation: "2"
  - name: Oscar A. Esquivel-Flores
    orcid: 0000-0003-1451-1285
    equal-contrib: true
    affiliation: "1"
affiliations:
 - name: Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas, Universidad Nacional Autónoma de México, Circuto Escolar 3000, C.U., Coyoacán, 04510, Ciudad de México, México.
   index: 1
 - name: Instituto de Química, Universidad Nacional Autónoma de México, Circuito Exterior, C. U., Coyoacán, 04510, Ciudad de México, México.
   index: 2
date: 20 November 2022
bibliography: paper.bib

---

# Summary
`Extreme.jl` is a *Quantum Chemical Topology* (QCT) package which computes *critical bound points* (BCP) of the electron charge distribution $\varrho(\mathbf{r})$ and determines the stable manifolds of the  BCP of a molecule. The Julia programming language[^1] helps to address the main bottleneck for exploring  *Interacting Quantum Atoms* (IQA) energy partitions through calculation of the several integrals in parallel using the CUDA.jl package[^2]. `Extreme.jl` was created by reverse-engineering the original Fortran code and porting it to Julia in order to harness its support for parallelism, GPU compute, and extensibility.

[^1]: https://julialang.org
[^2]: https://github.com/JuliaGPU/CUDA.jl

# Statement of Need

Chemical bonding is one of the most fundamental concepts in chemistry.
The most sound way to examine different sorts of chemical
interactions under the same physical basis is arguably via the examination of
quantum mechanical observables and their expectation values. The electron charge distribution, $\varrho(\mathbf{r})$, the pair density, $\varrho(\mathbf{r}_{1}, \mathbf{r}_{2})$, kinetic and potentical energy densities and quantities derived from these functions, e.g., the localisation electron function, the reduced density gradient or the Non-Covalent Index are examples of the above mentionted Dirac observables. The analysis of the local and integrated values of these functions has resulted in the emerging of the field of theoretical computational chemistry known as *Quantum Chemical Topology* (QCT). The origins of QCT are based on the *Quantum Theory of Atoms in Molecules* (QTAIM) which provides a division of the three-dimensional space into basins which are related with the atoms of chemistry envisioned by Dalton at the beginning of the XIX century. These basins are illustrated for the purine molecule ($C_{5}H_{4}N_{4}$) in Figure \ref{qtaim-purine}. The region comprising any of these basins equals the stable manifold of attractors of the trajectories of $\nabla \varrho(\mathbf{r})$, which typically coincide with the position of the nuclei of the system. Such stable manifolds are displayed in black, magenta and blue for the carbon, hydrogen and nitrogen atoms of $C_5H_4N_4$ in Figure \ref{qtaim-purine}. We note that the complete space of purine is exhaustively divided in disjoint regions, the QTAIM-atoms, which are separated by the stable manifold of *Critical Bound Points* (BCP) of $\varrho(\mathbf{r})$ shown as small green spheres in Figure \ref{qtaim-purine}. Such separatrices are known as *Inter-Atomic Surfaces* (IAS).

![Trajectories of $\nabla \varrho(\mathbf{r})$ of purine computed with the MP2/cc-pVDZ approximation. The basins correspondingto the carbon, hydrogen and nitrogen atoms are shown in black, magenta and blue respectively. The bond and ring critical points ofthe system are displayed as green and red spheres respectively. \label{qtaim-purine}](purina.png){width=80%}

Several tools of QCT take advantage of the partition of the 3D space defined by QTAIM, e.g., the *Interacting Quantum Atoms* (IQA) energy partition. The IQA approach has been exploited for the examination of many different types of chemical interactions. Given a partition of the 3D space, e.g., that provided by QTAIM, the IQA energy partition dissects the electronic energy in one-($E_{\mathrm{self}}^{\Omega_{\mathrm{A}}}$) and
two-atom ($E_{\mathrm{int}}^{\Omega_{\mathrm{A}}\Omega_{\mathrm{B}}}$) contributions:

\begin{equation} \label{div_iqa}
E = \sum_{\Omega_{\mathrm{A}}}
E_{\mathrm{self}}^{\Omega_{\mathrm{A}}} +
\frac{1}{2} \sum_{\Omega_{\mathrm{A}} \neq \Omega_{\mathrm{B}}}
E_{\mathrm{int}}^{\Omega_{\mathrm{A}}\Omega_
{\mathrm{B}}}
\end{equation}

\noindent The one- and two- atom contributions in expression (\ref{div_iqa}) can in turn be expressed as,

\begin{align} 
E_{\mathrm{self}}^{\Omega_{\mathrm{A}}} & = 
T^{\Omega_{\mathrm{A}}} + 
V_{\mathrm{en}}^{\Omega_{\mathrm{A}}\Omega_{\mathrm{A}}} +
V_{\mathrm{ee}}^{\Omega_{\mathrm{A}}\Omega_{\mathrm{A}}}
\label{e_neta} \\
E_{\mathrm{int}}^{\Omega_{\mathrm{A}}\Omega_{\mathrm{B}}} & =
\frac{Z_{\mathrm{A}}Z_{\mathrm{B}}}{r_{\mathrm{AB}}} +
V_{\mathrm{en}}^{\Omega_{\mathrm{A}}\Omega_{\mathrm{B}}} +
V_{\mathrm{en}}^{\Omega_{\mathrm{B}}\Omega_{\mathrm{A}}} +
V_{\mathrm{ee}}^{\Omega_{\mathrm{A}}\Omega_{\mathrm{B}}}
\label{e_inter}
\end{align} 

\noindent wherein $Z_A$ is the nuclear charge within atom $\Omega_{\mathrm{A}}$ together with
$\varrho(\mathbf{r})$

\begin{align}
T^{\Omega_{\mathrm{A}}} & = -\frac{1}{2} \int_{\mathbf{r}_{1}^{\, \prime} =
\mathbf{r}_{1}} \omega_{\Omega_{\mathrm{A}}}(\mathbf{r}_{1}) \nabla_1^2
\varrho_{1}(\mathbf{r}_{1}; \mathbf{r}_{1}^{\, \prime}) \mathrm{d}\mathbf{r}_1,
\label{cinetica} \\[1em]
V_{\mathrm{en}}^{\Omega_{\mathrm{A}}\Omega_{\mathrm{B}}} & = -
Z_\mathrm{B} \int \omega_{\Omega_{\mathrm{A}}}(\mathbf{r}_{1})
\frac{\varrho(\mathbf{r}_{1})}{\mathbf{r}_{1}\mathrm{B}} \mathrm{d}\mathbf{r}_{1},
\label{e_nucleo}
\end{align}

\begin{align}
V_{\mathrm{ee}}^{\Omega_{\mathrm{A}}\Omega_{\mathrm{B}}} & =
\frac{2 - \delta_{\mathrm{AB}}}{2}
\int \omega_{\Omega_{\mathrm{A}}}(\mathbf{r}_{1})
 \omega_{\Omega_{\mathrm{B}}}(\mathbf{r}_{2})
\frac{\varrho_2(\mathbf{r}_{1}, \mathbf{r}_{2})}{r_{12}} \mathrm{d} \mathbf{r}_{1}
\mathrm{d}\mathbf{r}_{2}, \ \mathrm{and} 
\label{e_e}  \\[1em]
\omega_{\Omega_\mathrm{A}}(r) & = \left\{
\begin{array}{l}
1 \ \mbox{if} \ \mathbf{r} \in \Omega_\mathrm{A}. \\
0 \ \mbox{if} \ \mathbf{r} \notin \Omega_\mathrm{A}. 
\label{omega}
\end{array}
 \right.
\end{align}

<!-- \noindent The function $\varrho_1(\mathbf{r}_{1};\mathbf{r}_{1}^{\prime})$ is the first-order reduced density matrix and $\delta_{\mathrm{AB}}$ is the Kronecker delta. The terms in equations (\ref{e_neta}) and (\ref{e_inter}) are easily interpretable. The quantity $T^{\Omega_{\mathrm{A}}}$ is the kinetic energy due to basin $\Omega_{\mathrm{A}}$ and $V_{\mathrm{e\tau}}^{\Omega_{\mathrm{A}}\Omega_{\mathrm{B}}}$ is the contribution to the potential energy due to (i) the electrons in basin $\Omega_{\mathrm{A}}$ and (ii) $\tau$, either electrons $\tau =\mathrm{e}$ or the nucleus $\tau = \mathrm{n}$, in basin $\Omega_{\mathrm{B}}$. Indeed, the Coulombic nature of the electronic Hamiltonian and the QTAIM partition, allows the electronic energy to be divided as put forward in equation (\ref{div_iqa}). 

The IQA partition has been very useful in the investigation of a wide diversity of interactions in chemistry, e.g., covalent, polar, ionic, intermolecular interactions as well as chemical bonding in solid state systems. Despite the recognised utility of the IQA analysis, formulae (\ref{cinetica})--(\ref{omega}) imply that the IQA approach requires the integration of scalar fields over very irregular volume shapes. In particular expression (\ref{e_e}) entails the six-dimensional integral over two QTAIM-basins. This endeavour is far from straightforward and it involves a computational effort which severely hampers the applicability of IQA for the study of electronic systems. Currently, the IQA can only be applied to systems with only a few tens of atoms. Indeed, the main bottleneck for the explotaition of the IQA approach is the calculation of the above mentioned integrals. Therefore new algorithms and software is needed for the amelioration of this situation. Such enhacement comprises the main contribution of this development.

More specifically, one must determine the QTAIM-basins prior to perform the integrals involved in equations(\ref{cinetica})--(\ref{e_e}). One way to do it is via the determination of the IAS of the whole system. For this purpose, one must first find all the critical points of $\varrho(\mathbf{r})$ and second determine the stable manifolds of the BCP of the molecule or molecular critical under investigation. Therefore new algorithms and software is needed for the amelioration of this situation. We herein report a software which performs both tasks and that afterwards can be expanded for the computation of the integrals in formulae (\ref{cinetica})--(\ref{e_e}). -->


# High Performance and Expresiveness for Numerical Computations

In a short period of time, the Julia programming language [@bezanson2012] has positioned itself as one of the most promising programming languages for scientific and high-performance computing. Among its most innovative features we can highlight: its ease of use as a dynamic language with powerful features that make it very productive for writing code, and its fast execution speed, at least as fast as code written in statically typed [@sengupta2019].  The use of this programming language in the area of numerical computation has increased in recent years as well as various applications exploiting parallel computation [@huo2020, @suslov2020, @huo2021] and has been tested in high-performance architectures [@hunold2020, @weichen2021]. Moreover several Julia packages support parallel computing and NVIDIA GPU's programming as CUDA.jl [@besard2017], KernelAbstractions.jl[^3] that allows to write GPU-like kernels targetting different execution backends and Tullio.jl[@tullio2022] to perform array operations written in Einstein index notation.

[^3]: https://github.com/JuliaGPU/KernelAbstractions.jl
 
# The Extreme.jl Julia package

Here we present `Extreme.jl`, a highly-performant Julia package developed with the aim of analyzing the Quantum Chemical Topology of molecules. In its current state, its main functionality is finding critical points in the electron charge distribution of any given molecule. The usual workflow with `Extreme.jl` looks as follows:

1. Load a molecule file (WFN or WFX format).
2. Select $n$ starting initialization points for critical point search.
3. Run the critical point finding algorithm.

There are other analysis tools baked into the package such as functions for finding the electron density and its laplacian for any given points in 3D space.

# Parallelizing Quantum Chemical Topolical Analysis

`Extreme.jl` was created by reverse engineering the ext94 (extreme) tool of the AIMPAC suite of software applications[^4]. In particular, we explored the source code for its critical-point finding routine. In its original form, the algorithm uses a Newton-Raphson optimization routine over the electron density space for a single point. This involves the usage of nested loop structures which correspond to inner products of vectors and matrix multiplications. The parallelization of these routines is well-known and handled by any given BLAS library, and we decided to use Tullio.jl because of its many advantages:

- It allows for the usage of Einstein notation in order to define multidimensional structures. This gives the code expressiveness, brevity, and an implementation that is remarkably close to the pure mathematical form of the underlying equations.
- Tullio.jl builds upon the KernelAbstractions.jl[^5] package, which allows for a transparent translation of higher-level expressions into the underlying operations of multiple different compute architectures. This allowed us to use the same source code in order to generate parallel CPU or GPU bound operations.
- Through KernelAbstractions.jl, it supports mapping arbitrary Julia functions over subsets of a multidimensional data structure. With this, we can create complex functionality for any of the supported architectures. Crucially, it allows us to forgo writing CUDA kernels for GPU processing.
- KernelAbstractions.jl handles parsing expressions, ordering and subdividing computations, and executing operations concurrently. This allowed us to focus on writing the code without needing to specify parallel waits, locks, or how to launch each individual step in the program's data flow.

We used the aforementioned technology stack to reframe the problem from sequential operations to find a single critical point, into finding an arbitrary number of critical points in parallel. The strategy used for this process was as follows:

1. Derive the mathematical expressions used in the algorithm from both literature and the ext94 source code. In particular, working from the innermost nested loops towards the outermost loops allowed us to identify vector inner products, Hadamard products, matrix multiplications, transposes, inverses, and possible function mappings over the data structures.
2. By adding another dimension to the identified data structures (creating matrices from n vectors, and 3D tensors from n matrices), we created the necessary data structures in order to allow the identified expressions to work for n points concurrently. This "naive" parallelization was possible because there are no operational or temporal dependencies between the processes of finding the critical points derived from different initial states.
3. For any calculations that do not use straightforward operations such as sums or multiplications, we created kernelized Julia functions. These can take subsets of input data along with any other inputs in order to replicate complex behavior like masking values or creating mappings.

[^4]: https://www.chemistry.mcmaster.ca/aimpac/imagemap/imagemap.htm
[^5]: https://github.com/JuliaGPU/KernelAbstractions.jl

# Critical point finding benchmarks

Benchmarks were performed on a PC with an i7-5820k processor and an Nvidia GTX 1080 GPU, running CUDA toolkit 11.4.1, and NVIDIA driver 470.63.1. For `Extreme.jl`, running times were measured as the average of 10 runs using BenchmarkTools.jl[^6], while ext94 running times were measured using the multitime[^7] utility, averaging the execution times of 10 different runs. Compilation for ext94 was done on ifort 2021.7.1 20221019, and in the case of `Extreme.jl` Julia version 1.6.3 was used. It is important to note that all of the presented runs were measured after an initial warm-up run in order to remove the effect of Julia's JIT compiler. All measured times can therefore be considered the real-world execution time of each compiled program.

Each test was performed over a different WFN file, containing the topological information associated with a different molecule. These were selected based on an increasing number of atoms, electrons, and desired critical points. The tests were designed so that each program would find the same number of critical points.

An important factor to note is that because point search initialization utilities are not yet implemented in `Extreme.jl`, a random initialization was used for the search. On the other hand, preimplemented initializations are used in ext94. While this affects the critical points found, the execution time is invariant in `Extreme.jl`. This is because any given starting point will be iterated over the same amount of times using the same operations. In this case, the default number of iterations (15) was used.

| Name           | No. Atoms | No. electrons | No. Crit points | Execution Time ext94 | File Load Time Extreme.jl (CPU) | Calculation Time Extreme.jl (CPU) |
|----------------|-----------|---------------|-----------------|:--------------------:|:-------------------------------:|:---------------------------------:|
| H2O            | 3         | 10            | 5               | 0.066 [s]            | 0.476 [ms]                      | 0.0021 [s]                        |
| C2H6           | 8         | 18            | 15              | 0.297 [s]            | 1.669 [ms]                      | 0.0043 [s]                        |
| (H2O)24        | 4         | 18            | 7               | 0.115 [s]            | 1.298 [ms]                      | 0.0028 [s]                        |
| THF            | 13        | 40            | 27              | 1.261 [s]            | 7.089 [ms]                      | 0.0125 [s]                        |
| C6H6           | 12        | 42            | 25              | 1.256 [s]            | 8.286 [ms]                      | 0.0109 [s]                        |
| Cysteine       | 14        | 64            | 29              | 2.345 [s]            | 15.071 [ms]                     | 0.0198 [s]                        |
| Adenine        | 15        | 70            | 33              | 3.275 [s]            | 20.928 [ms]                     | 0.0262 [s]                        |
| Ti(H2O)6       | 19        | 80            | 37              | 6.057 [s]            | 24.049 [ms]                     | 0.0288 [s]                        |
| Phenanthroline | 22        | 94            | 49              | 14.415 [s]           | 39.274 [ms]                     | 0.0492 [s]                        |

| Name           | No. Atoms | No. electrons | No. Crit points | Execution Time ext94 | File Load Time Extreme.jl (GPU) | Calculation Time Extreme.jl (GPU) |
|----------------|-----------|---------------|-----------------|:--------------------:|:-------------------------------:|:---------------------------------:|
| H2O            | 3         | 10            | 5               | 0.066 [s]            | 0.528 [ms]                      | 0.0435 [s]                        |
| C2H6           | 8         | 18            | 15              | 0.297 [s]            | 1.726 [ms]                      | 0.0413 [s]                        |
| (H2O)24        | 4         | 18            | 7               | 0.115 [s]            | 1.358 [ms]                      | 0.0416 [s]                        |
| THF            | 13        | 40            | 27              | 1.261 [s]            | 7.213 [ms]                      | 0.0397 [s]                        |
| C6H6           | 12        | 42            | 25              | 1.256 [s]            | 8.361 [ms]                      | 0.0409 [s]                        |
| Cysteine       | 14        | 64            | 29              | 2.345 [s]            | 15.1 [ms]                       | 0.0342 [s]                        |
| Adenine        | 15        | 70            | 33              | 3.275 [s]            | 21.019 [ms]                     | 0.0337 [s]                        |
| Ti(H2O)6       | 19        | 80            | 37              | 6.057 [s]            | 23.468 [ms]                     | 0.0373 [s]                        |
| Phenanthroline | 22        | 94            | 49              | 14.415 [s]           | 38.217 [ms]                     | 0.0367 [s]                        |

Table 1 presents the results from a set of test cases designed to compare the execution times between ext94 and `Extreme.jl` executed in CPU.Table 2 shows the same comparison between ext94 and `Extreme.jl` executed in GPU. In order to control for file-loading overhead, `Extreme.jl` measured times are divided between File Load Time and Calculation Time. Our results show a decrease in execution time of several orders of magnitude when compared to the ext94 implementation. [When comparing the execution times of `Extreme.jl` in CPU and GPU, it can be noted that the GPU implementation shows a semi-constant execution time, which is - in general - higher than the CPU implementation, which grows as the topological complexity and number of critical points to search increase]. We believe this to be due caused by the overheads inherent to the CUDA architecture, such as kernel launch times and CPU-GPU communication latency.

In order to compare the exection time for each of the supported architectures in `Extreme.jl` as problem complexity grows, Table 2 shows the search of $n$ critical points over the topological space of the phen.wfn test file. 

| No. Crit Points | Execution Time Extreme.jl (CPU) | Execution Time Extreme.jl (GPU) | GPU speedup vs CPU |
|-----------------|---------------------------------|---------------------------------|----------------------|
| 50              | 0.045 [s]                       | 0.037 [s]                       | 1.21x                |
| 100             | 0.105 [s]                       | 0.037 [s]                       | 2.83x                |
| 500             | 0.469 [s]                       | 0.057 [s]                       | 8.22x                |
| 1,000           | 0.870 [s]                       | 0.093 [s]                       | 9.35x                |
| 5,000           | 6.244 [s]                       | 0.402 [s]                       | 15.53x               |
| 10,000          | 13.165 [s]                      | 0.806 [s]                       | 16.33x               |

The higher degree of parallelism present in a GPU architecture when compared to CPU processing causes a significant speedup when the problem complexity grows. While searching over thousands of critical points is not a common use case in QCT applications, this synthetic test shows that there is a viable use case for `Extreme.jl` running on different architectures based on the size of the underlying problem.

[^6]: https://github.com/JuliaCI/BenchmarkTools.jl
[^7]: https://tratt.net/laurie/src/multitime/

# Future development

In its current state, `Extreme.jl` serves as both a starting point and a proof of concept of what is possible when refactoring well-known code used for the field of Quantum Chemical Topology. Some of the next steps in order to broaden its scope and ease of use are the following:

- Implement utilities to propose starting points for the Newton-Raphson search algorithm.
- Implement algorithms for calculating the stable manifolds of the BCP.
- Generate an easy-to-use interface that does not require directly interacting with the Julia language.
- Create more concise code by using automatic code differentiation utilities such as Zygote.jl[zygote2019].