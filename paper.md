---
title: 'Extreme.jl: A Julia package for Quantum Chemical Topology' 
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
`Extreme.jl` is a *Quantum Chemical Topology* (QCT) package designed to compute *bound critical points* (BCP) of the electron charge distribution $\varrho(\mathbf{r})$.  It uses the Julia programming language[^1] to overcome processing bottlenecks by parallelizing electronic density calculations and finding critical points using the CUDA.jl package[^2]. `Extreme.jl` was created by analyzing algorithms from a legacy QCT package and porting them to Julia, taking advantage of its support for parallelism, GPU compute, and overall extensibility. The package is still in development, and future plans include implementing utilities for proposing starting points, stable manifold calculations, and creating a more user-friendly interface. The package's performance has been compared to the ext94 implementation and shows a significant decrease in execution time, especially when using GPU processing for large problem complexities.

[^1]: https://julialang.org
[^2]: https://github.com/JuliaGPU/CUDA.jl

# Statement of Need

The soundest way to examine different sorts of chemical interactions under the same physical basis is arguably via the examination of quantum mechanical observables and their expectation values. The analysis of the local and integrated values of these Dirac observables has resulted in the emerging of the field of theoretical computational chemistry known as *Quantum Chemical Topology* (QCT). The origins of QCT are based on the *Quantum Theory of Atoms in Molecules* (QTAIM) which provides a division of the three-dimensional space into basins which are related with the atoms of chemistry envisioned by Dalton at the beginning of the XIX century. Several tools of QCT take advantage of the partition of the 3D space defined by QTAIM, e.g., the *Interacting Quantum Atoms* (IQA) energy partition. The IQA partition has been very useful in the investigation of a wide diversity of interactions in chemistry, e.g., covalent, polar, ionic, intermolecular interactions as well as chemical bonding in solid state systems. Despite the recognized utility of the IQA analysis this approach requires the integration of scalar fields over very irregular volume shapes. This endeavor is far from straightforward, and it involves a computational effort which severely hampers the applicability of IQA for the study of electronic systems. Currently, the IQA can only be applied to systems with only a few tens of atoms. Indeed, the main bottleneck for the exploration of the IQA approach is the calculation of the abovementioned integrals. Therefore, new algorithms and software is needed for the amelioration of this situation.
In order to speed up computations necessary to locate critical points certain methods have been implemented in parallel. @rodgz2013 2013 presents an efficient algorithm to obtain all critical points of the electron density, vectorized and parallel version of the algorithm was implemented using message passing interface library; @herdz2014 propose a Rodriguez’ algorithm modification in order to show how critical points of the electron density are found using GPUs, this algorithm was implemented in C and CUDA programming techniques. 
Calculation of bound critical points (BCP) requires numerical exploration of electron density by following the direction of the gradient towards the maximum of the density. `Extreme.jl provides a Julia implementation to compute bound critical points of an electronic system, it has been parallelized for exploring electronic density efficiently employing GPUs. Extreme.jl uses CUDA.jl and benefits from Tullio.jl and KernelAbstractions.jl packages.
 

<!-- Chemical bonding is one of the most fundamental concepts in chemistry.
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
\end{align} -->

<!-- \noindent The function $\varrho_1(\mathbf{r}_{1};\mathbf{r}_{1}^{\prime})$ is the first-order reduced density matrix and $\delta_{\mathrm{AB}}$ is the Kronecker delta. The terms in equations (\ref{e_neta}) and (\ref{e_inter}) are easily interpretable. The quantity $T^{\Omega_{\mathrm{A}}}$ is the kinetic energy due to basin $\Omega_{\mathrm{A}}$ and $V_{\mathrm{e\tau}}^{\Omega_{\mathrm{A}}\Omega_{\mathrm{B}}}$ is the contribution to the potential energy due to (i) the electrons in basin $\Omega_{\mathrm{A}}$ and (ii) $\tau$, either electrons $\tau =\mathrm{e}$ or the nucleus $\tau = \mathrm{n}$, in basin $\Omega_{\mathrm{B}}$. Indeed, the Coulombic nature of the electronic Hamiltonian and the QTAIM partition, allows the electronic energy to be divided as put forward in equation (\ref{div_iqa}). 

The IQA partition has been very useful in the investigation of a wide diversity of interactions in chemistry, e.g., covalent, polar, ionic, intermolecular interactions as well as chemical bonding in solid state systems. Despite the recognised utility of the IQA analysis, formulae (\ref{cinetica})--(\ref{omega}) imply that the IQA approach requires the integration of scalar fields over very irregular volume shapes. In particular expression (\ref{e_e}) entails the six-dimensional integral over two QTAIM-basins. This endeavour is far from straightforward and it involves a computational effort which severely hampers the applicability of IQA for the study of electronic systems. Currently, the IQA can only be applied to systems with only a few tens of atoms. Indeed, the main bottleneck for the explotaition of the IQA approach is the calculation of the above mentioned integrals. Therefore new algorithms and software is needed for the amelioration of this situation. Such enhacement comprises the main contribution of this development.

More specifically, one must determine the QTAIM-basins prior to perform the integrals involved in equations(\ref{cinetica})--(\ref{e_e}). One way to do it is via the determination of the IAS of the whole system. For this purpose, one must first find all the critical points of $\varrho(\mathbf{r})$ and second determine the stable manifolds of the BCP of the molecule or molecular critical under investigation. Therefore new algorithms and software is needed for the amelioration of this situation. We herein report a software which performs both tasks and that afterwards can be expanded for the computation of the integrals in formulae (\ref{cinetica})--(\ref{e_e}). -->


# High Performance and Expresiveness for Numerical Computations

The Julia programming language [@bezanson2012] has rapidly established itself as a highly promising language for scientific and high-performance computing. Among its most innovative features we can highlight its dynamic nature which, combined with its powerful features, makes it a highly productive language for code development. Additionally, its execution speed is comparable to that of statically typed languages [@sengupta2019]. These factors have led to Julia's increasing adoption in numerical computation and other applications that require parallel computation [@huo2020; @suslov2020;@huo2021], as well as its successful testing on high-performance architectures [@hunold2020; @weichen2021]. Several Julia packages support parallel computing and NVIDIA GPU programming, such as CUDA.jl [@besard2017], KernelAbstractions.jl[^3], and Tullio.jl [@tullio2022], which enables the writing of array operations using Einstein index notation. Overall, these advancements demonstrate the substantial potential of the Julia programming language in the realm of scientific and high-performance computing.

[^3]: https://github.com/JuliaGPU/KernelAbstractions.jl
 
# The Extreme.jl Julia package

Here we present `Extreme.jl`, a highly-performant Julia package developed with the aim of analyzing the topological properties of the electron density of molecules and molecular clusters. In its current state, one of its main functionalites is to find critical points in the electron charge distribution of any given molecular system. The usual workflow with `Extreme.jl` looks as follows:

1. Load a molecule file (WFN or WFX format).
2. Select $n$ starting initialization points for critical point search.
3. Run the critical point finding algorithm.

There are other analysis tools implemented in the package such as functions for finding the electron density and its laplacian for any given points in 3D space.

# Parallelizing Quantum Chemical Topolical Analysis

The development of `Extreme.jl` was initiated through the reverse engineering of the ext94 (extreme) tool of the AIMPAC suite of software applications[^4]. Specifically, we focused on the critical-point finding routine contained within the source code. The original algorithm employs a Newton-Raphson optimization routine over the electron density space for a single point, which necessitates the use of nested loop structures consisting of inner products of vectors and matrix multiplications. The parallelization of these routines is a well-established technique that can be handled by various BLAS libraries. In our case, we opted to use Tullio.jl due to its numerous advantages:

- It facilitates the use of Einstein notation for defining multidimensional structures. This gives the code expressiveness, brevity, and an implementation that is remarkably close to the pure mathematical form of the underlying equations.
- Tullio.jl is built on the KernelAbstractions.jl[^5] package, which provides a transparent translation of high-level expressions into underlying operations across multiple different compute architectures. This enables seamless generation of parallel CPU or GPU operations from the same source code.
- Through KernelAbstractions.jl, Tullio.jl supports mapping arbitrary Julia functions over subsets of a multidimensional data structure. This allowed us to create complex functionality for any of the supported architectures. Crucially, it allowed us to forgo writing CUDA kernels for GPU processing.
- KernelAbstractions.jl handles the parsing of expressions, ordering and subdividing computations, and concurrent execution of operations, freeing the programmer from specifying parallel waits, locks, or launch procedures for individual program steps.

We used the aforementioned technology stack to transform a sequential critical point finding algorithm into a parallelized approach capable of identifying an arbitrary number of critical points. Our methodology proceeded as follows:

1. We derived the relevant mathematical expressions from both the ext94 source code and equations from the Theory of Atoms in Molecules. By working from the innermost nested loops outward, we identified operations such as vector inner products, Hadamard products, matrix multiplications, transposes, inverses, and function mappings over data structures.
2. To enable concurrent processing of n points, we augmented the identified data structures by adding an additional dimension, generating matrices from n vectors and 3D tensors from n matrices. This "naive" parallelization was feasible due to the lack of operational or temporal dependencies between the processes of identifying critical points derived from distinct initial states.
3. For any calculations that did not involve basic operations such as sums or multiplications, we designed kernelized Julia functions. These functions can accept subsets of input data and other inputs to reproduce complex behavior, such as masking values or generating mappings.

[^4]: https://www.chemistry.mcmaster.ca/aimpac/imagemap/imagemap.htm
[^5]: https://github.com/JuliaGPU/KernelAbstractions.jl

# Critical point finding benchmarks

The benchmarks were conducted on a system comprising an i7-5820k processor and an Nvidia GTX 1080 GPU, using CUDA toolkit 11.4.1 and NVIDIA driver 470.63.1. In order to assess the performance of `Extreme.jl` and ext94, running times were recorded as the average of ten runs using the BenchmarkTools.jl package[^6], and the multitime[^7] utility was used to determine the average execution times of ten different runs, respectively. Compilation for ext94 was carried out using ifort 2021.7.1 20221019, while for `Extreme.jl`, the Julia version 1.6.3 was utilized. It should be noted that all reported runtimes were measured after an initial warm-up run to eliminate the influence of Julia's JIT compiler, and hence, can be considered as the real-world execution times for each compiled program.

To ensure comprehensive evaluation, we conducted tests using distinct WFN files, which correspond to different molecules and offer varying quantities of atoms, electrons, and critical points. The selection was made in a progressive manner to increase the complexity of the system. The tests were designed so that both programs would aim to identify the same number of critical points, thus ensuring equitable comparisons.

One important point to consider is that `Extreme.jl` currently lacks point search initialization utilities, and hence, a random initialization method was employed for the search process. Conversely, ext94 uses pre-implemented initialization methods. While this could impact the critical points discovered, the execution time of `Extreme.jl` remains unaffected. This is due to the fact that for any given starting point, the algorithm would iterate over the same number of times using the same operations. In the present case, the default number of iterations (15) was used.

| Name           | Atoms | Electrons | Critical points | Execution Time ext94 [s] | File Load Time Extreme.jl (CPU) [ms] | Calculation Time Extreme.jl (CPU) [s] |
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

| Name           | Atoms | Electrons | Critical points | Execution Time ext94 | File Load Time Extreme.jl (GPU) | Calculation Time Extreme.jl (GPU) |
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

Table 1 summarizes the results of the test cases comparing the execution times of ext94 and `Extreme.jl` on CPU. In Table 2, the same comparison is made for GPU execution. To account for file-loading overhead, `Extreme.jl` measured times are divided into File Load Time and Calculation Time. The results indicate a significant decrease in execution time, by several orders of magnitude, for `Extreme.jl` compared to ext94. However, when comparing the execution times of `Extreme.jl` on CPU and GPU, it can be observed that the GPU implementation exhibits a semi-constant execution time, which is generally higher than the CPU implementation. This could be due to the overheads inherent to the CUDA architecture, such as kernel launch times and CPU-GPU communication latency.

To assess the performance of `Extreme.jl` on different compute architectures as the problem complexity increases, Table 3 displays the execution time for searching $n$ critical points over the topological space of the phen.wfn (Phenanthroline) test file.

| Critical Points | Execution Time Extreme.jl (CPU) | Execution Time Extreme.jl (GPU) | GPU speedup vs CPU |
|-----------------|---------------------------------|---------------------------------|----------------------|
| 50              | 0.045 [s]                       | 0.037 [s]                       | 1.21x                |
| 100             | 0.105 [s]                       | 0.037 [s]                       | 2.83x                |
| 500             | 0.469 [s]                       | 0.057 [s]                       | 8.22x                |
| 1,000           | 0.870 [s]                       | 0.093 [s]                       | 9.35x                |
| 5,000           | 6.244 [s]                       | 0.402 [s]                       | 15.53x               |
| 10,000          | 13.165 [s]                      | 0.806 [s]                       | 16.33x               |

The greater parallelism available in a GPU architecture provides a substantial speed boost as the complexity of the problem increases. Although it may not be typical to search for thousands of critical points in QCT applications, it is not an unrealistic scenario. The results of our synthetic test demonstrate that `Extreme.jl` can be used effectively on different architectures depending on the scale of the underlying problem.

[^6]: https://github.com/JuliaCI/BenchmarkTools.jl
[^7]: https://tratt.net/laurie/src/multitime/

# Future development

The current version of `Extreme.jl` provides a starting point and demonstrates the potential of refactoring established code used in Quantum Chemical Topology. However, there are several areas where improvements can be made to increase its scope and user-friendliness. These include:

- Developing utilities that suggest starting points for the Newton-Raphson search algorithm.
- Introducing algorithms for computing the stable manifolds of the BCP.
- Creating a more user-friendly interface that does not require direct interaction with the Julia language.
- Using automatic code differentiation tools like Zygote.jl [@zygote2019] to produce more concise code.
