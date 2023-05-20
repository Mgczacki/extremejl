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
`Extreme.jl` is a *Quantum Chemical Topology* (QCT) package designed to compute *bound critical points* (BCP) of the electron charge distribution $\varrho(\mathbf{r})$.  It uses the Julia programming language[^1] to overcome processing bottlenecks by parallelizing electronic density calculations and finding critical points using the CUDA.jl package[^2]. `Extreme.jl` was created by analyzing algorithms from a legacy QCT package and porting them to Julia, taking advantage of this language's support for parallelism, GPU compute, and overall extensibility. The package is still in development, and future plans include implementing utilities for proposing starting points, stable manifold calculations, and creating a more user-friendly interface. The package's performance has been compared to the ext94 implementation and shows a significant decrease in execution time, especially when using GPU processing for large problem complexities.

[^1]: https://julialang.org
[^2]: https://github.com/JuliaGPU/CUDA.jl

# Statement of Need

The soundest way to examine different types of chemical interactions under the same physical basis is arguably via the examination of quantum mechanical observables and their expectation values. The analysis of the local and integrated values of these Dirac observables has resulted in the emergence of the field of theoretical computational chemistry known as *Quantum Chemical Topology* (QCT). The origins of QCT are based on the *Quantum Theory of Atoms in Molecules* (QTAIM) which provides a division of the three-dimensional space into basins which are related with the atoms of chemistry envisioned by Dalton at the beginning of the XIX century. Several tools of QCT take advantage of the partition of the 3D space defined by QTAIM, e.g., the *Interacting Quantum Atoms* (IQA) energy partition. The IQA partition has been very useful in the investigation of a wide diversity of interactions in chemistry, e.g., covalent, polar, ionic, intermolecular interactions as well as chemical bonding in solid state systems. Despite the recognized utility of the IQA analysis, this approach requires the integration of scalar fields over very irregular volume shapes. This endeavor is far from straightforward, and it involves a computational effort which severely hampers the applicability of IQA analysis for the study of electronic systems. At present, the IQA method is limited to systems containing only a few tens of atoms. The primary obstacle hindering the advancement of IQA exploration is the computation of the abovementioned integrals. Thus, novel algorithms and software must be developed to address this issue and improve the situation.

In previous works, several parallel implementations have been created in order to speed up the computations necessary to locate critical points. [@rodgz2013] presents an efficient method to obtain all critical points of the electron density, for which a vectorized and parallel algorithm was implemented using a message passing interface library; [@herdz2014] proposed a modification to Rodriguez’ algorithm in order to show how critical points of the electron density are found using GPUs, the C Language, and CUDA programming techniques. 

Calculation of bound critical points (BCP) requires numerical exploration of electron density by following the direction of the gradient towards the maximum of the density. `Extreme.jl` provides a Julia implementation to compute bound critical points of an electronic system, and it has been parallelized for exploring electronic density efficiently employing GPUs. `Extreme.jl` uses CUDA.jl and benefits from the Tullio.jl and KernelAbstractions.jl packages.
 

# High Performance and Expresiveness for Numerical Computations

The Julia programming language [@bezanson2012] has rapidly established itself as a highly promising language for scientific and high-performance computing. Among its most innovative features we can highlight its dynamic nature which, combined with its powerful features, makes it a highly productive language for code development. Additionally, its execution speed is comparable to that of statically typed languages [@sengupta2019]. These factors have led to Julia's increasing adoption in numerical computation and other applications that require parallel computation [@huo2020; @suslov2020; @huo2021], as well as its successful testing on high-performance architectures [@hunold2020; @weichen2021]. Several Julia packages support parallel computing and NVIDIA GPU programming, such as CUDA.jl [@besard2019], KernelAbstractions.jl[^3], and Tullio.jl [@tullio2022], which enables the writing of array operations using Einstein index notation. Overall, these advancements demonstrate the substantial potential of the Julia programming language in the field of scientific and high-performance computing.

[^3]: https://github.com/JuliaGPU/KernelAbstractions.jl
 
# The Extreme.jl Julia package

Here we present `Extreme.jl`, a highly-performant Julia package developed with the aim of analyzing the topological properties of the electron density of molecules and molecular clusters. In its current state, one of its main functionalites is to find critical points in the electron charge distribution of any given molecular system. The usual workflow with `Extreme.jl` looks as follows:

1. Load a molecule file (WFN or WFX format).
2. Select $n$ starting initialization points for critical point search.
3. Run the critical point finding algorithm.

There are other analysis tools implemented in the package such as functions for finding the electron density and its laplacian for any given points in 3D space.

# Parallelizing Quantum Chemical Topolical Analysis

The development of `Extreme.jl` was initiated through the reverse engineering of the ext94 (extreme) tool of the AIMPAC suite of software applications[^4]. Specifically, we focused on the critical-point finding routine contained within the source code. The original algorithm employs a Newton-Raphson optimization routine over the electron density space for a single point, which necessitates the use of nested loop structures consisting of inner products of vectors and matrix multiplications. The parallelization of these routines is a well-established technique that can be handled by various BLAS libraries. In our case, we opted to use Tullio.jl due to its numerous advantages:

- It facilitates the use of Einstein notation for defining multidimensional structures. This notation gives the code expressiveness, brevity, and an implementation that is remarkably close to the pure mathematical form of the underlying equations.
- Tullio.jl is built on the KernelAbstractions.jl[^5] package, which provides a transparent translation of high-level expressions into underlying operations across multiple different compute architectures. This strategy enables seamless generation of parallel CPU or GPU operations from the same source code.
- Through KernelAbstractions.jl, Tullio.jl supports mapping arbitrary Julia functions over subsets of a multidimensional data structure wich allowed us to create complex functionality for any of the supported architectures. Hence,  The use of these techniques reduces the complexity in writing CUDA kernels for GPU processing.
- KernelAbstractions.jl handles the parsing of expressions, ordering and subdividing computations, and concurrent execution of operations, freeing the programmer from specifying parallel waits, locks, or launch procedures for individual program steps.

We used the aforementioned technology stack to transform a sequential critical point finding algorithm into a parallelized approach capable of identifying an arbitrary number of critical points. Our methodology proceeded as follows:

1. We derived the relevant mathematical expressions from both the ext94 source code and equations from the Theory of Atoms in Molecules. By working from the innermost nested loops outward, we identified operations such as vector inner products, Hadamard products, matrix multiplications, transposes, inverses, and function mappings over data structures.
2. To enable concurrent processing of n points, we augmented the identified data structures by adding an additional dimension, generating matrices from n vectors and 3D tensors from n matrices. This "naive" parallelization was feasible due to the lack of operational or temporal dependencies between the processes of identifying critical points derived from distinct initial states.
3. For any calculations that did not involve basic operations such as sums or multiplications, we designed kernelized Julia functions. These functions can accept subsets of input data and other inputs to reproduce complex behavior, such as masking values or generating mappings.

[^4]: https://www.chemistry.mcmaster.ca/aimpac/imagemap/imagemap.htm
[^5]: https://github.com/JuliaGPU/KernelAbstractions.jl

# Critical point finding benchmarks

The benchmarks were conducted on a system comprising an i7-5820k processor and a Nvidia GTX 1080 GPU, using the CUDA toolkit 11.4.1 and the NVIDIA driver 470.63.1. In order to assess the performance of `Extreme.jl` and ext94, running times were recorded as the average of ten runs using the BenchmarkTools.jl package[^6], and the multitime[^7] utility was used to determine the average execution times of ten different runs, respectively. Compilation for ext94 was carried out using ifort 2021.7.1 20221019, while for `Extreme.jl`, the Julia version 1.6.3 was utilized. It should be noted that all reported runtimes were measured after an initial warm-up run to eliminate the influence of Julia's JIT compiler, and hence, can be considered as the real-world execution times for each compiled program.

To ensure comprehensive evaluation, we conducted tests using distinct WFN files, which correspond to different molecules and offer varying quantities of atoms, electrons, and critical points. The selection was made in a progressive manner to increase the complexity of the system. The tests were designed so that both programs would aim to identify the same number of critical points, thus ensuring equitable comparisons.

One important point to consider is that `Extreme.jl` currently lacks point search initialization utilities, and hence, a random initialization method was employed for the search process. Conversely, ext94 uses pre-implemented initialization methods. While this could impact the critical points discovered, the execution time of `Extreme.jl` remains unaffected. This is due to the fact that for any given starting point, the algorithm would iterate over the same number of times using the same operations. In the present case, the default number of iterations (15) was used.

| Name           | Atoms | Electrons | Critical<br > points | Execution Time <br > ext94 [s] | File Load Time <br >  Extreme.jl (CPU) [ms] | Calculation Time <br >  Extreme.jl (CPU) [s] |
|----------------|-----------|---------------|-----------------|:----------------:|:--------------------------:|:----------------------------:|
| H2O            | 3         | 10            | 5               |  0.066           |  0.476                     | 0.0021                       |
| C2H6           | 8         | 18            | 15              |  0.297           |  1.669                     | 0.0043                       |
| (H2O)24        | 4         | 18            | 7               |  0.115           |  1.298                     | 0.0028                       |
| THF            | 13        | 40            | 27              |  1.261           |  7.089                     | 0.0125                       |
| C6H6           | 12        | 42            | 25              |  1.256           |  8.286                     | 0.0109                       |
| Cysteine       | 14        | 64            | 29              |  2.345           | 15.071                     | 0.0198                       |
| Adenine        | 15        | 70            | 33              |  3.275           | 20.928                     | 0.0262                       |
| Ti(H2O)6       | 19        | 80            | 37              |  6.057           | 24.049                     | 0.0288                       |
| Phenanthroline | 22        | 94            | 49              | 14.415           | 39.274                     | 0.0492                       |

| Name           | Atoms | Electrons | Critical<br > points | Execution Time<br > ext94 [s] | File Load Time<br > Extreme.jl (GPU) [ms] | Calculation Time <br >  Extreme.jl (GPU) [s] |
|----------------|-----------|---------------|-----------------|:-----------------:|:--------------------------:|:---------------------------:|
| H2O            | 3         | 10            | 5               |  0.066            | 0.528                      | 0.0435                      |
| C2H6           | 8         | 18            | 15              |  0.297            | 1.726                      | 0.0413                      |
| (H2O)24        | 4         | 18            | 7               |  0.115            | 1.358                      | 0.0416                      |
| THF            | 13        | 40            | 27              |  1.261            | 7.213                      | 0.0397                      |
| C6H6           | 12        | 42            | 25              |  1.256            | 8.361                      | 0.0409                      |
| Cysteine       | 14        | 64            | 29              |  2.345            | 15.1                       | 0.0342                      |
| Adenine        | 15        | 70            | 33              |  3.275            | 21.019                     | 0.0337                      |
| Ti(H2O)6       | 19        | 80            | 37              |  6.057            | 23.468                     | 0.0373                      |
| Phenanthroline | 22        | 94            | 49              | 14.415            | 38.217                     | 0.0367                      |

Table 1 summarizes the results of the test cases comparing the execution times of ext94 and `Extreme.jl` on CPU. In Table 2, the same comparison is made for GPU execution. To account for file-loading overhead, `Extreme.jl` measured times are divided into File Load Time and Calculation Time. The results indicate a significant decrease in execution time, by several orders of magnitude, for `Extreme.jl` compared to ext94. However, when comparing the execution times of `Extreme.jl` on CPU and GPU, it can be observed that the GPU implementation exhibits a semi-constant execution time, which is generally higher than the CPU implementation. This could be due to the overheads inherent to the CUDA architecture, such as kernel launch times and CPU-GPU communication latency.

To assess the performance of `Extreme.jl` on different compute architectures as the problem complexity increases, Table 3 displays the execution time for searching $n$ critical points over the topological space of the phen.wfn (Phenanthroline) test file.

| Critical<br > Points | Execution Time<br >  Extreme.jl (CPU) [s] | Execution Time <br > Extreme.jl (GPU) [s] | GPU speedup vs CPU |
|-----------------|:----------------------------:|:---------------------------:|:--------------------:|
| 50              |  0.045                       | 0.037                       |  1.21x               |
| 100             |  0.105                       | 0.037                       |  2.83x               |
| 500             |  0.469                       | 0.057                       |  8.22x               |
| 1,000           |  0.870                       | 0.093                       |  9.35x               |
| 5,000           |  6.244                       | 0.402                       | 15.53x               |
| 10,000          | 13.165                       | 0.806                       | 16.33x               |

The greater parallelism available in a GPU architecture provides a substantial speed boost as the complexity of the problem increases. Although it may not be typical to search for thousands of critical points in QCT applications, it is not an unrealistic scenario. The results of our synthetic test demonstrate that `Extreme.jl` can be used effectively on different architectures depending on the scale of the underlying problem.

[^6]: https://github.com/JuliaCI/BenchmarkTools.jl
[^7]: https://tratt.net/laurie/src/multitime/

# Future development

The current version of `Extreme.jl` provides a starting point and demonstrates the potential of refactoring established code used in Quantum Chemical Topology. However, there are several areas where improvements can be made to increase its scope and user-friendliness. These include:

- Developing utilities that suggest starting points for the Newton-Raphson search algorithm.
- Introducing algorithms for computing the stable manifolds of the BCP.
- Creating a more user-friendly interface that does not require direct interaction with the Julia language.
- Using automatic code differentiation tools like Zygote.jl [@zygote2019] to produce more concise code.

# References
