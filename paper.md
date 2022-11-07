---
title: 'SoftwareName: A Julia package for Interacting Quantum Atoms'
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
date: 5 November 2022
bibliography: paper.bib

---

# Summary
`SoftwareName` is a *Quantum Chemical Topology* (QCT) package which computes *critical bound points* (BCP) of the electron charge distribution $\varrho(\mathbf{r})$ and determine the stable manifolds of the  BCP of a molecule. Julia programming language[^1] helps to address the main bottleneck for exploting  *Interacting Quantum Atoms* (IQA) energy partition through calculation of the several integrals in parallel using CUDA.jl package[^2]. `SoftwareName` was created by reverse-engineering the original Fortran code and porting it to Julia in order to harness its support for parallelism, GPU compute, and extensibility.

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

\noindent The function $\varrho_1(\mathbf{r}_{1};\mathbf{r}_{1}^{\prime})$ is the first-order reduced density matrix and $\delta_{\mathrm{AB}}$ is the Kronecker delta. The terms in equations (\ref{e_neta}) and (\ref{e_inter}) are easily interpretable. The quantity $T^{\Omega_{\mathrm{A}}}$ is the kinetic energy due to basin $\Omega_{\mathrm{A}}$ and $V_{\mathrm{e\tau}}^{\Omega_{\mathrm{A}}\Omega_{\mathrm{B}}}$ is the contribution to the potential energy due to (i) the electrons in basin $\Omega_{\mathrm{A}}$ and (ii) $\tau$, either electrons $\tau =\mathrm{e}$ or the nucleus $\tau = \mathrm{n}$, in basin $\Omega_{\mathrm{B}}$. Indeed, the Coulombic nature of the electronic Hamiltonian and the QTAIM partition, allows the electronic energy to be divided as put forward in equation (\ref{div_iqa}). 

The IQA partition has been very useful in the investigation of a wide diversity of interactions in chemistry, e.g., covalent, polar, ionic, intermolecular interactions as well as chemical bonding in solid state systems. Despite the recognised utility of the IQA analysis, formulae (\ref{cinetica})--(\ref{omega}) imply that the IQA approach requires the integration of scalar fields over very irregular volume shapes. In particular expression (\ref{e_e}) entails the six-dimensional integral over two QTAIM-basins. This endeavour is far from straightforward and it involves a computational effort which severely hampers the applicability of IQA for the study of electronic systems. Currently, the IQA can only be applied to systems with only a few tens of atoms. Indeed, the main bottleneck for the explotaition of the IQA approach is the calculation of the above mentioned integrals. Therefore new algorithms and software is needed for the amelioration of this situation. More specifically, one must determine the QTAIM-basins prior to perform the integrals involved in equations(\ref{cinetica})--(\ref{e_e}). One way to do it is via the determination of the IAS of the whole system. For this purpose, one must first find all the critical points of $\varrho(\mathbf{r})$ and second determine the stable manifolds of the BCP of the molecule or molecular critical under investigation. Therefore new algorithms and software is needed for the amelioration of this situation.


# High Performance and Expresiveness for Numerical Computations

In a short period of time the programming language [@bezanson2012] has positioned itself as one of the most promising programming languages for scientific and high-performance computing. Among its most innovative features we can highlight: ease of use as a dynamic language with powerful features that make it very productive for writing code, and at the same time, the developed code runs fast, at least as fast as code written in statically typed [@sengupta2019].  The use of this programming language in the area of numerical computation has increased in recent years as well as various applications exploiting parallel computation [@huo2020, @suslov2020, @huo2021] and has been tested in high-performance architectures [@hunold2020, @weichen2021]. Moreover several Julia packages support NVIDIA GPU's programming as CUDA.jl [@besard2017], KernelAbstractions.jl[^3] that allows to write GPU-like kernels targetting different execution backends and Tullio.jl[^4] to perform array operations written in index notation.

[^3]: https://github.com/JuliaGPU/KernelAbstractions.jl
[^4]: https://github.com/mcabbott/Tullio.jl
 


We herein report a software which performs both tasks and that afterwards can be expanded for the computation of the integrals in formulae (\ref{cinetica})--(\ref{e_e})
Such enhacement comprises the main contribution of this paper.

# References




