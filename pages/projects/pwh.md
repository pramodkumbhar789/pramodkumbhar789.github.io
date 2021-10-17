---
title: "Plate with a circular hole under uniaxial loading"
exclude: true
---
**Problem statement:** Let us consider an infinite rectangular plate with a circular hole.
Let us assume that a circular hole with a radius $$0.1$$ is centered at $$(0, 0)$$. Moreover, the thickness of the plate and elasticity of the material
are assumed to be $$1$$ with appropriate units. Uniaxial traction $$ T $$ is applied in x-direction on both ends of the plate. The problem can be solved 
by considering a quarter of the infinite plate and applying Dirichlet boundary conditions as per symmetery. Moreover, for the purpose of simulation, the domain
is assumed to be finite as shown in the diagram below:

<p align="center">
  <img width=350mm src="/assets/img/pwh.png">
</p>

**Neural network architecture:** Let us say that $$ \tilde{u} (\mathbf{x}):\mathcal{\Omega} \to \mathcal{R}^2$$ is the neural network model which maps 
spatial coordinates vector $$(\mathbf{x})$$ to a real valued vector in $$\mathcal{R}^2$$. $$\Omega$$ represents the set of points that consititute the plate while $$d\Omega$$ is the set of points on the boundary of the plate. 
In order to ensure Dirchlet
boundary conditions, the following transformation is applied:

$$ u(\mathbf{x}) = \mathbf{x} \odot \tilde{u}(\mathbf{x})$$

where $$u(\mathbf{x})$$ gives displacement vector and $$\odot$$ represents Hadamard product.

**Material properties:** Plane stress conditions are assumed and hence, the stress - strain relation, with the assumption of elastcity to be unity, is given as:

$$ \mathbf{\sigma} = \mathrm{C}\mathbf{\varepsilon}$$

$$ \begin{bmatrix}\sigma_{11} \\ \sigma_{22} \\ \sigma_{12} \end{bmatrix} = \begin{bmatrix} \dfrac{1}{1-\nu^2} & \dfrac{\nu}{1-\nu^2} & 0 \\ \dfrac{\nu}{1-\nu^2} & \dfrac{1}{1-\nu^2} & 0 \\ 0 & 0 & \dfrac{1}{(1+\nu)}\end{bmatrix}\begin{bmatrix}\varepsilon_{11} \\ \varepsilon_{22} \\ \varepsilon_{12}\end{bmatrix}$$

where $$\sigma, \mathrm{C}$$ and $$\varepsilon$$ represent stress, stiffness matrix and strain respectively.
The radius of the hole is assumed to be $$0.1$$ unit and traction $$T=1$$ is applied on right edge.
Moreover, nominal dimension $$L$$ of the plate is assumed to be $$1$$. Poisson's ratio is assumed to be $$0.3$$.


**Loss function and mesh:** The loss for training the model is given as:

$$ \mathcal{L} = \sum_{\mathbf{x}_i \in \Omega}{\dfrac{1}{2}w(\mathbf{x}_i)\varepsilon(\mathbf{x}_i):\mathrm{C}:\varepsilon(\mathbf{x}_i)} - \sum_{\mathbf{x}_i \in d\Omega} w'(\mathbf{x}_i)T(\mathbf{x}_i)\cdot \mathbf{u}(\mathbf{x}_i)$$

where $$ w(\mathbf{x}_i)$$ and $$ w'(\mathbf{x}_i)$$ are weights for integration over the domain and the boundary respectively. Mesh and weights are shown in the diagram below:


<p align="center">
  <img width=350mm src="/assets/img/pwh_mesh.png">
</p>

**Results:** The physics informed neural network is trained for 2500 epochs. The loss which is `internal energy - work done` decreased rapidly at the begining of training
process and then saturated to a value of approximately $$-0.512$$. The decline in loss vs. number of epochs is shown in 
figure below:

<p align="center">
  <img width=350mm src="/assets/img/pwh_loss.png">
</p>

The prediction of primary variables $$u_1$$ and $$u_2$$ is shown in the plots below. 
<p align="center">
  <img width=350mm src="/assets/img/u_1.png">
  <img width=350mm src="/assets/img/u_2.png">
</p>

The comparision between analytical and predicted stress fields can be observed in the following
figures. The values predicted are in good agreement with the analytical solution.
<p align="center">
  <img width=700mm src="/assets/img/a_sxx.png">
</p>

<p align="center">
  <img width=700mm src="/assets/img/a_syy.png">
</p>

<p align="center">
  <img width=700mm src="/assets/img/a_sxy.png">
</p>
