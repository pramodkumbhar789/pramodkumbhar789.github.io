---
title: "Using physics informed neural network to predict strain in functionally graded material"
exclude: true
---
**Problem statement:** Let us consider a rectangular domain $$\Omega$$ as shown in the diagram below:

<p align="center">
  <img width=250mm src="/assets/img/fgm_2d.png">
</p>

The bottom-left corner of the plate is fixed while the bottom edge is restricted to move in the $$x_2$$ direction.
The elasticiy of the plate is varying along $$x_2$$ direction and is given as:

$$ E(x_1,x_2) = \dfrac{E_0}{1+x_2}$$

A uniform load $$T$$ with dimension force per unit length is applied on the top boundry.

**Neural network architecture:** Let us say that $$ \tilde{u} (\mathbf{x}):\mathcal{\Omega} \to \mathcal{R}^2$$ is the neural network model which maps 
spatial coordinates vector $$\mathbf{x}$$ in domain $$\Omega$$ to a real valued vector in $$\mathcal{R}^2$$. $$\Omega$$ represents the set of points that consititute the plate while $$d\Omega$$ is the set of points on the boundary of the plate. 
In order to ensure Dirchlet
boundary conditions, the following transformation is applied:

$$ u(\mathbf{x}) = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} = \begin{bmatrix} x_1 + x_2 \\ x_2\end{bmatrix} \odot \begin{bmatrix} \tilde{u}_1 \\ \tilde{u}_2 \end{bmatrix}$$

where $$u(\mathbf{x})$$ gives displacement vector and $$\odot$$ represents Hadamard product.

**Material properties:**  Under plane stress condition, the stress - strain relation is given as:

$$ \mathbf{\sigma}(\mathbf{x}) = \mathrm{C}(\mathbf{x})\mathbf{\varepsilon} (\mathbf{x})$$

$$ \begin{bmatrix}\sigma_{11} \\ \sigma_{22} \\ \sigma_{12} \end{bmatrix} = \dfrac{E_0}{1+x_2}\begin{bmatrix} \dfrac{1}{1-\nu^2} & \dfrac{\nu}{1-\nu^2} & 0 \\ \dfrac{\nu}{1-\nu^2} & \dfrac{1}{1-\nu^2} & 0 \\ 0 & 0 & \dfrac{1}{1+\nu}\end{bmatrix}\begin{bmatrix}\varepsilon_{11} \\ \varepsilon_{22} \\ \varepsilon_{12}\end{bmatrix}$$

where $$\sigma, \mathrm{C}$$ and $$\varepsilon$$ represent stress, stiffness matrix and strain respectively.
Let us assume the parameter $$E_0$$ to be $$1$$.

**Loss function and mesh:** The loss for training the model is given as:

$$ \mathcal{L} = \sum_{\mathbf{x}_i \in \Omega}{\dfrac{1}{2}w(\mathbf{x}_i)\varepsilon(\mathbf{x}_i):\mathrm{C}(\mathbf{x}_i):\varepsilon(\mathbf{x}_i)} - \sum_{\mathbf{x}_i \in d\Omega} w'(\mathbf{x}_i)T(\mathbf{x}_i)\cdot \mathbf{u}(\mathbf{x}_i)$$

where $$ w(\mathbf{x}_i)$$ and $$ w'(\mathbf{x}_i)$$ are weights for integration over the domain and the boundary respectively.


**Results:**

<p align="center">
  <img width=350mm src="/assets/img/fgm_loss.png">
</p>


<p align="center">
  <img width=200mm src="/assets/img/u1_fgm.png">
  <img width=200mm src="/assets/img/u2_fgm.png">
</p>

<p align="center">
  <img width=200mm src="/assets/img/s11_fgm.png">
  <img width=200mm src="/assets/img/s22_fgm.png">
  <img width=200mm src="/assets/img/s12_fgm.png">
</p>
