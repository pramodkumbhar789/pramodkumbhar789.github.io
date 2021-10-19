---
title: "Using physics informed neural network to predict strain in functionally graded material"
exclude: true
---
**Case I:**

**Problem statement:** Let us consider a rectangular domain $$\Omega$$ as shown in the diagram below:

<p align="center">
  <img width=250mm src="/assets/img/neumann_fgm_2d.png">
</p>

The bottom-left corner of the plate is fixed while the bottom edge is restricted to move in the $$x_2$$ direction.
The elasticiy of the plate is varying along $$x_2$$ direction and is given as:

$$ E(x_1,x_2) = \dfrac{E_0}{1+x_2}$$

A uniform traction $$T$$ is applied on the top boundry.

**Neural network architecture:** Let us say that $$ \tilde{u} (\mathbf{x}):\mathcal{\Omega} \to \mathcal{R}^2$$ is the neural network model which maps 
spatial coordinates vector $$\mathbf{x}$$ in domain $$\Omega$$ to a real valued vector in $$\mathcal{R}^2$$. $$\Omega$$ represents the set of points that consititute the plate while $$d\Omega$$ is the set of points on the boundary of the plate. 
In order to ensure Dirchlet
boundary conditions, the following transformation is applied:

$$ u(\mathbf{x}) = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} = \begin{bmatrix} x_2 \\ x_2\end{bmatrix} \odot \begin{bmatrix} \tilde{u}_1 \\ \tilde{u}_2 \end{bmatrix}$$

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
Decrease in loss (internal strain energy - work done by external force) with epoch count as the PINN is trained:
<p align="center">
  <img width=350mm src="/assets/img/neumann_fgm_loss.png">
</p>

Comparision between displacement fields obtained from Abaqus and using PINN method:
<p align="center">
  <img width=550mm src="/assets/img/neumann_u1_fgm.png">
</p>

<p align="center">
  <img width=550mm src="/assets/img/neumann_u2_fgm.png">
</p>

Comparision between stress fields obtained from Abaqus and using PINN method:
<p align="center">
  <img width=550mm src="/assets/img/neumann_S11_fgm.png">
</p>

<p align="center">
  <img width=550mm src="/assets/img/neumann_S22_fgm.png">
</p>

<p align="center">
  <img width=550mm src="/assets/img/neumann_S12_fgm.png">
</p>

**Case II:**

Let's consider the same domain and material properties with a different boundary condition. A uniform displacement 
boundary condition is applied on the top edge as shown in the diagram below:

<p align="center">
  <img width=250mm src="/assets/img/dirichlet_fgm_2d.png">
</p>

**Implementing boundary conditions:**

$$ u(\mathbf{x}) = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} = \begin{bmatrix} x_2 \\ x_2/3L + (3L-x_2)x_2\end{bmatrix} \odot \begin{bmatrix} \tilde{u}_1 \\ \tilde{u}_2 \end{bmatrix}$$

**Loss function:**

Since there is no traction in this case, the loss function will only have the internal strain energy component:

$$ \mathcal{L} = \sum_{\mathbf{x}_i \in \Omega}{\dfrac{1}{2}w(\mathbf{x}_i)\varepsilon(\mathbf{x}_i):\mathrm{C}(\mathbf{x}_i):\varepsilon(\mathbf{x}_i)} $$

where $$ w(\mathbf{x}_i)$$ and $$\mathrm{C}(\mathbf{x}_i) $$ are respectively the weight for integration over the domain and spatially varying stifness matrix.


**Results:**

Decrease in loss (internal strain energy) with epoch count as the PINN is trained:
<p align="center">
  <img width=350mm src="/assets/img/dirichlet_fgm_loss.png">
</p>

Comparision between displacement fields obtained from Abaqus and using PINN method:

<p align="center">
  <img width=550mm src="/assets/img/dirichlet_u1_fgm.png">
</p>


<p align="center">
  <img width=550mm src="/assets/img/dirichlet_u2_fgm.png">
</p>

Comparision between stress fields obtained from Abaqus and using PINN method:
<p align="center">
  <img width=550mm src="/assets/img/dirichlet_S11_fgm.png">
</p>

<p align="center">
  <img width=550mm src="/assets/img/dirichlet_S22_fgm.png">
</p>

<p align="center">
  <img width=550mm src="/assets/img/dirichlet_S12_fgm.png">
</p>
