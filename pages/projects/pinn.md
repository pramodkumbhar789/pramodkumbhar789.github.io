---
title: "Physics informed neural network (PINN)"
exclude: true
---
**Backgroud:**
The term "neural network" encompasses a wide range of machine learing models with one similarity i.e., so called "neurons" as the basic building block. The variety in the family of neural network models can be in various forms such as their architecture (arrangement of neurons and layers), learning process (supervised or unsupervised), purpose of the model (regression or classification). Physics informed neural network is yet another variant in this family of neural networks. Currently in its infancy, PINN can potentially disrupt the way differential equations are solved.

Diffrential equations are indispensable to computational engineering. Irrespective of engineering subdomains, diffrential equations appear in one form or the other. Often one needs to numerically solve such equations on the domain of interest. For example, one needs to solve the <img src="https://render.githubusercontent.com/render/math?math=\nabla \cdot \sigma = 0"> under certain boundary conditions in order to obtain the elastic mechanical response of a solid object. The state of the art of technique for numerically solving such differential equations is finite element method. While finite element method comes with its own pros and cons, physics informed neural network is a promising alternative to solve differential equations. PINN has an apparent advantage of being a meshless method over FEM. 

**Physics Informed Neural Network:**
In abstract sense, physics informed neural network tries to optimize the parameters of a parametric function that approximates the solution of diffrential equation on a given domain. The network does the optimization by minimising an objective which is called the loss function. The prefix "Physics Informed" comes from the fact that the loss function is obtained using physics of the diffrential equation being solved. Often, the loss is the functional corresponding to the differential equation. Hence, in machine learning jargon, the neural network model is said to be trained using unsupervised learning as there is no target variable to compute the loss. Let's try to understand this approach using a standard problem from the domain of solid mechanics.

**Problem Formulation (Functionally graded elasticity with Dirichlet BC):**
Let's solve the following diffrential equation under the given boundary condition using physics informed neural network:

$$\dfrac{\partial}{\partial x}\left( \dfrac{1}{1 + x} \dfrac{\partial u}{\partial x}\right)=0$$

such that

$$u(x=0) =0, u(x=1) = 1$$


The differential equation given above will appear in solving the mechanical response of elastic bar whose elastic modulus varies as $$1/(1+x)$$. The left end of the bar is fixed and the right end is given a unit displacement boundary condition. The solution $$(u(x))$$ to the differential equation is the 1D displacement field of the bar and can be analytical derived to be:

$$ u(x) = \dfrac{x^2+2x}{3} $$

In the following section, we describe how to solve this diffrential equation using PINN.

**Preparation of the dataset:**
Before jumping into the details of PINN and the loss function, let us understand the dataset on which the network is trained. The input to the PINN is going to be the spatial coordinates of nodes in the domain. Hence, follow are the steps to creating the input dataset.

    1. Identify the domain on which the diffrential equation is to be solved.
    2. Discretize the domain into a set of nodes.
    3. Compute weights corresponding to the nodes for carrying out numerical integration.
    
Hence, in the case of 1D domain, the dataset is coordinates of nodes from $$x=0$$ to $$x=1$$. If the nodes are distributed uniformly in the domain, the corresponding weights for integration will be constant. The number of nodes is a choice to be made by the user.

There is no target dataset in case of PINN as the network is trained in unsupervised manner.

**Architecture of the PINN:**
<p align="center">
  <img width=500mm src="/assets/img/pinn_1d.png">
</p>

While the PINN framework offers a lot of flexibility in deciding the model architecture, a few of such parameters get determined by the task at hand. For example, the number of nodes in the input layer of the model must be equal to the dimension of domain. Similarly, the number of nodes in the output layer should be equal to the number of target variables. In the case of 1D solid mechanics problem, both the input and output layer has one neuron each. While the core of the PINN model can be traditional fully connected feed forward architecture, some changes are required in order to implement (dirichlet) boundary conditions. In general following steps are to be followed:

    1. Decide number of nodes in the input and output layers depending on dimensionality of problem.
    2. Decide number of hidden layers, corresponding number of nodes, activation values etc.
    4. Design the architecture in a way that dirichlet boundary conditions are satisfied.
    
 **Implementing Dirichlet BC:**
 Let us assume that there is a blackbox neural network function $$\tilde{u}(x):\mathcal{\Omega} \to \mathcal{R}$$ which maps points in the domain to a real number.
 Now, Dirichlet BCs can be implemented by the following transformation:
  
  $$ u(x) = x + x(1-x)\tilde{u}(x)$$
  
Hence,
<p align="center">
$$ u(0) = 0 $$ and $$ u(1) = 1$$
</p>

In the diagram above, connections in black colour correspond to $$\tilde{u}(x)$$, while the 
connection in blue is the transformation applied over $$\tilde{u}(x)$$ to get $$u(x)$$

**Note:**
One `cannot` do any transformation to implement Dirichlet BC such as:

 $$ u(x) = x^2 + x^2(1-x^2)\tilde{u}(x)$$

It is because the above transformation will not only ensure Dirichlet BC, rather it will simultaneosly put constraints on the displacement gradient ($$u'(x)$$) at $$x=0$$ and $$x=1$$.
 
 **Training Loss of PINN:**
 As highlighted earlier, it is the loss used that sets the PINN model apart from rest of the neural network family. The loss is basically the functional of the underlying differential equation. Hence, the loss function will also change from one differential equation to the other. The neural network model trains (fits the differential equation) by minimising the functional (loss) of the differential equation. However, the tricky part is that the functional/loss often depends not only the primary target variables, rather it's derivative as well. For example in case of 1D diffrential equation given before, the loss function defined on the discretized domain is:

  $$\mathcal{L}=\sum_{i}^N\dfrac{1}{2(x_i+1)}\left(\left.\dfrac{\partial u}{\partial x}\right\rvert_{x_{i}}\right)^2$$
  
In the context of solid mechanics, the loss above is the internal strain energy of the system. As it is evident, the loss involves computing the strain i.e. the derivative of $$u$$ w.r.t $$x$$. Since, the neural network is basically a parametric function, it is possible to compute the derivate of the target variable $$u$$ w.r.t. the input variable $$x$$ in terms of the parameters of the neural network. While computing such derivatives can be a tedious task given the complexity of the parametric neural network, the tensorflow library turns out to be the saviour. The derivative can be easily computed using the technique called "automatic differention" which is implemented in the tensorflow library. While we do not go into the details of automatic differention, we provide a brief in the next section.


**Automatic diffrentiation:**
Automatic diffrentiation is at core of any deep learning library such as tensorflow or pyTorch. In order to understand automatic diffrentiation, let's take a step back to reflect on how neural networks (or any such parametric function) is built in tensorflow. To build a neural network, one uses the so called building block parametric operators in tensorflow. Examples of a such operators are nothing but matrix multiplication,  addition and an activation function. It is to be noted that the gradient to all such operators is already defined in tensorflow. Let's look at the matrix mulplication operator for example. Let us say there is a matrix multiplication operator $$\phi_W$$, which when applied on $$x$$, it returns the matrix multiplication result $$Wx$$ (assuming compatible size of $$W$$ and $$x$$). Hence, whenever one needs to compute the derivative of this operation, the result is the matrix $$W$$. 

In a nutshell, the parametric PINN model is built of such fundamental operators whose derivtive is well defined in terms of parameters. Hence, when one needs to compute the derivative, one just needs to multply the derivatives of the fundamental operations in a logical way.

**Training the PINN:**
Now that we have understood the building fundamentals of building a PINN, let's us look at the complete pipeline and steps involved in training the model:

    1. Create the dataset
    2. Decide the PINN architecture
    3. Initialize the PINN model by setting random values to parameters (weights & biases)
    4. Iterate to train the PINN
      a. Compute the target variable at nodal coordinates in the forward pass of PINN
      b. Compute required gradients of the target variable/s w.r.t input variable/s
      c. Compute loss 
      d. Update parameters (weights and biases) of PINN
    5. Predict final solution at the nodal coordinates using the trained PINN model
    
 **Results:**
Following the steps described before, the given 1D differential equation is solved. The comparision between the PINN solution and analytical solution is shown in the figure below. 
<p align="center">
  <img width=500mm src="/assets/img/pinn_1d_dirchlet_fgm.png">
</p>

Now we will see some more examples on solving linear elasticity problem under different boundary conditions and body force. Then we will discuss a 1D case where PINN fails to find the true solution. 

In all the examples discussed hereafter, we assume the domain to be from $$x=0$$ to $$x=1$$ (i.e. a 1D rod).

**Functionally graded elasticity with Neumann BC:**
The governing equation and the boundary conditions are given as:

$$ \dfrac{\partial }{ \partial x}\left( E(x) \dfrac{\partial u}{\partial x}\right) = 0 $$

where

$$ E(x) = \dfrac{1 }{1+x}$$

such that:

$$ u(0) = 0, \quad T(1) = E(1) \left.\dfrac{\partial u}{\partial x}\right|_{x=1} = 1$$

The analytical solution for this case is given as:

$$ u(x) = \dfrac{2x + x^2}{2}$$

While the Dirichlet boundary conditions were implicity ensured by the neural network 
architecture, implementation of Neumann boundary conditions requires modification of the loss
function as one would expect:

$$ \mathcal{L} = \sum_{i}^N\dfrac{1}{2(x_i+1)}\left(\left.\dfrac{\partial u}{\partial x}\right\rvert_{x_{i}}\right)^2 - T(1)u(1)$$

In terms of the most fundamental of energy balance:
<center>
        <b><em>Loss = Internal Energy - Work Done by External Forces</em></b>
</center>

Hence, unlike the other use cases of neural network, loss function does have a physical significance in case of
physics informed neural network. The loss can be expected to converge to zero in situations where internal energy stored is equal to the work done.
However, internal energy is more often less than the work done unless as the loss function does not account for quasi-static assumption. Hence, the loss function
usually converges to a negative value, which basically is the unaccounted energy or the energy lost due to the static nature of diffrential equations.
The unacounted energy will also include the work done by reaction force in case of Dirchlet boundary conditions. 

The comparison between analytical and PINN solution is shown in the figure below:
<p align="center">
  <img width=500mm src="/assets/img/pinn_1d_neumann_fgm.png">
</p>

**Constant elasticity, body force and Neumann BC:**
We have seen how to modify both the neural network architecture and the loss function in case of different Neumann and Dirchlet
boundary conditions. Let us look at one more example with body force. In this case, we will assume the elastic modulus to be $$1$$ over the
domain. Other properties such as area of cross-section, mass density are assumed to be unity. Hence the governing equations are:

$$ \dfrac{\partial^2 u}{\partial^2 x} + 1= 0$$

with the boundary conditions:

$$ u(0) = 0, \quad T(1) = \left.\dfrac{\partial u}{\partial x}\right|_{x=1} = 1$$

The analytical solution for this case is given by:

$$ u(x) = \dfrac{4x - x^2}{2}$$

In order to account for the effect of body force, the loss function in this case must be modified. The work done by body force should also be subtracted from the loss.
This modification of loss function also becomes apparent when one derives the functional of the above governing equation with body force. Hence, the loss for this case is:

$$ \mathcal{L} = \sum_{i}^N\left(\dfrac{1}{2}\left(\left.\dfrac{\partial u}{\partial x}\right\rvert_{x_{i}}\right)^2  - u_i\right)- T(1)u(1)$$

Unsurprisingly, the PINN model is able to match the analytical solution in this case as well. The comparison is shown in figure below:
<p align="center">
  <img width=500mm src="/assets/img/pinn_1d_neumann_bf.png">
</p>