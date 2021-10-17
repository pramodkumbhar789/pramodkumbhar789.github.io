---
title: "On challenges in training physics informed neural network loss"
exclude: true
---

**What is the right activation function for PINNs?**

Activation functions are important component of neural network. While all of them are there to induce non-linearity 
in neural networks, the choice often has an impact on trainability of neural networks. For example, sigmoid and tanh
activations are losing popularity due to their saturating regions on both ends of their domains.The saturating regions
result in problem called "vanishing grdients". Vanishing gradients means the derivative of loss w.r.t. the model weights
turning out to be zero. Therefore, the activation function "ReLu" became quite problem as it does not have saturation regions
on both sides. Moreover, "Relu" is compulationally inexpensive when compared to "sigmoid" and "tanh". Hence, a model with
"Relu" activation can loop through the weight update epochs faster than sigmoid or tanh. But will this popular "ReLu"
will also work in case of PINN?


<p align="center">
  <img width=500mm src="/assets/img/model_arch.png">
</p>

From the above architecture,

$$ \dfrac{\partial u_1}{\partial x_1} = \dfrac {\partial u_1}{\partial s_1}\dfrac {\partial s_1}{\partial x_1} + \dfrac {\partial u_1}{\partial s_2}\dfrac {\partial s_2}{\partial x_1}$$


$$ \dfrac{\partial u_1}{\partial x_1} = \dfrac {\partial u_1}{\partial s_1}\dfrac {\partial s_1}{\partial a_1}\dfrac {\partial a_1}{\partial x_1} + \dfrac {\partial u_1}{\partial s_2}\dfrac {\partial s_2}{\partial a_2}\dfrac{\partial a_2}{\partial x_1}$$

$$ \dfrac{\partial u_1}{\partial x_1} = m_{11}f'(a_1)w_{11} + m_{21}f'(a_2)w_{12}$$

Similarly,

$$ \dfrac{\partial u_2}{\partial x_2} = m_{12}f'(a_1)w_{21} + m_{22}f'(a_2)w_{22}$$

$$ \dfrac{\partial u_1}{\partial x_2} = m_{11}f'(a_1)w_{12} + m_{12}f'(a_2)w_{22}$$

$$ \dfrac{\partial u_2}{\partial x_1} = m_{12}f'(a_1)w_{11} + m_{22}f'(a_2)w_{12}$$


Loss:

$$ \mathcal{L} = \sum_{i}^N\dfrac{1}{2(x_i+1)}\left(\left.\dfrac{\partial u}{\partial x}\right\rvert_{x_{i}}\right)^2 - T(1)u(1)$$

For 1D case:


$$ \mathcal{L} = \sum_{i}^N\dfrac{1}{2(x_i+1)}\left(mf'(wx_i)w \right)^2 - T(1)mf(w)$$

Now in order to update the weights $$m$$ and $$w$$, the following derivatives are requried:

$$ \dfrac{\partial \mathcal{L}}{\partial w} = \sum_{i}^N\dfrac{2m^2f'(wx_i)w(f'(wx_i) + wx_if''(wx_i))}{2(x_i+1)} - T(1)mf'(w) $$

$$ \dfrac{\partial \mathcal{L}}{\partial m} = \sum_{i}^N\dfrac{2m(f'(wx_i)w)^2}{2(x_i+1)} - T(1)f(w) $$


Let us assume $$f(x)$$ to be Relu

$$ \dfrac{\partial \mathcal{L}}{\partial w} = 2wm^2\sum_{i}^N\dfrac{(f'(wx_i))^2}{2(x_i+1)} - T(1)m $$

$$ \dfrac{\partial \mathcal{L}}{\partial m} = 2mw^2\sum_{i}^N\dfrac{(f'(wx_i))^2}{2(x_i+1)} - T(1)f(w) $$

if $$w<0$$

$$ \dfrac{\partial \mathcal{L}}{\partial w} = T(1)m $$

$$ \dfrac{\partial \mathcal{L}}{\partial m} = 0 $$

if $$w>0$$

$$ \dfrac{\partial \mathcal{L}}{\partial w} = m\left(2wm\sum_{i}^N\dfrac{1}{2(x_i+1)} - T(1)\right) $$

$$ \dfrac{\partial \mathcal{L}}{\partial m} = w\left(2wm\sum_{i}^N\dfrac{1}{2(x_i+1)} - T(1)\right) $$
