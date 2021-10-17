---
title: "Estimation of local strain field in composite material using supervised training of UNet model"
exclude: true
---
**Abstract:**
The knowledge of the distribution of local micromechanical fields is crucial in
 the design of composite materials. Traditionally full-field methods
 (such as finite element methods) and fast Fourier transformation-based methods
 are used to obtain the local fields. However, full-field simulations are
    computationally expensive and time-consuming. Recently, there has been a push 
 toward using the big-data-driven machine learning approaches to estimate the
 local fields and establish the structure–property linkages. In this work, we
 use one of the deep learning-based algorithms known as the UNet to predict the 
 local strain fields in a two-phase composite material subjected to uniaxial
 tensile load. The model is trained and tested on 1200 two-phase microstructures
 comprising two-volume fraction categories and six different morphological
 classes. An R2 score of 94% is achieved on the test dataset.
 A detailed statistical analysis is performed to understand 
 the role of the volume fraction and the ratio of elastic moduli
 of the phases in the deep learning model’s trainability. 
 The insights drawn in this work are then discussed in the context
 of generating artificial datasets and training a robust predictive 
 deep learning model for localization.