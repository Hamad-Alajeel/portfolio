# Introduction to Variational Autoencoders:
---
Variational Autoencoders are models which learn latent space representations in lower dimensions of data samples in higher dimensional spaces, by implementing two steps in the trianing process: Encoding and Decoding. The ultimate goal of a Variational Autoencoder is to learn the joint distribution of a given dataset $p(x,z)$ in order to be able to reconstruct input data samples from a given dataset. However, in order to learn this joint distrubution, knowledge of the posterior $p(z|x)$ is required


