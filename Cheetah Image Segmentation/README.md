# Project Details:
  In this project I semantically segment an image of a cheetah into two classes: Background/Foreground by learning the underlying densities of the two classes using the expectation maximization Algorithm. The training dataset is a set of vectors that have discrete cosine transformation applied to them and their elements zig-zagged. I experimented on different amounts of latent variables for the densities of the classes when applying EM which yielded varying results. The optimal amount of components was found to be 8 for each class, and the classification with least error occurred when using 16 features only. 

  [Classification Results C8F16](https://github.com/user-attachments/assets/c032292f-876d-4ee0-af89-bceacc1ad96e)
*Figure 1: Classification results with 8 components and 16 features:*

Expectation maximization is an algorithm used to find the optimal parameters of a statistical model where the model contains hidden components to them.  
