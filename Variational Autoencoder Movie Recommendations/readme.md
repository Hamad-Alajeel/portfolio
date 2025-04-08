# Introduction to Variational Autoencoders:
---
  Variational Autoencoders are models which learn latent space representations in lower dimensions of data samples in higher dimensional spaces, by implementing two steps in the trianing process: Encoding and Decoding. The ultimate goal of a Variational Autoencoder is to learn the joint distribution of a given dataset $p(x,z)$ in order to be able to reconstruct input data samples. However, in order to learn this joint distrubution, knowledge of the posterior $p(z|x)$ is required. In most cases this is intractable to calculate which is a reason for why one wouldn't be able to apply an algorithm like Expectation Maximization to learn this joint distribution. Therefore, the variational autoencoder takes this into account by maximizing a lower bound on the log likelihood of the conditional probability which uses a function that approximates the posterior. This function is learnt at the encoder stage of the variational autoencoder, and is parameterized by $\phi$: $q_{\phi}(z|x)$. The variational autoencoder samples values for z after the decoder stage, and uses that to compute the conditional probability $p_{\theta}(x|z)$ by feedforwarding through the decoder stage. The output of this model represents an attempt at reconstructing the input data given learnt conditional and posterior distributions for this dataset. The lower bound of the log likelihood of the conditional distribution is as so:

$` \log \; p\bigl(x_u; \theta\bigr)
\;\ge\;
\mathbb{E}_{q_{\phi}(z_u \mid x_u)}
\Bigl[\log\;p_{\theta}\bigl(x_u \mid z_u\bigr)\Bigr] -
\mathrm{KL}\!\Bigl(q_{\phi}(z_u \mid x_u) \,\Bigl\|\, p(z_u)\Bigr) \;\equiv\;
\mathcal{L}\bigl(x_u;\theta,\phi\bigr) `$

  In this loss function, the KL divergence term can be interpreted to be a regularization term which ensures that the posterior does not deviate from the prior by very much, thus preventing the model from overfitting to the data. This preserves the model's generative capabilities. However, for different tasks, different levels of generativity and fitting to datasets are required. Therefore, many researchers have introduced the $\beta$ hyperparameter to control how strongly the KL divergence term regularizes the posterior:

  $`
\mathcal{L}_{\beta}(x_u; \theta, \phi)
\;\equiv\;
\mathbb{E}_{q_{\phi}(z_u \mid x_u)}
\bigl[\log p_{\theta}(x_u \mid z_u)\bigr] - \beta \cdot 
\mathrm{KL}\bigl(q_{\phi}(z_u \mid x_u)\,\|\,p(z_u)\bigr).
  `$






