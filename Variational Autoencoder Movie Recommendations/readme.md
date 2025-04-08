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
\;=\;
\mathbb{E}_{q_{\phi}(z_u \mid x_u)}
\bigl[\log\; p_{\theta}(x_u \mid z_u)\bigr] - \beta \cdot 
\mathrm{KL}\bigl(q_{\phi}(z_u \mid x_u)\,\|\,p(z_u)\bigr).
 `$

The traditional heuristic for controlling $\beta$ is to slowly anneal it to a value of 1 in order for the model to learn the latent state representations well before stabilizing the encoder stage's learning. In my paper, I attempt to improve a variational autencoder used for movie recommendations, the mult-vae, by experimenting with two novel methods for controlling $\beta$. The first one inspired by NLP tasks cyclically anneals $\beta$. 

<div align="center">
  <img src="https://github.com/Hamad-Alajeel/portfolio/blob/main/assets/beta%20annealing%20(1).png">
</div>

In NLP tasks, as $\beta$ is annealed to a value of 1, the VAE loses its ability to improve its learning of the latent space, and eventually ignores contextual information of previous tokens. Therefore, by cyclically annealing $\beta$,  the approximation of the posterior distribution is broken and it attempts to re-learn the hidden space using what it has learned in the previous cycle. In the task of movie recommendaitons though, contextual information is not used to generate novel movie recommendations, so experimenting with this approach was tentative. The more promising approach, however, was scaling the KL term in proportion to the amount of movies a user has interacted with. The expectation behind using this method was that anomalous users who did not have many movies they had positively interacted with would not have a disproportionate effect on the model's learning of the latent space:


$$\beta(x_u)=\gamma|x_u|=\gamma \sum_{i=1}^{N} x_{ui} $$

  In addition to experimenting with these different methods for controlling $\beta$, I performed an ablation study on the model to understand the effects of chaging different aspects of the architecture, the hyperparameters of the model, or the training process.

# Main Results:

The ablation study revealed that altering the model's architecture by increasing the amount of hidden layers, or the latent space size, or changing the activation function to relu, only lead to worse performance by the model. As expected, cyclical Annealing to a max level of $\beta=0.2$ did not lead to any improvement in model performance. However, the most promising results came from scaling the KL term in proportion to the amount of user interactions. This method resulted in an normalized discounted cumulative gain (NDCG) which was the best out of all other experiments, and almost equivalent to the original model with the traditional heurisitic. Moreover, its recall@50 value was higher than the original model, meaning it was more accurately able to reconstruct the original data sample, while also introducing novel recommendations. The results are provided below. First is a table of metrics obtained from the original model, and the succeeding two are plots and tables for both using the cyclical annealling method and scaling beta to user interaction rate. 


