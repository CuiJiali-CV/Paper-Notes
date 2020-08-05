<h1 align="center">Beyond The FRONTIER</h1>

<p align="center">
    <a href="https://www.tensorflow.org/">
        <img src="https://img.shields.io/badge/Tensorflow-1.13-green" alt="Vue2.0">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Author-JialiCui-blueviolet" alt="Author">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Email-cuijiali961224@gmail.com-blueviolet" alt="Author">
    </a>
    <a href="https://www.stevens.edu/">
        <img src="https://img.shields.io/badge/College-SIT-green" alt="Vue2.0">
    </a>
</p>




# Learning Latent Space Energy-Based Prior Model

[Paper Here](https://arxiv.org/pdf/2006.08205.pdf)
<br /><br />

## Back Ground

* #####  Bayes philosophy

  - We can learn a generator model that maps a low-dimensional latent vector to a high-dimensional data space via a top-down network. The latent vector follows a simple prior distribution, such as uniform or Gaussian white noise distribution. However, we can also learn an expressive prior model instead of assuming a given prior distribution. 
  - Learning an energy-based prior model from the observed data follows the philosophy of empirical Bayes. 

* ##### Simpler than other Generator models

  * In recent years, generator models are in the contexts of VAEs and GANs. In both frameworks, the generator models is jointly learned with a complementary model, such as the inference model in VAE and the discriminator model in GAN.
  * More recently, the inference process is carried out by MCMC, which makes the learning is simpler in the sense that complementary network is not included.
  * Compared to GAN, MLE learning does not suffer from the issues like instability or mode collapsing.
  * Compared to VAE, for the generator model with latent space EBM prior, VAE has the issue in intractability of the normalizing constant of the latent space EBM.

* ##### Ineffectiveness

  * The assumption that latent vector follows a given simple prior distribution, such as isotropic Gaussian white noise distribution or uniform distribution, may cause ineffective generator learning.
  * Although we can increase the complexity of the top-down network to enhance the expressive power of the model, learning a prior model could be an alternative method.

* ##### High-dimension Inference

  * Unlike inferencing the vector from high-dimensional training data, such as images or videos, MCMC sampling in the low-dimensional latent space  is easily affordable on modern computing platforms. 

  

<br /><br />

## Solutions

#### Assume the latent vector follows an energy-based model, such a prior model adds more expressive power to the generator model.



## Process

- ##### Overall

  - The MLE learning of the generator model with a latent space EBM prior involves MCMC sampling of latent vector from both the prior and posterior distributions. 
  - Parameters of the prior model can then be updated based on the statistical difference samples from the two distributions.
  - Parameters of the top-down network can be updated based on the samples from the posterior distribution as well as the training data.
  - Always initialize MCMC from the fixed Gaussian white noise distribution(Cold Start), and always run a fixed and small number of steps, in both training and testing processes.

- ##### Details

  - Let <img src="https://latex.codecogs.com/gif.latex?x&space;\in&space;R^{D}" title="z_0 \in R^{K_0}" /> be an observed example such as an image, and let <img src="https://latex.codecogs.com/gif.latex?z&space;\in&space;R^{d}" title="z_0 \in R^{K_0}" /> be the latent variables. So that the desired joint distribution of <img src="https://latex.codecogs.com/gif.latex?(x,z)" title="z_0 \in R^{K_0}" /> is

    <div align=center><img src="https://latex.codecogs.com/gif.latex?p_\theta(x,&space;z)=p_\alpha(z)p_\beta(x\mid&space;z)" title="p_\theta(x, z)=p_\alpha(z)p_\beta(x\mid z)" /></div> 

    where <img src="https://latex.codecogs.com/gif.latex?p_\alpha(z)" title="z_0 \in R^{K_0}" /> is the prior model with parameters <img src="https://latex.codecogs.com/gif.latex?\alpha" title="z_0 \in R^{K_0}" />, <img src="https://latex.codecogs.com/gif.latex?p_\beta(x \mid z)" title="z_0 \in R^{K_0}" />is the top-down generation model with parameters <img src="https://latex.codecogs.com/gif.latex?\beta" title="z_0 \in R^{K_0}" />, and <img src="https://latex.codecogs.com/gif.latex?\theta = (\alpha, \beta)" title="z_0 \in R^{K_0}" />.

    - The prior model <img src="https://latex.codecogs.com/gif.latex?p_\alpha(z)" title="z_0 \in R^{K_0}" /> is formulated as an energy-based model,

      <div align=center><img src="https://latex.codecogs.com/gif.latex?p_\alpha(z)=\frac{{1}}{Z(\alpha)}exp(f_\alpha(z))p_0(z)" title="p_\alpha(z)=\frac{{1}}{Z(\alpha)}exp(f_\alpha(z))p_0(z)" /></div>

    where <img src="https://latex.codecogs.com/gif.latex?p_0(z)" title="z_0 \in R^{K_0}" /> is a reference distribution, assumed to be isotropic Gaussian. <img src="https://latex.codecogs.com/gif.latex?f_\alpha(z)" title="z_0 \in R^{K_0}" /> is the negative energy and is parameterized by a small multi-layer perceptron with parameters <img src="https://latex.codecogs.com/gif.latex?\alpha" title="z_0 \in R^{K_0}" />. <img src="https://latex.codecogs.com/gif.latex?Z(\alpha)=\int&space;exp(f_\alpha(z))p_0(z)dz=E_{p_0}[exp(f_\alpha(z))]" title="Z(\alpha)=\int exp(f_\alpha(z))p_0(z)dz=E_{p_0}[exp(f_\alpha(z))]" /> is the normalizing constant or partition function.  

    - The prior model can be interpreted as an energy-based correction or exponential tilting of the original prior distribution <img src="https://latex.codecogs.com/gif.latex?p_0" title="z_0 \in R^{K_0}" />, which is the prior distribution in the generator model in VAE.

      <br />

    - The generation model is a top-down network,

      <div align=center><img src="https://latex.codecogs.com/gif.latex?x=g_\beta(z)&plus;\epsilon" title="x=g_\beta(z)+\epsilon" /></div> 

      where <img src="https://latex.codecogs.com/gif.latex?\epsilon&space;\sim&space;N(0,\sigma&space;^2I_D)" title="\epsilon \sim N(0,\sigma ^2I_D)" />, so that <img src="https://latex.codecogs.com/gif.latex?p_\beta(x&space;\mid&space;z)&space;\sim&space;N(g_\beta(z),&space;\sigma^2I_D)" title="p_\beta(x \mid z) \sim N(g_\beta(z), \sigma^2I_D)" />.

  - In the original generator, the top-down network maps the unimodal prior distribution <img src="https://latex.codecogs.com/gif.latex?p_0" title="z_0 \in R^{K_0}" /> to be close to the usually highly multi-modal data distribution. However, the prior model refines <img src="https://latex.codecogs.com/gif.latex?p_0" title="z_0 \in R^{K_0}" /> so that <img src="https://latex.codecogs.com/gif.latex?g_\beta" title="z_0 \in R^{K_0}" /> maps the prior model <img src="https://latex.codecogs.com/gif.latex?p_\alpha" title="z_0 \in R^{K_0}" /> to be close to the data distribution.

  - The prior model <img src="https://latex.codecogs.com/gif.latex?p_\alpha" title="z_0 \in R^{K_0}" /> does not need to be highly multi-modal because of the expressiveness of <img src="https://latex.codecogs.com/gif.latex?g_\beta" title="z_0 \in R^{K_0}" />

  - ##### Maximum likelihood

    - The log-likelihood function is 

    - <div align=center> <img src="https://latex.codecogs.com/gif.latex?L(\theta)&space;=\sum_{i=1}^{n}logp_\theta(x_i)" title="L(\theta) =\sum_{i=1}^{n}logp_\theta(x_i)" /></div> 

      where <img src="https://latex.codecogs.com/gif.latex?p_\theta(x)=\int&space;p_\alpha(z)p_\beta(x&space;\mid&space;z)dz" title="p_\theta(x)=\int p_\alpha(z)p_\beta(x \mid z)dz" />, so that the learning gradient can be calculated to 

      <div align=center><img src="https://latex.codecogs.com/gif.latex?\small&space;\bigtriangledown_\theta&space;logp_\theta(x)=E_{p_\theta(z&space;\mid&space;x)}[\bigtriangledown_\theta&space;logp_\theta(x,z)]=E_{p_\theta(z&space;\mid&space;x)}[\bigtriangledown_\theta(logp_\alpha(z)&plus;logp_\beta(x&space;\mid&space;z))]" title="\small \bigtriangledown_\theta logp_\theta(x)=E_{p_\theta(z \mid x)}[\bigtriangledown_\theta logp_\theta(x,z)]=E_{p_\theta(z \mid x)}[\bigtriangledown_\theta(logp_\alpha(z)+logp_\beta(x \mid z))]" /></div> 

      <br />

    - For the prior model, <img src="https://latex.codecogs.com/gif.latex?\bigtriangledown_\alpha&space;logp_\alpha(z)=\bigtriangledown_\alpha&space;f_\alpha(z)&space;-&space;E_{p_\alpha(z)}[\bigtriangledown_\alpha&space;f_\alpha(z)]" title="\bigtriangledown_\alpha logp_\alpha(z)=\bigtriangledown_\alpha f_\alpha(z) - E_{p_\alpha(z)}[\bigtriangledown_\alpha f_\alpha(z)]" />. Thus the learning gradient for an example x is

    - <div align=center><img src="https://latex.codecogs.com/gif.latex?\delta&space;_\alpha(x)&space;=&space;\bigtriangledown_\alpha&space;logp_\theta(x)=E_{p_\theta(z&space;\mid&space;x)}[\bigtriangledown_\alpha&space;f_\alpha(z)]-E_{p_\alpha(z)}[\bigtriangledown_\alpha&space;f_\alpha(z)]" title="\delta _\alpha(x) = \bigtriangledown_\alpha logp_\theta(x)=E_{p_\theta(z \mid x)}[\bigtriangledown_\alpha f_\alpha(z)]-E_{p_\alpha(z)}[\bigtriangledown_\alpha f_\alpha(z)]" /></div>

      <br />

    - so that <img src="https://latex.codecogs.com/gif.latex?\alpha" title="z_0 \in R^{K_0}" /> is updated based on the difference between <img src="https://latex.codecogs.com/gif.latex?z" title="z_0 \in R^{K_0}" /> inferred from empirical observation x, and <img src="https://latex.codecogs.com/gif.latex?z" title="z_0 \in R^{K_0}" /> sampled from the current prior.

    - For  the generation model, the learning gradient can be calculated to

      <div align=center><img src="https://latex.codecogs.com/gif.latex?\delta&space;_\beta(x)&space;=&space;\bigtriangledown_\beta&space;logp_\theta(x)=E_{p_\theta(z&space;\mid&space;x)}[\bigtriangledown_\beta&space;logp_\beta(x&space;\mid&space;z)]" title="\delta _\beta(x) = \bigtriangledown_\beta logp_\theta(x)=E_{p_\theta(z \mid x)}[\bigtriangledown_\beta logp_\beta(x \mid z)]" /></div>

      where <img src="https://latex.codecogs.com/gif.latex?logp_\beta(x&space;\mid&space;z)=-\left&space;\|&space;x-g_\beta(z)&space;\right&space;\|^2/(2\sigma&space;^2)&plus;constant" title="logp_\beta(x \mid z)=-\left \| x-g_\beta(z) \right \|^2/(2\sigma ^2)+constant" />, which is actually the reconstruction error.

      <br />

    - For the expectations, we use MCMC sampling for the according model.





# Divergence Triangle for Joint Training of Generator Model, Energy-based Model, and Inferential Model

[Paper Here](https://arxiv.org/pdf/1812.10907.pdf)
<br /><br />

## Back Ground

- #### EBM usually requires MCMC, generator model does not have an explicit likelihood, inference model also requires MCMC sampling to approximate the posterior  of the latent variables.

- ### Combining the EMB, the generator model and the inference model is an attractive goal.



## Solutions

- #### High level

  - #### the energy-based model is learned based on the samples supplied by the generator model.

  - ####  With the help of inference model, the generator model is trained by both the observed data and the energy-based model. 

  - #### The inference model is learned from both the real data fitted by the generator model as well as the synthesized data generated by the generator model.

- #### Details

  - 

 

​    
















## Author

```javascript
var iD = {
  name  : "Jiali Cui",
  
  bachelor: "Harbin Institue of Technology",
  master : "Stevens Institute of Technology",
  
  Interested: "CV, ML"
}
```
