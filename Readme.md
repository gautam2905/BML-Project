# BML Project  

In this project, I implement and compare Bayesian Linear Layers (BLL) and their variant with derivative-based regularization (LDBLL) as described in the paper "Latent Derivative Bayesian Last Layer Networks". The goal is to evaluate how incorporating derivative information can improve model performance and uncertainty estimation.   
I alse explored "VARIATIONAL BAYESIAN LAST LAYERS", Instead of calculating the exact posterior, VBLL uses Variational Inference (VI). It approximates the true posterior $p(w|D)$ with a variational distribution $q(w|\eta)$ (parameterized by $\eta$). The goal is to maximize the Evidence Lower Bound (ELBO) on the log marginal likelihood 

Then i observed the results and compared the performance of LDBLL and VBLL with standard BLL. I observed 