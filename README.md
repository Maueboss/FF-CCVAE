# Forward-Forward Conditional Convolutional Variational AutoEncoder

My implementation revolves around the idea of generating images with an architecture trained using a forward-forward algorithm, rather than backpropagation. The idea is quite straightforward, use a modified version of the VAE, known as Conditional VAE using CNNs. The conditioning is done for the labels, this allows not only reconstruction of images and generation from the latent space, as it usually happens for the standard VAE. It also allows for conditional generation for a given class from the latent space.

The transition from BP to FFA is achieved thanks to the introduction of a modified version of the ELBO loss that gets computed locally for each decoder layer with respect its encoder simmetric layer:

![ELBO equation](https://latex.codecogs.com/svg.image?\dpi{120}\mathcal{L}_{t}'(\boldsymbol{\theta},%20\boldsymbol{\phi};%20\mathbf{x}'^{(i)}_{t})%20\simeq%20\frac{1}{2}%20\sum_{j=1}^{J}%20\left(%201%20+%20\log((\sigma_j^{(i)})^2)%20-%20(\mu_j^{(i)})^2%20-%20(\sigma_j^{(i)})^2%20\right)%20+%20\frac{1}{L}%20\sum_{l=1}^{L}%20\log%20p_{\boldsymbol{\theta}}(\mathbf{x}'^{(i)}_{t}%20|%20\mathbf{z}^{(i,l)}))

while CwC loss is computed layer by layer for the encoder:

![Loss Equation](https://latex.codecogs.com/svg.image?\dpi{120}\mathcal{L}_{t}=L_{CwC}%20=%20-\frac{1}{N}%20\sum_{n=1}^{N}%20\log\left(\frac{\exp(g_n^+)}{\sum_{j=1}^{J}%20\exp(G_{n,j})}\right))

The reason why this work is important is basically it being the first architecture trained using FFA for generating images using a VAE. 

<img width="280" alt="Picture 1" src="https://github.com/user-attachments/assets/dd4210af-dd48-4ca7-9db4-a937dee44dec" />

## How to Use

- Install required dependencies from requirements.txt
- update config.py with the required parameters for choosing a dataset
- From the official_python_implementation folder, run the following command:
```bash
python main.py
```
