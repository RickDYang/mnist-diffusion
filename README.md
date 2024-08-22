# MNIST Diffusion
This repositry is for AI beginers to learn Diffusion model (actually Denoising Diffusion Probabilistic model) from code.

Currently, the AI models are becoming more and more complicated, and cost more and more computing resourse. It is getting harder for AI beginer to learn. I would like to implement the foundation models from minimal requirements.
My GPU is **NVIDIA GeForce RTX 2070**, which has 8Gb GPU Memory. I try to implement the models based on this minimal GPU requirements.

# Results
Here are the results generated of this model.

![0](./test/mnist_diffusion_sample_0.png)
![1](./test/mnist_diffusion_sample_1.png)
![2](./test/mnist_diffusion_sample_2.png)
![3](./test/mnist_diffusion_sample_3.png)
![4](./test/mnist_diffusion_sample_4.png)
![5](./test/mnist_diffusion_sample_5.png)
![6](./test/mnist_diffusion_sample_6.png)
![7](./test/mnist_diffusion_sample_7.png)
![8](./test/mnist_diffusion_sample_8.png)
![9](./test/mnist_diffusion_sample_9.png)


# References
Firstly, I would like to recommend to read the following articles to understand what Diffusion model is.

- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
- [What are Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [What is Stable Diffusion?](https://poloclub.github.io/diffusion-explainer/)

# Setup
- [Setup torch with CUDA](https://pytorch.org/get-started/locally/)
- Clone the repo
- Setup requirements.txt
```shell
pip install -r requirements.txt
```

# How to run
The code is running in 4 modes: train, infer, upload and from_pretrain. Use the following command to run:
```shell
python main.py [train, infer, upload, from_pretrain]
```
## Environment Variables
Before running upload/from_pretrain, you need to set the following environment variables for huggingface access.
```
HUGGINGFACE_REPO=username/reponame
HUGGINGFACE_TOKEN=write_token_to_upload
```
## Modes
- train

    During train, the model will be saved to "mnist_diffusion_model.pt" for the best model
- infer

    In the inferencing, the model will be loaded from "mnist_diffusion_model.pt"
- upload

    You can use upload to upload the model to huggingface hub
- from_pretrain

    You can use from_pretrain to download the model from huggingface hub and make inference

    My trained models is in [Here](https://huggingface.co/RickDYang/ai-mini/blob/main/mnist_diffusion)