# Diffusion Model Roadmap for Image Generation

A practical, implementation-oriented reading and development roadmap for building image generation models progressively: from a basic DDPM with a small U-Net to a modern latent Diffusion Transformer / Rectified Flow model.

## Target Outcome

By the end of this roadmap, the goal is to understand and implement a small-scale version of a modern text-to-image generative system inspired by Stable Diffusion 3 / FLUX-style architectures:

```text
Text Prompt
  ↓
Frozen Text Encoder: CLIP or T5
  ↓
Text Embeddings

Noise Latent
  ↓
Patchify
  ↓
Diffusion Transformer / Flow Transformer
  ↓
Predicted Noise or Velocity
  ↓
Sampler Loop
  ↓
Generated Latent
  ↓
VAE Decoder
  ↓
Final Image
```

The roadmap is split into modules. Each module has:

- What we will learn
- What we will build
- Suggested architecture
- Required papers
- Optional papers for deeper understanding

---

# Module 0 — Core Building Blocks

## Goal

Before implementing diffusion models, we need to understand the architectural components that appear repeatedly in modern image generation systems:

- U-Net
- Variational Autoencoder / Autoencoder
- Transformer
- Vision Transformer

This module is not about training a diffusion model yet. It is about understanding the major neural network components that will later become part of the full pipeline.

## What We Will Learn

- Encoder-decoder architectures
- Skip connections
- Latent variables
- Reparameterization trick
- Self-attention
- Multi-head attention
- Patch embeddings
- Positional embeddings
- How images can be represented as token sequences

## What We Will Build

Small standalone experiments:

1. A simple convolutional autoencoder for image reconstruction.
2. A small U-Net-like encoder-decoder.
3. A minimal Vision Transformer classifier on CIFAR-10 or a small image dataset.

## Suggested Architecture

### U-Net Skeleton

```text
Input Image
  ↓
Conv Block
  ↓
Downsample
  ↓
Conv Block
  ↓
Downsample
  ↓
Bottleneck
  ↓
Upsample + Skip Connection
  ↓
Conv Block
  ↓
Upsample + Skip Connection
  ↓
Output
```

### Autoencoder Skeleton

```text
Image
  ↓
Encoder
  ↓
Latent Representation
  ↓
Decoder
  ↓
Reconstructed Image
```

### Vision Transformer Skeleton

```text
Image
  ↓
Split into Patches
  ↓
Linear Patch Projection
  ↓
Transformer Blocks
  ↓
Prediction Head
```

## Required Papers

### 1. U-Net: Convolutional Networks for Biomedical Image Segmentation

- Authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
- Year: 2015
- Link: https://arxiv.org/abs/1505.04597

Focus on:

- Contracting path
- Expanding path
- Skip connections
- Multi-resolution feature maps

Why it matters:

U-Net became the standard backbone for many early diffusion models because it combines local convolutional processing with multi-scale feature aggregation.

---

### 2. Auto-Encoding Variational Bayes

- Authors: Diederik P. Kingma, Max Welling
- Year: 2013
- Link: https://arxiv.org/abs/1312.6114

Focus on:

- Latent variables
- Encoder and decoder
- Mean and variance prediction
- Reparameterization trick
- KL divergence

Why it matters:

Latent Diffusion Models rely on an autoencoder/VAE-like system to compress images before diffusion training. This reduces computational cost dramatically.

---

### 3. Attention Is All You Need

- Authors: Ashish Vaswani et al.
- Year: 2017
- Link: https://arxiv.org/abs/1706.03762

Focus on:

- Self-attention
- Multi-head attention
- Positional encoding
- Residual connections
- LayerNorm
- Feed-forward blocks

Why it matters:

Modern image generation models increasingly replace U-Net backbones with Transformer-based backbones.

---

### 4. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

- Authors: Alexey Dosovitskiy et al.
- Year: 2020
- Link: https://arxiv.org/abs/2010.11929

Focus on:

- Patchify operation
- Linear patch projection
- Positional embeddings
- Image as a sequence of tokens
- Transformer blocks for visual data

Why it matters:

Diffusion Transformers apply the same core idea to latent image patches instead of classification patches.

## Module Deliverable

By the end of this module, we should have:

```text
A working mental model of U-Net, VAE, Transformer, and Vision Transformer.
A basic autoencoder implementation.
A small ViT experiment.
```

---

# Module 1 — Basic DDPM in Pixel Space

## Goal

Build the first working diffusion model. It will generate simple images directly in pixel space.

This is the most important implementation module because it teaches the core diffusion loop:

```text
Clean Image → Add Noise → Train Model to Predict Noise → Sample by Removing Noise
```

## What We Will Learn

- Forward diffusion process
- Reverse denoising process
- Gaussian noise schedules
- Timestep embeddings
- Epsilon prediction
- Mean squared error loss for noise prediction
- Sampling loop

## What We Will Build

A small DDPM trained on MNIST, Fashion-MNIST, or CIFAR-10.

Recommended implementation order:

1. Dataset loader
2. Image normalization to `[-1, 1]`
3. Noise scheduler
4. Timestep embedding
5. Small U-Net
6. Training loop
7. Sampling loop
8. Image grid generation
9. Checkpoint saving

## Suggested Architecture

### Pixel-space DDPM with Small U-Net

```text
Clean image x0
  ↓
Add Gaussian noise at timestep t
  ↓
Noisy image xt
  ↓
U-Net(xt, t)
  ↓
Predicted noise εθ
  ↓
MSE(εθ, real noise ε)
```

### Initial Configuration

For MNIST:

```text
Image size: 28x28
Channels: 1
Base channels: 32
Timesteps: 1000
Prediction target: epsilon/noise
Loss: MSE
```

For CIFAR-10:

```text
Image size: 32x32
Channels: 3
Base channels: 64
Timesteps: 1000
Prediction target: epsilon/noise
Loss: MSE
```

## Required Paper

### 1. Denoising Diffusion Probabilistic Models

- Authors: Jonathan Ho, Ajay Jain, Pieter Abbeel
- Year: 2020
- Link: https://arxiv.org/abs/2006.11239

Focus on:

- Forward process `q(x_t | x_0)`
- Reverse process `p_theta(x_{t-1} | x_t)`
- Noise schedule
- Epsilon prediction
- MSE training objective
- Sampling loop

Why it matters:

This is the foundational paper for the modern DDPM formulation.

## Optional Paper

### 2. Improved Denoising Diffusion Probabilistic Models

- Authors: Alex Nichol, Prafulla Dhariwal
- Year: 2021
- Link: https://arxiv.org/abs/2102.09672

Focus on:

- Learned variance
- Cosine noise schedule
- Faster sampling
- Scaling with model capacity and compute

Why it matters:

This paper introduces practical improvements that make DDPMs more efficient and more competitive.

## Module Deliverable

By the end of this module, we should have:

```text
A working DDPM that generates simple images from pure noise.
A trained U-Net denoiser.
A reusable noise scheduler and sampling loop.
```

---

# Module 2 — Faster Sampling and Better Diffusion Design

## Goal

Improve the sampling process and understand how diffusion models can be made faster and cleaner from an engineering point of view.

After Module 1, the model works, but sampling may require hundreds or thousands of denoising steps. In this module, we study faster sampling techniques.

## What We Will Learn

- DDPM vs DDIM
- Deterministic sampling
- Non-Markovian sampling
- Noise schedule choices
- Sigma parameterization
- ODE/SDE interpretation
- Euler and Heun samplers
- Preconditioning

## What We Will Build

1. A DDIM sampler for the Module 1 model.
2. A cosine noise schedule.
3. A configurable sampler interface.
4. Optional: Euler or Heun sampler experiments inspired by EDM.

## Suggested Architecture

We keep the same U-Net trained in Module 1, but change the sampling strategy:

```text
Trained U-Net DDPM
  ↓
Sampling Strategy A: DDPM
Sampling Strategy B: DDIM
Sampling Strategy C: Euler/Heun-style sampler
  ↓
Generated Image
```

## Required Paper

### 1. Denoising Diffusion Implicit Models

- Authors: Jiaming Song, Chenlin Meng, Stefano Ermon
- Year: 2020
- Link: https://arxiv.org/abs/2010.02502

Focus on:

- DDIM sampling
- Deterministic generation
- Fewer sampling steps
- Same training objective as DDPM
- Trade-off between compute and image quality

Why it matters:

DDIM allows faster sampling without retraining the DDPM model.

## Optional Paper

### 2. Elucidating the Design Space of Diffusion-Based Generative Models

- Authors: Tero Karras, Miika Aittala, Timo Aila, Samuli Laine
- Year: 2022
- Link: https://arxiv.org/abs/2206.00364

Focus on:

- Sigma parameterization
- Preconditioning
- Noise schedules
- ODE/SDE sampling
- Euler sampler
- Heun sampler
- Separating architecture, training objective, and sampling design

Why it matters:

This paper helps you stop treating diffusion as a single fixed algorithm and start seeing it as a design space.

## Module Deliverable

By the end of this module, we should have:

```text
The same trained model sampled with fewer steps.
A cleaner abstraction for schedulers and samplers.
A better understanding of why sampling design matters.
```

---

# Module 3 — Latent Diffusion

## Goal

Move from pixel-space diffusion to latent-space diffusion.

Instead of denoising full-resolution images, we first compress images into a latent space using an autoencoder or VAE, then train diffusion on those latents.

This is the key idea behind Stable Diffusion-style systems.

## What We Will Learn

- Image compression into latent space
- Autoencoder/VAE training
- Reconstruction quality
- Latent-space denoising
- Why latent diffusion is cheaper than pixel diffusion
- How decoder quality affects final image quality

## What We Will Build

1. A convolutional autoencoder or VAE.
2. A latent dataset generated by encoding images.
3. A latent DDPM using a U-Net denoiser.
4. A decoder pipeline to turn generated latents back into images.

## Suggested Architecture

```text
Training Autoencoder:

Image
  ↓
Encoder
  ↓
Latent z
  ↓
Decoder
  ↓
Reconstructed Image
```

```text
Training Latent Diffusion:

Image
  ↓
Frozen Encoder
  ↓
Latent z0
  ↓
Add noise
  ↓
Noisy latent zt
  ↓
U-Net Denoiser
  ↓
Predicted noise
```

```text
Sampling:

Random latent noise
  ↓
U-Net denoising loop
  ↓
Generated latent
  ↓
Frozen Decoder
  ↓
Generated image
```

## Initial Configuration

```text
Image size: 128x128 or 256x256
Latent size: 16x16 or 32x32
Latent channels: 4
Autoencoder loss: L1 or MSE first
Diffusion target: epsilon/noise prediction
Dataset: CelebA, Flowers, Pokemon, or a small curated image dataset
```

## Required Paper

### 1. High-Resolution Image Synthesis with Latent Diffusion Models

- Authors: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer
- Year: 2021/2022
- Link: https://arxiv.org/abs/2112.10752

Focus on:

- Perceptual image compression
- Latent diffusion
- Autoencoder bottleneck
- Cross-attention for conditioning
- Efficiency gains from operating in latent space

Why it matters:

This paper is the conceptual foundation for Stable Diffusion-like systems.

## Optional Supporting Paper

### 2. Auto-Encoding Variational Bayes

- Authors: Diederik P. Kingma, Max Welling
- Link: https://arxiv.org/abs/1312.6114

Revisit this paper if you decide to implement a true VAE instead of a simpler deterministic autoencoder.

## Module Deliverable

By the end of this module, we should have:

```text
A trained autoencoder/VAE.
A latent-space DDPM.
A pipeline that generates images by denoising latent noise and decoding it.
```

---

# Module 4 — Text Conditioning and Classifier-Free Guidance

## Goal

Turn the image generator into a text-to-image model.

The model should learn to generate images conditioned on prompts such as:

```text
"a small yellow creature with wings"
"a red flower on a green background"
"a blue cartoon monster"
```

## What We Will Learn

- Text tokenization
- Frozen text encoders
- CLIP text embeddings
- T5 text embeddings
- Cross-attention
- Prompt dropout
- Classifier-Free Guidance
- Conditional vs unconditional prediction

## What We Will Build

1. A frozen text encoder pipeline.
2. A paired image-caption dataset loader.
3. Prompt dropout during training.
4. Cross-attention conditioning inside the U-Net or Transformer.
5. Classifier-Free Guidance during inference.

## Suggested Architecture — U-Net Version

```text
Prompt
  ↓
CLIP Text Encoder
  ↓
Text Embeddings

Image
  ↓
VAE Encoder
  ↓
Latent z
  ↓
Add noise
  ↓
Noisy latent zt
  ↓
U-Net with Cross-Attention(text embeddings)
  ↓
Predicted noise
```

## Suggested Architecture — DiT-Compatible Version

```text
Prompt
  ↓
CLIP or T5 Text Encoder
  ↓
Text Tokens

Noisy Latent
  ↓
Patchify
  ↓
Image Tokens

Image Tokens + Text Conditioning + Timestep
  ↓
Transformer Denoiser
  ↓
Predicted noise or velocity
```

## Required Paper

### 1. Classifier-Free Diffusion Guidance

- Authors: Jonathan Ho, Tim Salimans
- Year: 2022
- Link: https://arxiv.org/abs/2207.12598

Focus on:

- Conditional model
- Unconditional model
- Prompt dropout
- Guidance scale
- Combining conditional and unconditional predictions

Core idea:

```text
prediction = prediction_uncond + guidance_scale * (prediction_cond - prediction_uncond)
```

Why it matters:

Classifier-Free Guidance is one of the most important techniques for making text-to-image models follow prompts more strongly.

## Additional Papers

### 2. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models

- Authors: Alex Nichol et al.
- Year: 2021
- Link: https://arxiv.org/abs/2112.10741

Focus on:

- Text-conditional diffusion
- CLIP guidance vs Classifier-Free Guidance
- Inpainting
- Text-guided editing

Why it matters:

GLIDE is an important bridge between basic diffusion and modern text-guided diffusion systems.

---

### 3. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding

- Authors: Chitwan Saharia et al.
- Year: 2022
- Link: https://arxiv.org/abs/2205.11487

Focus on:

- Imagen
- T5 as text encoder
- Deep language understanding
- Cascaded diffusion
- Super-resolution diffusion
- DrawBench

Why it matters:

This paper shows that the quality of the text encoder is extremely important for prompt alignment.

---

### 4. Hierarchical Text-Conditional Image Generation with CLIP Latents

- Authors: Aditya Ramesh et al.
- Year: 2022
- Link: https://arxiv.org/abs/2204.06125

Focus on:

- DALL-E 2 / unCLIP
- CLIP image embeddings
- Prior model
- Diffusion decoder
- Image variations
- Semantic and style preservation

Why it matters:

This paper presents an alternative way to connect text and image generation through CLIP latent representations.

## Recommended Dataset

Start small:

```text
Pokemon dataset with captions
COCO captions subset
Oxford Flowers with captions
A small curated dataset with synthetic captions from BLIP or another captioning model
```

## Module Deliverable

By the end of this module, we should have:

```text
A small text-to-image model.
A working prompt-conditioning pipeline.
Classifier-Free Guidance implemented in the sampler.
```

---

# Module 5 — Diffusion Transformer / DiT

## Goal

Replace the U-Net denoiser with a Transformer-based denoiser.

This module moves the architecture closer to modern image generators. Instead of applying convolutions over spatial feature maps, we split the latent representation into patches and process them as tokens.

## What We Will Learn

- Latent patchification
- Token sequence modeling for images
- Diffusion Transformer blocks
- Adaptive LayerNorm / AdaLN
- Timestep conditioning in Transformer blocks
- Attention cost and token count
- Scaling through depth, width, and number of tokens

## What We Will Build

1. Patchify and unpatchify utilities.
2. A small DiT denoiser.
3. A latent diffusion model using DiT instead of U-Net.
4. Optional: text-conditioned DiT with cross-attention.

## Suggested Architecture

```text
Image
  ↓
Frozen VAE Encoder
  ↓
Latent z
  ↓
Add noise
  ↓
Noisy latent zt
  ↓
Patchify
  ↓
Latent Tokens
  ↓
DiT Blocks conditioned on timestep
  ↓
Output Tokens
  ↓
Unpatchify
  ↓
Predicted noise or velocity
```

## Initial Configuration

```text
Image size: 128x128 or 256x256
Latent size: 16x16 or 32x32
Latent channels: 4
Patch size: 2 or 4
Hidden dim: 384 or 512
Layers: 6 to 12
Heads: 6 or 8
MLP ratio: 4
Conditioning: timestep first, text later
```

## Required Paper

### 1. Scalable Diffusion Models with Transformers

- Authors: William Peebles, Saining Xie
- Year: 2022
- Link: https://arxiv.org/abs/2212.09748

Focus on:

- Replacing U-Net with Transformer
- Latent patches
- DiT block
- Adaptive LayerNorm
- Scaling with Gflops
- Patch size
- Depth, width, and token count

Why it matters:

This is the central paper for understanding Diffusion Transformers.

## Module Deliverable

By the end of this module, we should have:

```text
A working latent DiT model.
Patchify/unpatchify utilities.
A clear comparison between U-Net-based latent diffusion and Transformer-based latent diffusion.
```

---

# Module 6 — Flow Matching and Rectified Flow

## Goal

Move beyond the classical DDPM objective and study the modern flow-based training objective used in newer systems such as Stable Diffusion 3-style models.

Instead of training the model only to predict added noise, we train it to predict a velocity field that moves samples from noise to data.

## What We Will Learn

- Continuous Normalizing Flows
- Vector fields
- Probability paths
- Velocity prediction
- ODE sampling
- Euler sampling
- Heun sampling
- Rectified paths from noise to data

## What We Will Build

1. A simple velocity-prediction training objective.
2. A Flow Matching or Rectified Flow training loop.
3. An Euler sampler.
4. Optional: Heun sampler.
5. A DiT + Flow Matching prototype in latent space.

## Suggested Architecture

```text
Real image
  ↓
Frozen VAE Encoder
  ↓
Real latent z1

Gaussian noise
  ↓
Noise latent z0

Interpolate:
zt = (1 - t) * z0 + t * z1

Model:
DiT(zt, t, text_conditioning)
  ↓
Predicted velocity

Target:
v = z1 - z0

Loss:
MSE(predicted_velocity, target_velocity)
```

## Required Papers

### 1. Flow Matching for Generative Modeling

- Authors: Yaron Lipman et al.
- Year: 2022
- Link: https://arxiv.org/abs/2210.02747

Focus on:

- Continuous Normalizing Flows
- Simulation-free training
- Vector field regression
- Probability paths
- Optimal Transport paths

Why it matters:

Flow Matching provides the theoretical basis for training generative models through vector fields instead of only classical diffusion denoising.

---

### 2. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

- Authors: Xingchao Liu, Chengyue Gong, Qiang Liu
- Year: 2022
- Link: https://arxiv.org/abs/2209.03003

Focus on:

- Straight paths
- ODE generative models
- Transport from noise to data
- Euler sampling
- Velocity targets

Why it matters:

Rectified Flow is conceptually simple and closely related to the direction taken by modern Transformer-based image generation models.

## Module Deliverable

By the end of this module, we should have:

```text
A latent DiT model trained with a velocity objective.
A basic Euler sampler.
A working comparison between epsilon prediction and velocity/flow prediction.
```

---

# Module 7 — Modern Text-to-Image Transformer Systems

## Goal

Study how recent high-end image generation systems combine:

- Latent autoencoders
- Text encoders
- Diffusion Transformers
- Rectified Flow
- Multimodal token interaction

This module is the conceptual target of the roadmap.

## What We Will Learn

- MMDiT-style architecture
- Separate modality weights for image and text tokens
- Bidirectional interaction between text and image tokens
- Rectified Flow at scale
- Noise sampling strategies for high-resolution image synthesis
- Scaling behavior

## What We Will Build

A small “Mini-SD3-like” or “Mini-FLUX-like” architecture:

```text
Frozen Text Encoder
  ↓
Text Tokens

Frozen VAE Encoder/Decoder
  ↓
Latent Image Tokens

Image Tokens + Text Tokens
  ↓
Multimodal Diffusion Transformer / Flow Transformer
  ↓
Velocity Prediction
  ↓
Euler/Heun Sampler
  ↓
Generated Latent
  ↓
VAE Decoder
  ↓
Generated Image
```

## Suggested Architecture

```text
Text Encoder:
  CLIP or T5 frozen

Image Autoencoder:
  Frozen VAE or trained autoencoder

Backbone:
  DiT or simplified MMDiT

Conditioning:
  Timestep embedding
  Text-image cross-attention or joint token mixing

Training Objective:
  Velocity prediction / Rectified Flow

Sampler:
  Euler first
  Heun later
```

## Required Paper

### 1. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

- Authors: Patrick Esser et al.
- Year: 2024
- Link: https://arxiv.org/abs/2403.03206

Focus on:

- Rectified Flow for text-to-image synthesis
- MMDiT architecture
- Separate weights for image and text modalities
- Bidirectional text-image interaction
- Scaling trends
- Typography and prompt comprehension improvements

Why it matters:

This is the Stable Diffusion 3 paper and one of the best references for understanding the modern shift from U-Net diffusion models to Transformer/Flow-based systems.

## Module Deliverable

By the end of this module, we should have a technical design for:

```text
A small latent text-to-image Rectified Flow Transformer.
```

This does not need to match production-level quality. The purpose is to understand the modern architecture end-to-end.

---

# Practical Implementation Order

This is the recommended coding sequence:

```text
01. Dataset loader
02. Image normalization
03. Noise scheduler
04. Timestep embedding
05. Small U-Net
06. DDPM training loop
07. DDPM sampling loop
08. Checkpoint saving
09. Image grid generation
10. DDIM sampler
11. Autoencoder/VAE
12. Autoencoder training loop
13. Freeze autoencoder
14. Latent diffusion with U-Net
15. Patchify/unpatchify utilities
16. Small DiT
17. Latent diffusion with DiT
18. Frozen text encoder
19. Caption dataset loader
20. Cross-attention or text conditioning
21. Classifier-Free Guidance
22. Velocity prediction
23. Euler sampler
24. Flow Matching / Rectified Flow training
25. Mini-SD3-like architecture design
```

---

# Suggested Reading Order

Read the papers in this order:

```text
1. U-Net: Convolutional Networks for Biomedical Image Segmentation
2. Auto-Encoding Variational Bayes
3. Attention Is All You Need
4. An Image is Worth 16x16 Words
5. Denoising Diffusion Probabilistic Models
6. Improved Denoising Diffusion Probabilistic Models
7. Denoising Diffusion Implicit Models
8. High-Resolution Image Synthesis with Latent Diffusion Models
9. Classifier-Free Diffusion Guidance
10. GLIDE
11. Imagen
12. DALL-E 2 / unCLIP
13. Scalable Diffusion Models with Transformers
14. Flow Matching for Generative Modeling
15. Flow Straight and Fast: Rectified Flow
16. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
```

---

# Minimal Essential Reading List

If we wanted to reduce the reading list to only the most essential papers, we would keep these:

1. **Denoising Diffusion Probabilistic Models**  
   https://arxiv.org/abs/2006.11239

2. **Improved Denoising Diffusion Probabilistic Models**  
   https://arxiv.org/abs/2102.09672

3. **Denoising Diffusion Implicit Models**  
   https://arxiv.org/abs/2010.02502

4. **High-Resolution Image Synthesis with Latent Diffusion Models**  
   https://arxiv.org/abs/2112.10752

5. **Classifier-Free Diffusion Guidance**  
   https://arxiv.org/abs/2207.12598

6. **Scalable Diffusion Models with Transformers**  
   https://arxiv.org/abs/2212.09748

7. **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis**  
   https://arxiv.org/abs/2403.03206

---

# Recommended Local vs Cloud Work

## Good to Run Locally on a Mac M4 Pro with 24 GB RAM

```text
MNIST DDPM
Fashion-MNIST DDPM
CIFAR-10 DDPM small version
Small autoencoder
Small latent diffusion at 64x64 or 128x128
Small DiT on tiny latent size
Sampler experiments
Debugging the full pipeline
```

## Better to Run on Colab or GPU Cloud

```text
Text-to-image training
256x256 latent diffusion
Larger DiT models
Flow Matching training
Large caption datasets
Long training runs
Any experiment requiring high VRAM
```

---

# Final Target Architecture

The final architecture we are aiming to understand and prototype is:

```text
Mini Latent Text-to-Image Rectified Flow Transformer
```

Full pipeline:

```text
Prompt
  ↓
Frozen Text Encoder: CLIP or T5
  ↓
Text Tokens

Random Gaussian Latent Noise
  ↓
Patchify
  ↓
Image Latent Tokens

Image Tokens + Text Tokens + Timestep
  ↓
Diffusion Transformer / Multimodal Transformer
  ↓
Predicted Velocity

Sampler Loop: Euler or Heun
  ↓
Generated Latent
  ↓
Frozen VAE Decoder
  ↓
Generated Image
```

This architecture gives us the conceptual bridge from a simple DDPM to modern models such as Stable Diffusion 3 and FLUX-style systems.
