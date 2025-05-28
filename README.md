# Text-to-Image Diffusers
- This repository contains a curated collection of Jupyter Notebooks focused on **text-to-image generation** based on the [🤗 Hugging Face Diffusers](https://github.com/huggingface/diffusers) tutorial. 
- The notebooks are organized and adapted from official tutorials and community resources to help me quickly understand and experiment with SOTA models.

&nbsp;
&nbsp;
&nbsp;
-----

## Table of Contents
0. [DiffusionPipeline](https://github.com/standing-o/Text-to-Image-Diffusers/blob/master/0.%20DiffusionPipeline.ipynb)
```
> Components of DiffusionPipeline
> Local Pipeline
> Swapping Schedulers
> Models
> Schedulers
> Efficient Diffusion: Speed
> Efficient Diffusion: Memory
> Efficient Diffusion: Quaility
```

1. [Tutorial](https://github.com/standing-o/Text-to-Image-Diffusers/blob/master/1.%20Tutorial.ipynb)
```
> Understanding Pipelines, Models and Schedulers
> AutoPipeline
> Train a Diffusion Model
> Load LoRAs for Inference
> Accelerate Inference of Text-to-image Diffusion Models
> Working with Big Models
```

2. [Load Pipeline](https://github.com/standing-o/Text-to-Image-Diffusers/blob/master/2.%20Load%20Pipeline.ipynb)
```
> Load Pipelines
> Load Community Pipelines and Components
> Load Schedulers and Models
> Model Files and Layout
> Load Adapter
```

3. [Generative Tasks](https://github.com/standing-o/Text-to-Image-Diffusers/blob/master/3.%20Generative%20Tasks.ipynb)
```
> Unconditional Image Generation
> Text-guided Depth-to-image Generation
```

4. [Inference Techniques](https://github.com/standing-o/Text-to-Image-Diffusers/blob/master/4.%20Inference%20Techniques.ipynb)
```
> Create a server
> Distributed Inference
> Merge LoRAs
> Scheduler Features
> Pipeline Callbacks
> Reproducible Pipelines
> Controlling Image Quality
> Prompt Techniques
```

5. [Outpainting](https://github.com/standing-o/Text-to-Image-Diffusers/blob/master/5.%20Outpainting.ipynb)
```
> Image Preparation
> Outpaint
```

6. [T2I Pipelines](https://github.com/standing-o/Text-to-Image-Diffusers/blob/master/6.%20T2I%20Pipelines.ipynb)
```
> Stable Diffusion XL
> Stable Diffusion XL Turbo
> Kandinsky
> IP-Adapter
> OmniGen
> Perturbed-Attention Guidance(PAG)
> ControlNet
> T2I-Adapter
> Latent Consistency Model
> Textual inversion
> DiffEdit
> Trajectory Consistency Distillation-LoRA
```

7. [Training](https://github.com/standing-o/Text-to-Image-Diffusers/blob/master/7.%20Training.ipynb)
```
> Create a Dataset for Training
> Adapt a Model to a New Task
> Models
  - Unconditional Image Generation
  - Text-to-Image
  - Stable Diffusion XL
  - Kandinsky 2.2, Wuerstchen, ControlNet, T2I-Adapter, InstructPix2Pix, etc.
> Methods
  - Textual Inversion
  - DreamBooth, LoRA, Custom Diffusion, Latent Consistency Distillation
```

8. [Quantization Methods](https://github.com/standing-o/Text-to-Image-Diffusers/blob/master/8.%20Quantization%20Methods.ipynb)
```
> bitsandbytes
> gguf
> torchao
> quanto
```

9. [Accelerate Inference and Reduce Memory](https://github.com/standing-o/Text-to-Image-Diffusers/blob/master/9.%20Accelerate%20Inference%20and%20Reduce%20Memory.ipynb)
```
> Accelerate inference
> Reduce memory usage
> Diffusers supports
  - PyTorch 2.0
  - xFormers
  - Token merging
  - DeepCache
  - TGATE
  - xDiT
  - ParaAttention
> Optimized Model Format
  - JAX/Flax
  - ONNX
  - OpenVINO
  - CoreML
```

- Optional
  - [I2I Pipelines](https://github.com/standing-o/Text-to-Image-Diffusers/blob/master/%5BOptional%5D%20I2I%20Pipelines.ipynb)    
  - [Inpainting Pipelines](https://github.com/standing-o/Text-to-Image-Diffusers/blob/master/%5BOptional%5D%20Inpainting%20Pipelines.ipynb)


&nbsp;
&nbsp;
&nbsp;
-----


## Setting
```commandline
conda create -n diffusers-t2i python=3.9 ipykernel
conda activate diffusers-t2i

pip install diffusers["torch"] accelerate transformers peft opencv-python
pip install -r requirements.txt
pip install -q optimum["onnxruntime"] optimum["openvino"]
pip install --upgrade-strategy eager
```
