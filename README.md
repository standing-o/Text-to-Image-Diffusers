# Text-to-Image Diffusers
- This repository contains a curated collection of Jupyter Notebooks focused on **text-to-image generation** based on the [🤗 Hugging Face Diffusers](https://github.com/huggingface/diffusers) tutorial. 
- The notebooks are organized and adapted from official tutorials and community resources to help me quickly understand and experiment with SOTA models.



## Table of Contents
0. [DiffusionPipeline]()
1. [Tutorial]()
2. [Load Pipeline]()
3. [Generative Tasks]()
4. [Inference Techniques]()
5. [Outpainting]()
6. [T2I Pipelines]()
7. [Training]()
8. [Quantization Methods]()
9. [Accelerate Inference and Reduce Memory]()

- Optional
  - [I2I Pipelines]()    
  - [Inpainting Pipelines]()



## Setting
```commandline
conda create -n diffusers-t2i python=3.9 ipykernel
conda activate diffusers-t2i

pip install diffusers["torch"] accelerate transformers peft opencv-python
pip install -r requirements.txt
pip install -q optimum["onnxruntime"] optimum["openvino"]
pip install --upgrade-strategy eager
```