### Gradio Interface for SpatialLM Model
SpatialLM: A 3D Large Language Model for Structured Scene Understanding, Processing Point Cloud Data from Monocular Videos, RGBD Images, and LiDAR.

<div align="center">
  
[![Watch the video](https://github.com/miladfa7/SpatialLM-Gradio/blob/master/figures/Thumbnail.png)](https://www.youtube.com/watch?v=j5SykSmAsfQ)

</div>

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="figures/logo_light.png#gh-light-mode-only" width="60%" alt="SpatialLM" />
  <img src="figures/logo_dark.png#gh-dark-mode-only" width="60%" alt="SpatialLM" />
</div>
<hr style="margin-top: 0; margin-bottom: 8px;">

<div align="center" style="margin-top: 0; padding-top: 0; line-height: 1;">
    <a href="https://manycore-research.github.io/SpatialLM" target="_blank" style="margin: 2px;"><img alt="Project"
    src="https://img.shields.io/badge/🌐%20Website-SpatialLM-ffc107?color=42a5f5&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://github.com/manycore-research/SpatialLM" target="_blank" style="margin: 2px;"><img alt="GitHub"
    src="https://img.shields.io/badge/GitHub-SpatialLM-24292e?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
</div>
<div align="center" style="line-height: 1;">
    <a href="https://huggingface.co/manycore-research/SpatialLM-Llama-1B" target="_blank" style="margin: 2px;"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SpatialLM%201B-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
    <a href="https://huggingface.co/datasets/manycore-research/SpatialLM-Testset" target="_blank" style="margin: 2px;"><img alt="Dataset"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-SpatialLM-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/></a>
</div>

## Introductiond

SpatialLM is a 3D large language model designed to process 3D point cloud data and generate structured 3D scene understanding outputs. These outputs include architectural elements like walls, doors, windows, and oriented object bounding boxes with their semantic categories. Unlike previous methods that require specialized equipment for data collection, SpatialLM can handle point clouds from diverse sources such as monocular video sequences, RGBD images, and LiDAR sensors. This multimodal architecture effectively bridges the gap between unstructured 3D geometric data and structured 3D representations, offering high-level semantic understanding. It enhances spatial reasoning capabilities for applications in embodied robotics, autonomous navigation, and other complex 3D scene analysis tasks.   [Project Page](https://manycore-research.github.io/SpatialLM) | [Official Code](https://github.com/manycore-research/SpatialLM)





## SpatialLM Models

<div align="center">

|      **Model**      | **Download**                                                                   |
| :-----------------: | ------------------------------------------------------------------------------ |
| SpatialLM-Llama-1B  | [🤗 HuggingFace](https://huggingface.co/manycore-research/SpatialLM-Llama-1B)  |
| SpatialLM-Qwen-0.5B | [🤗 HuggingFace](https://huggingface.co/manycore-research/SpatialLM-Qwen-0.5B) |

</div>

## Usage

### Installation

Tested with the following environment:

- Python 3.11
- Pytorch 2.4.1
- CUDA Version 12.4

```bash
# clone the repository
git clone https://github.com/manycore-research/SpatialLM-Gradio.git
cd SpatialLM-Gradio

# create a conda environment with cuda 12.4
conda create -n spatiallm-gradio python=3.11
conda activate spatiallm-gradio
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit conda-forge::sparsehash

# Install dependencies with poetry
pip install poetry && poetry config virtualenvs.create false --local
poetry install
poe install-torchsparse # Building wheel for torchsparse will take a while
pip install gradio_rerun
```

### Inference

In the current version of SpatialLM, input point clouds are considered axis-aligned where the z-axis is the up axis. This orientation is crucial for maintaining consistency in spatial understanding and scene interpretation across different datasets and applications.
Example preprocessed point clouds, reconstructed from RGB videos using [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM), are available in [SpatialLM-Testset](#spatiallm-testset).

Download an example point cloud:

```bash
huggingface-cli download manycore-research/SpatialLM-Testset pcd/scene0000_00.ply --repo-type dataset --local-dir .
```


## License

SpatialLM-Llama-1B is derived from Llama3.2-1B-Instruct, which is licensed under the Llama3.2 license.
SpatialLM-Qwen-0.5B is derived from the Qwen-2.5 series, originally licensed under the Apache 2.0 License.

All models are built upon the SceneScript point cloud encoder, licensed under the CC-BY-NC-4.0 License. TorchSparse, utilized in this project, is licensed under the MIT License.


## Acknowledgements

We would like to thank the following projects that made this work possible:

[Llama3.2](https://github.com/meta-llama) | [Qwen2.5](https://github.com/QwenLM/Qwen2.5) | [Transformers](https://github.com/huggingface/transformers) | [SceneScript](https://github.com/facebookresearch/scenescript) | [TorchSparse](https://github.com/mit-han-lab/torchsparse)
