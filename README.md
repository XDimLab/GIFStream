# GIFStream: 4D Gaussian-based Immersive Video with Feature Stream (CVPR 2025)
[![Website](https://img.shields.io/badge/website-GIFStream-orange)](https://xdimlab.github.io/GIFStream/) [![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2505.07539)
> Hao Li, Sicheng Li, Xiang Gao, Abudouaihati Batuer, Lu Yu, Yiyi Liao <br>

## Abstract
**Overview:** *we introduce GIFStream, a novel 4D Gaussian representation enabling high quality representation and efficient compression*

Immersive video offers a 6-Dof-free viewing experience, potentially playing a key role in future video technology. Recently, 4D Gaussian Splatting has gained attention as an effective approach for immersive video due to its high rendering efficiency and quality, though maintaining quality with manageable storage remains challenging. To address this, we introduce GIFStream, a novel 4D Gaussian representation using a canonical space and a deformation field enhanced with time-dependent feature streams. These feature streams enable complex motion modeling and allow efficient compression by leveraging their motion-awareness and temporal correspondence. Additionally, we incorporate both temporal and spatial compression networks for endto-end compression. Experimental results show that GIFStream delivers high-quality immersive video at 30 Mbps, with real-time rendering and fast decoding on an RTX 4090.

## 💻 Installation and Experiments
### Repo. & Environment
```bash
# Clone the repo.
git clone https://github.com/XDimLab/GIFStream.git --recursive
cd GIFStream

# Make a conda environment
conda create --name GIFStream python=3.10
conda activate GIFStream
```

### Packages Installation

Please install [Pytorch](https://pytorch.org/get-started/locally/) first. 

Then, you can install the extended gsplat library with GIFStream training, rendering and compression features.

```bash
pip install .
```

If you want to do further development based on this framework, you can use following command to install Python packages in editable mode.
```bash
pip install -e . # (develop)
```

Same as gsplat, we need to install some extra dependencies.

```bash
cd examples
pip install -r requirements.txt

cd ../third_party/MLEntropy
mkdir build
cd build
cmake ../cpp -DCMAKE_BUILD_TYPE=Release
make -j
```
### Dataset Preparation
For [Neur3D](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0) dataset, please first download the dataset and the organization of files should be like this:
```md

└── Neur3D/
    ├── coffee_martini/
    │   ├── cam00.mp4
    │   └── ...
    ├── cook_spinach/
    │   ├── cam00.mp4
    │   └── ...
    └── ...
```
Then preprocess the data using the script as below.
```bash
python dataset_process/n3d_video_process.py --root_dir your_path_to_neur3d_dataset
```

For readers who are accessible to MPEG GSC dataset, please first download the dataset and the organization of files should be like this:
```md

└── mpeg/
    ├── Bartender/
    │   ├── yuv
    │   │   └── vxx_framexxx.yuv
    │   │   └── ...
    ├── Cinema/
    │   ├── yuv
    │   │   └── vxx_framexxx.yuv
    │   │   └── ...
    └── ...
```
Then preprocess the data using the script as below. It will transform yuv files into png files and conduct colmap for the first frame setted in gsc_tool/scene_info.py. Note that this is not the process conducted in the article and performance is not perfectly matched, since our code has been reimplemented.

```bash
bash gsc_tool/preprocess.sh
```
### GIFStream Training and Compression

We provide a script that enables end-to-end compression-aware training and compression for videos containing several gops.

```bash
bash examples/benchmarks/multigop_gifstream.sh
```
## ✅ TODO
- [x] Release code using [gsplat](https://github.com/nerfstudio-project/gsplat/tree/main) and [gscodec studio](https://github.com/JasonLSC/GSCodec_Studio) framework.

## ⭐ Acknowledgement
This project is bulit on [gsplat](https://github.com/nerfstudio-project/gsplat), [GScodec Studio](https://github.com/JasonLSC/GSCodec_Studio), [Scaffold-GS](https://github.com/city-super/Scaffold-GS) and [DCVC-HEM](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-HEM). We thank all contributors for such great open-source projects.

## 🎓 Citation
Please cite our paper if you find this repository useful:

```bibtex
@misc{li2025gifstream4dgaussianbasedimmersive,
    title={GIFStream: 4D Gaussian-based Immersive Video with Feature Stream}, 
    author={Hao Li and Sicheng Li and Xiang Gao and Abudouaihati Batuer and Lu Yu and Yiyi Liao},
    year={2025},
    eprint={2505.07539},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2505.07539}, 
}
```
