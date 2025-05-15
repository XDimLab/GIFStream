# GIFStream: 4D Gaussian-based Immersive Video with Feature Stream (CVPR 2025)
[![Website](https://img.shields.io/badge/website-GIFStream-orange)](https://xdimlab.github.io/GIFStream/) [![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2505.07539)
> Hao Li, Sicheng Li, Xiang Gao, Abudouaihati Batuer, Lu Yu, Yiyi, Liao <br>

## Abstract
**Overview:** *we introduce GIFStream, a novel 4D Gaussian representation enabling high quality representation and efficient compression*

Immersive video offers a 6-Dof-free viewing experience, potentially playing a key role in future video technology. Recently, 4D Gaussian Splatting has gained attention as an effective approach for immersive video due to its high rendering efficiency and quality, though maintaining quality with manageable storage remains challenging. To address this, we introduce GIFStream, a novel 4D Gaussian representation using a canonical space and a deformation field enhanced with time-dependent feature streams. These feature streams enable complex motion modeling and allow efficient compression by leveraging their motion-awareness and temporal correspondence. Additionally, we incorporate both temporal and spatial compression networks for endto-end compression. Experimental results show that GIFStream delivers high-quality immersive video at 30 Mbps, with real-time rendering and fast decoding on an RTX 4090.

## ✅ TODO
- [ ] Release code using [gsplat](https://github.com/nerfstudio-project/gsplat/tree/main) and [gscodec studio](https://github.com/JasonLSC/GSCodec_Studio) framework.

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