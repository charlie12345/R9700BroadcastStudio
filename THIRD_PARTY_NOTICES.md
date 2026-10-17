# Third-party notices and credits

The code in this repository is MIT-licensed (see `LICENSE`). It contains **no** third-party model
weights or third-party source code, except the DeepFilterNet MIT license text in
`third_party/deepfilter/`. Everything below is downloaded separately by `scripts/fetch-models.sh`
and keeps its own license. If you redistribute a bundle that includes any of these components,
you must follow that component's license.

## Models and code downloaded at setup

| Component | Used for | License | Source |
|---|---|---|---|
| **Robust Video Matting** (RVM), Lin et al. | realtime matting engine (default) | **GPL-3.0** | https://github.com/PeterL1n/RobustVideoMatting |
| **BiRefNet** / BiRefNet-lite matting, Zheng et al. | Studio / Studio Max engines (optional) | MIT | https://github.com/ZhengPeng7/BiRefNet · https://huggingface.co/ZhengPeng7/BiRefNet-matting |
| **DeepLabV3** (MobileNetV3-Large), torchvision | person / chair semantics, seed prompts | BSD-3-Clause | https://github.com/pytorch/vision |
| **SAM 2.1**, Meta AI | subject seed masks (optional) | Apache-2.0 | https://github.com/facebookresearch/sam2 |
| **Grounding DINO** (base), IDEA Research | finding headphones / headset / microphone by name (optional) | Apache-2.0 | https://huggingface.co/IDEA-Research/grounding-dino-base |
| **MatAnyone 2**, Yang et al., S-Lab NTU | subject-lock tracker (optional) | **NTU S-Lab License 1.0, non-commercial only** | https://github.com/pq-yang/MatAnyone2 |
| **DeepFilterNet** LADSPA plugin, Schröter et al. | noise-removal microphone | MIT / Apache-2.0 | https://github.com/Rikorose/DeepFilterNet |
| **BlazeFace short-range**, Google AI Edge | face box + eye/nose keypoints for fun props | Apache-2.0 | https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector |

### Important license consequences

* **RVM is GPL-3.0.** This repository only loads RVM at runtime from a separate download; it does
  not include RVM code. A distribution that *bundles* RVM with this app must be distributed under
  GPL-3.0 as a whole.
* **MatAnyone 2 is for non-commercial use only.** The app works without it (it falls back to RVM +
  the semantic guard). `scripts/fetch-models.sh --lock` asks you to accept its license before
  downloading. Commercial use needs permission from the authors.

## System software used at runtime (not distributed)

PyTorch (BSD-3-Clause), ROCm (mostly MIT), OpenCV (Apache-2.0), PyQt6 / Qt (GPL-3.0 / commercial;
PyQt6 is GPL-3.0), NumPy (BSD-3-Clause), MediaPipe Tasks (Apache-2.0), Hugging Face Transformers (Apache-2.0), OmegaConf / Hydra
(BSD-3-Clause / MIT), FFmpeg (LGPL/GPL), PipeWire (MIT), v4l2loopback (GPL-2.0, optional).

Note: PyQt6 is GPL-3.0. Like RVM, it is installed separately; a binary distribution that bundles
PyQt6 must follow its license.

## Papers and methods

Implemented from the papers (no code copied):

* Guided filter: K. He, J. Sun, X. Tang. *Guided Image Filtering.* ECCV 2010 / TPAMI 2013. Fast
  variant: K. He, J. Sun. *Fast Guided Filter.* arXiv:1505.00996, 2015.
* Blur-fusion foreground estimation: M. Forte, F. Pitié. *Approximate Fast Foreground Colour
  Estimation.* ICIP 2021.
* Box-filter Gaussian approximation: W. Wells. *Efficient Synthesis of Gaussian Filters by
  Cascaded Uniform Filters.* TPAMI 1986; P. Kovesi. *Fast Almost-Gaussian Filtering.* DICTA 2010.
* "Stable core, detailed boundary" fusion of a tracked subject with an edge matte: from MatAnyone
  (Yang et al., CVPR 2025).
* "Grounded SAM" recipe (open-vocabulary detector → SAM prompts): T. Ren et al. *Grounded SAM:
  Assembling Open-World Models for Diverse Visual Tasks.* arXiv:2401.14159, 2024.
* Clean-plate difference keying is a standard VFX technique. The coarse-hint → refine and
  straight-foreground ideas were inspired by Corridor Crew's CorridorKey. **No CorridorKey code
  or weights are used.**

Research consulted for the subject lock (not used in code): SAM2Matting (arXiv:2606.27339),
VideoMaMa (arXiv:2601.14255), Generative Video Matting (arXiv:2508.07905).

## Citations

```bibtex
@inproceedings{lin2022rvm,
  title     = {Robust High-Resolution Video Matting with Temporal Guidance},
  author    = {Lin, Shanchuan and Yang, Linjie and Saleemi, Imran and Sengupta, Soumyadip},
  booktitle = {WACV},
  year      = {2022}
}
@inproceedings{yang2026matanyone2,
  title     = {{MatAnyone 2}: Scaling Video Matting via a Learned Quality Evaluator},
  author    = {Yang, Peiqing and Zhou, Shangchen and Hao, Kai and Tao, Qingyi},
  booktitle = {CVPR},
  year      = {2026}
}
@inproceedings{yang2025matanyone,
  title     = {{MatAnyone}: Stable Video Matting with Consistent Memory Propagation},
  author    = {Yang, Peiqing and Zhou, Shangchen and Zhao, Jixin and Tao, Qingyi and Loy, Chen Change},
  booktitle = {CVPR},
  year      = {2025}
}
@article{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal = {arXiv preprint arXiv:2408.00714},
  year    = {2024}
}
@inproceedings{liu2024groundingdino,
  title     = {Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author    = {Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Jiang, Qing and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and Zhang, Lei},
  booktitle = {ECCV},
  year      = {2024}
}
@article{zheng2024birefnet,
  title   = {Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author  = {Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal = {CAAI Artificial Intelligence Research},
  volume  = {3},
  pages   = {9150038},
  year    = {2024}
}
@inproceedings{chen2017deeplabv3,
  title  = {Rethinking Atrous Convolution for Semantic Image Segmentation},
  author = {Chen, Liang-Chieh and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  booktitle = {arXiv:1706.05587},
  year   = {2017}
}
@inproceedings{schroeter2022deepfilternet2,
  title     = {{DeepFilterNet2}: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio},
  author    = {Schröter, Hendrik and Escalante-B., Alberto N. and Rosenkranz, Tobias and Maier, Andreas},
  booktitle = {IWAENC},
  year      = {2022}
}
@inproceedings{forte2021blurfusion,
  title     = {Approximate Fast Foreground Colour Estimation},
  author    = {Forte, Marco and Piti{\'e}, Fran{\c{c}}ois},
  booktitle = {ICIP},
  year      = {2021}
}
```

Thanks to every author above for releasing their work.
