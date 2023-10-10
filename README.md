# Ultra Dual-Path Compression and Decompression

This is the repository for a Pytorch-based implementation of the compression and decompression module in "Ultra Dual-Path Compression For Joint Echo Cancellation And Noise Suppression". The ultra dual-path compression module can compress the input multi-track spectra with large numbers of frames and frequency (T-F) bins into feature maps with small numbers of T-F bins, facilitating the fast processing for dual-path models (e.g., fullsubnet, 2D-convolution network). The decompression module transforms the compressed feature map back to the shapes of spectra for further processing. Details are recommended to be found in `ultra_dual_path_compression.ipynb`, while `ultra_dual_path_compression.py` contains some legacy code.

Demos can be found in [DemoPage](https://hangtingchen.github.io/ultra_dual_path_compression.github.io/).

Please refer to our paper with the latest version on [Arxiv](https://arxiv.org/abs/2308.11053) for details. This paper is also accepted by [INTERSPEECH2023](https://www.isca-speech.org/archive/interspeech_2023/chen23t_interspeech.html).

Please cite the paper if you found this module useful.
```
@article{DBLP:journals/corr/abs-2308-11053,
  author       = {Hangting Chen and
                  Jianwei Yu and
                  Yi Luo and
                  Rongzhi Gu and
                  Weihua Li and
                  Zhuocheng Lu and
                  Chao Weng},
  title        = {Ultra Dual-Path Compression For Joint Echo Cancellation And Noise
                  Suppression},
  journal      = {CoRR},
  volume       = {abs/2308.11053},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2308.11053},
  doi          = {10.48550/arXiv.2308.11053},
  eprinttype    = {arXiv},
  eprint       = {2308.11053},
  timestamp    = {Fri, 25 Aug 2023 12:09:57 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2308-11053.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Disclaimer
This is not an officially supported Tencent product.
