# MaskMVS
Yuxin Hou · [Arno Solin](http://arno.solin.fi) · [Juho Kannala](https://users.aalto.fi/~kannalj1/)

Codes for the paper:

* Yuxin Hou, Arno Solin, and Juho Kannala (2019). **Unstructured multi-view depth estimation using mask-based multiplane representation**. *Scandinavian Conference on Image Analysis (SCIA)*. [[preprint on arXiv](https://arxiv.org/abs/1902.02166)]

## Summary

MaskMVS is a method for depth estimation for unstructured multi-view image-pose pairs. In the plane-sweep procedure, the depth planes are sampled by histogram matching that ensures covering the depth range of interest. Unlike other plane-sweep methods, we do not rely on a cost metric to explicitly build the cost volume, but instead infer a multiplane mask representation which regularizes the learning. Compared to many previous approaches, we show that our method is lightweight and generalizes well without requiring excessive training. See the paper for further details.

## Requirements
Tested with:
* Python3
* Numpy
* Pytorch 0.3.0
* CUDA 9 (You can also run without CUDA, but then you need to remove all  `.cuda()` in codes)
* opencv
* imageio (with freeimage plugin)

To install imageio, run `conda install -c conda-forge imageio` or `pip install imageio`. To install the freeimage plugin, run the following Python script once:
```python 
import imageio
imageio.plugins.freeimage.download()
```

### Download pretrained models
We provide our pretrained models of our MaskNet and DispNet to run the example code. Please download the models via [the link]() 

### Run the example
1. Put both the model `masknet_model_best.pth.tar` and the model `dispnet_model_best.pth.tar` under the project folder.
2. Then just run the jupyter notebook file [example.ipynb](https://github.com/AaltoVision/MaskMVS/blob/master/example.ipynb)

## License

This software is distributed under the GNU General Public License (version 3 or later); please refer to the file `LICENSE`, included with the software, for details.
