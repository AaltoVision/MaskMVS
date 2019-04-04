# MaskMVS
## Unstructured Multi-View Depth Estimation Using Mask-Based Multiplane Representation

For the paper, please see our [arXiv link](https://arxiv.org/abs/1902.02166)


### Requirements

- **Python3**
- **Numpy**
- **Pytorch 0.3.0**
- **CUDA 9**
- **Opencv**
- **imageio** (with freeimage plugin): Run ``conda install -c conda-forge imageio`` or ``pip install imageio``. To install freeimage plugin, run the following Python script once:
    ```python 
    import imageio
    imageio.plugins.freeimage.download()
    ```
### Download pretrained models
We provide our pretrained models of our MaskNet and DispNet to run the example code.Please download the model via [the link]() 


### Run the example
Put both the model ```masknet_model_best.pth.tar``` and the model ```dispnet_model_best.pth.tar``` under the project folder.

Then just run the jupyter notebook file [example.ipynb](https://github.com/AaltoVision/MaskMVS/blob/master/example.ipynb)
