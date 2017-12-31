# FoolingConvNets

This repository contains code to fool conv nets in PyTorch.
The model can be easily run on CPU so GPU is not needed.


## Requirements
The code is tesed in `python3.5`

PyTorch is needed. You can install it from the following link:

[```http://pytorch.org/```](http://pytorch.org/)

It is recommended to install it using pip and not anaconda.

In addition to this, opencv-python is required. You can install it by:

```pip install opencv-python```

## How to run
Make sure you are in the project path. Verify by:

```pwd```

In order to run, you need to download the pre-trained alexnet model from the internet.
You can do it by issuing the following command:

```wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth```

Afterwards, just run the script using:

```python script.py /path/to/image/to/be/fooled.jpg```

Make sure there are no spaces in the image path.