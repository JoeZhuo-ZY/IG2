<h2 align="center"> [TPAMI 2024] IG<sup>2</sup> Integrated Gradient on Iterative Gradient Path for Feature Attribution </h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

![](./image/abstract.gif)

## 📰 News & Update

- **[2024.04.18]** The original code will be available soon! ```Welcome to try it according to the [tutorial](./tutorial-submodular-cub.ipynb)!```
- **[2024.04.06]** The paper has been accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence
<!-- 
## 🐳 Model Zoo

The checkpoints of models explained in the paper. (Please download and put the models to the path `ckpt/keras_model`):

| Datasets | Model |
| -|-|
| Celeb-A | [keras-ArcFace-R100-Celeb-A.h5](https://huggingface.co/RuoyuChen/SMDL-Attribution/resolve/main/keras_model/keras-ArcFace-R100-Celeb-A.h5?download=true) |
| VGG-Face2  | [keras-ArcFace-R100-VGGFace2.h5](https://huggingface.co/RuoyuChen/SMDL-Attribution/blob/main/keras_model/keras-ArcFace-R100-VGGFace2.h5)  |
| CUB-200-2011 | [cub-resnet101.h5](https://huggingface.co/RuoyuChen/SMDL-Attribution/resolve/main/keras_model/cub-resnet101.h5?download=true), [cub-resnet101-new.h5](https://huggingface.co/RuoyuChen/SMDL-Attribution/resolve/main/keras_model/cub-resnet101-new.h5?download=true), [cub-efficientnetv2m.h5](https://huggingface.co/RuoyuChen/SMDL-Attribution/resolve/main/keras_model/cub-efficientnetv2m.h5?download=true), [cub-mobilenetv2.h5](https://huggingface.co/RuoyuChen/SMDL-Attribution/resolve/main/keras_model/cub-mobilenetv2.h5?download=true), [cub-vgg19.h5](https://huggingface.co/RuoyuChen/SMDL-Attribution/resolve/main/keras_model/cub-vgg19.h5?download=true)|

Uncertainty Estimation Models (Please download and put the models to the path `ckpt/pytorch_model`):

| Datasets | Model |
| -|-|
|Celeb-A| [edl-101-10177.pth](https://huggingface.co/RuoyuChen/SMDL-Attribution/resolve/main/pytorch_model/edl-101-10177.pth?download=true) |
| VGG-Face2 | [edl-101-8631.pth](https://huggingface.co/RuoyuChen/SMDL-Attribution/resolve/main/pytorch_model/edl-101-8631.pth?download=true) |
| CUB-200-2011 | [cub-resnet101-edl.pth](https://huggingface.co/RuoyuChen/SMDL-Attribution/resolve/main/pytorch_model/cub-resnet101-edl.pth?download=true) |

## 🛠️ Environment (Updating)

```python
opencv-python
opencv-contrib-python
mtutils
xplique>=1.0.3
```

```
conda create -n smdl python=3.10
conda activate smdl
python3 -m pip install tensorflow[and-cuda]

pip install git+https://github.com/facebookresearch/segment-anything.git
```

## 😮 Highlights

|Sub-Region Division Method| Attribution Visualization | Org. Prediction Score | Highest Prediction Score | Insertion AUC Score | 
|:--:|:--:|:--:|:--:|:--:|
| SLICO | ![](image/slico.png) | 0.7262 | 0.9522 | 0.7604 |
| SEEDS | ![](image/seeds.png) | 0.7262 | 0.9918 | 0.8862 |
| Prior Saliency Map + Patch | ![](image/prior_saliency_division.png) | 0.7262 | 0.9710 | 0.7236 |
| Segment Anything Model | ![](image/sam.png) | 0.7262 | 0.9523 | 0.6803 |

## 🗝️ How to Run (Updating)

### 1. Generate prior saliency map

First, the priori saliency maps for sub-region division needs to be generated.

```
CUDA_VISIBLE_DEVICES=0 python generate_explanation_maps.py
```

Don't forget to open this file and revise the variable `mode` and `net_mode`:

- `mode`: ["Celeb-A", "VGGFace2", "CUB", "CUB-FAIR"]

- `net_mode`: ["resnet", "efficientnet", "vgg19", "mobilenetv2"], note that these net_mode only for `mode` is CUB-FAIR.



### 2. Compute attribution

```
CUDA_VISIBLE_DEVICES=0 python smdl_explanation.py
```

## 👍 Acknowledgement

[Xplique](https://deel-ai.github.io/xplique/latest/): a Neural Networks Explainability Toolbox

[Score-CAM](https://github.com/tabayashi0117/Score-CAM/): a third-party implementation with Keras.

[Segment-Anything](https://github.com/facebookresearch/segment-anything): a new AI model from Meta AI that can "cut out" any object, in any image, with a single click.

## ✏️ Citation

```bibtex
@inproceedings{chen2024less,
  title={Less is More: Fewer Interpretable Region via Submodular Subset Selection},
  author={Chen, Ruoyu and Zhang, Hua and Liang, Siyuan and Li, Jingzhi and Cao, Xiaochun},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
-->
