<h2 align="center"> [TPAMI 2024] IG<sup>2</sup>: Integrated Gradient on Iterative Gradient Path for Feature Attribution </h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>
[Y. Zhuo and Z. Ge, "IG2: Integrated Gradient on Iterative Gradient Path for Feature Attribution," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 11, pp. 7173-7190, Nov. 2024, doi: 10.1109/TPAMI.2024.3388092. ](https://arxiv.org/abs/2406.10852)
![](./image/abstract.gif)

## 📰 News & Update

- **[2024.06.17]** The detailed usage and examples on more datasets are updated!
- **[2024.04.18]** The implementation code has been uploaded! 
- **[2024.04.06]** The paper has been accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence.

## Acknowledgements
- [Saliency](https://github.com/PAIR-code/saliency) Our code is built under the framework of Saliency project by PAIR. Shout out to PAIR-code!
- [SHAP](https://github.com/PAIR-code/saliency) An explanation framework for machine learning models. 
- [Visualizing the Impact of Feature Attribution Baselines](https://distill.pub/2020/attribution-baselines/) A blog carefully introduces the IG (Integrated Gradients) method and its baseline choice.
- 

## Usage

Install the Saliency. Please refer to [saliency repositories](https://github.com/PAIR-code/saliency) for the detailed descriptions.
```
pip install saliency
```

Our method is implemented by the class ```IG2```, in file [```ig2.py```](https://github.com/JoeZhuo-ZY/IG2/blob/main/ig2.py). This class contains the following methods (the default parameters are suggested for ImageNet samples):
- Get_GradPath(x_value, baselines, call_model_function, call_model_args=None, steps=201, step_sizes=256.0, clip_min_max=[0,255]): Iteratively searchs the counterfactuals based on gradinet descent, building GradPath for integration.
- GetMask(x_value, baselines, call_model_function, call_model_args=None, steps=201, step_sizes=256.0, clip_min_max=[0,255]): Integrates the gradients on the GradPath, returns a saliency mask.


## Examples

[This example iPython notebook](https://github.com/JoeZhuo-ZY/IG2/blob/main/example_ImageNet.ipynb)
showing IG2 example for attritbuting the features of ImageNet samples with Pytorch.


```
import ig2
from saliency.core.base import ...

...

# Calculate the gradients of representation distance (MSE) between the explained image and searched path points.
def call_model_function(x_value_batched, call_model_args, expected_keys):
    elif REP_DISTANCE_GRADIENTS in expected_keys:
        loss_fn = torch.nn.MSELoss()         
        baseline_conv = call_model_args['layer_baseline']
        input_conv = rep_layer_outputs[REP_LAYER_VALUES]
        loss = -1 * loss_fn(input_conv, baseline_conv)
        loss.backward()
        grads = images.grad.data
        grads = torch.movedim(grads, 1, 3)
        gradients = grads.cpu().detach().numpy()
        return {REP_DISTANCE_GRADIENTS: gradients,
                'loss':loss}

...

# Load explained sample and references. (You can custom your own datasets here.)
rnd_idx = np.random.choice(all_references.shape[0],replace=False, size=n_reference)
references = all_references[rnd_idx]

# Compute IG2.
explainer = ig2.IG2()
ig2_mask = explainer.GetMask(im,references,
    call_model_function,call_model_args,steps=201,step_size=256.0,clip_min_max=[0,255],)

# Compute a 2D tensor for visualization.
mask_grayscale = vis.VisualizeImageGrayscale(ig2_mask)
f, ax = plt.subplots()
ShowGrayscaleImage(mask_grayscale, ax)
```
