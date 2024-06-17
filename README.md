<h2 align="center"> [TPAMI 2024] IG<sup>2</sup>: Integrated Gradient on Iterative Gradient Path for Feature Attribution </h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

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

Install the Saliency
```
pip install saliency
```
Please refer to [saliency repositories](https://github.com/PAIR-code/saliency) for the detailed decription about classes and methods.


##### Examples

[This example iPython notebook](http://github.com/pair-code/saliency/blob/master/Examples_core.ipynb)
showing these techniques is a good starting place.

Here is a condensed example of using IG+SmoothGrad with TensorFlow 2:

```
import saliency.core as saliency
import tensorflow as tf

...

# call_model_function construction here.
def call_model_function(x_value_batched, call_model_args, expected_keys):
	tape = tf.GradientTape()
	grads = np.array(tape.gradient(output_layer, images))
	return {saliency.INPUT_OUTPUT_GRADIENTS: grads}

...

# Load data.
image = GetImagePNG(...)

# Compute IG+SmoothGrad.
ig_saliency = saliency.IntegratedGradients()
smoothgrad_ig = ig_saliency.GetSmoothedMask(image, 
											call_model_function, 
                                            call_model_args=None)

# Compute a 2D tensor for visualization.
grayscale_visualization = saliency.VisualizeImageGrayscale(
    smoothgrad_ig)
```
