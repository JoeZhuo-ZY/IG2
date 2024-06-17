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
## Download

```
# To install the core subpackage:
pip install saliency

# To install core and tf1 subpackages:
pip install saliency[tf1]

```

or for the development version:
```
git clone https://github.com/pair-code/saliency
cd saliency
```


## Usage

The saliency library has two subpackages:
*	`core` uses a generic `call_model_function` which can be used with any ML 
	framework.
*	`tf1` accepts input/output tensors directly, and sets up the necessary 
	graph operations for each method.

### Core

Each saliency mask class extends from the `CoreSaliency` base class. This class
contains the following methods:

*   `GetMask(x_value, call_model_function, call_model_args=None)`: Returns a mask
    of
    the shape of non-batched `x_value` given by the saliency technique.
*   `GetSmoothedMask(x_value, call_model_function, call_model_args=None, stdev_spread=.15, nsamples=25, magnitude=True)`: 
    Returns a mask smoothed of the shape of non-batched `x_value` with the 
    SmoothGrad technique.


The visualization module contains two methods for saliency visualization:

* ```VisualizeImageGrayscale(image_3d, percentile)```: Marginalizes across the
  absolute value of each channel to create a 2D single channel image, and clips
  the image at the given percentile of the distribution. This method returns a
  2D tensor normalized between 0 to 1.
* ```VisualizeImageDiverging(image_3d, percentile)```: Marginalizes across the
  value of each channel to create a 2D single channel image, and clips the
  image at the given percentile of the distribution. This method returns a
  2D tensor normalized between -1 to 1 where zero remains unchanged.

If the sign of the value given by the saliency mask is not important, then use
```VisualizeImageGrayscale```, otherwise use ```VisualizeImageDiverging```. See
the SmoothGrad paper for more details on which visualization method to use.

##### call_model_function
`call_model_function` is how we pass inputs to a given model and receive the outputs
necessary to compute saliency masks. The description of this method and expected 
output format is in the `CoreSaliency` description, as well as separately for each method.


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
