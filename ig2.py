from saliency.core.base import CoreSaliency
import numpy as np
from saliency.core.base import INPUT_OUTPUT_GRADIENTS

# Output of representation layer
REP_LAYER_VALUES = 'REP_LAYER_VALUES'
# Gradients w.r.t. distance between representations of input and references
REP_DISTANCE_GRADIENTS = 'REP_DISTANCE_GRADIENTS'

# import torch
# def _get_norm_batch(x, p):
#     batch_size = x.size(0)
#     return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)

# def _batch_multiply_tensor_by_vector(vector, batch_tensor):
#     """Equivalent to the following
#     for ii in range(len(vector)):
#         batch_tensor.data[ii] *= vector[ii]
#     return batch_tensor
#     """
#     return (
#         batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()

# def batch_multiply(float_or_vector, tensor):
#     if isinstance(float_or_vector, torch.Tensor):
#         assert len(float_or_vector) == len(tensor)
#         tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
#     elif isinstance(float_or_vector, float):
#         tensor *= float_or_vector
#     else:
#         raise TypeError("Value has to be float or torch.Tensor")
#     return tensor

# def normalize_by_pnorm(x, p=2, small_constant=1e-6):
#     """
#     Normalize gradients for gradient (not gradient sign) attacks.
#     :param x: tensor containing the gradients on the input.
#     :param p: (optional) order of the norm for the normalization (1 or 2).
#     :param small_constant: (optional float) to avoid dividing by zero.
#     :return: normalized gradients.
#     """
#     # loss is averaged over the batch so need to multiply the batch
#     # size to find the actual gradient of each input sample

#     assert isinstance(p, float) or isinstance(p, int)
#     norm = _get_norm_batch(x, p)
#     norm = torch.max(norm, torch.ones_like(norm) * small_constant)
#     return batch_multiply(1. / norm, x)

# def clamp_by_pnorm(x, p, r):
#     assert isinstance(p, float) or isinstance(p, int)
#     norm = _get_norm_batch(x, p)
#     if isinstance(r, torch.Tensor):
#         assert norm.size() == r.size()
#     else:
#         assert isinstance(r, float)
#     factor = torch.min(r / norm, torch.ones_like(norm))
#     return batch_multiply(factor, x)

# Numpy implementation
def normalize_by_2norm(x):
    batch_size = x.shape[0]
    norm = np.power(np.sum(np.power(np.abs(x),2).reshape(batch_size, -1),axis=1),1./2)
    normed_x = np.moveaxis(x,0,-1) / norm
    return np.moveaxis(normed_x,-1,0)

class IG2(CoreSaliency):
    """
        Implementation of the integrated gradients on iterative gradient path method.
    """
    def GetMask(self, 
                x_value, 
                baselines, #reference
                call_model_function, 
                call_model_args=None,
                steps=201, 
                step_size=256.0, 
                clip_min_max=[0, 255],
                return_path=False,):
        """Returns an IG2 mask.

        Args:
            x_value: Input ndarray. Only support single sample, shape [C,H,W] or [H,W]
            baselines: Reference ndarray. Counter reference samples in batch, shape [B,C,H,W] or [B,H,W]

            call_model_function: A function that interfaces with a model to return
            specific data in a dictionary when given an input and other arguments.
            Expected function signature:
            - call_model_function(x_value_batch,
                                    call_model_args=None,
                                    expected_keys=None):
                x_value_batch - Input for the model, given as a batch (i.e. dimension
                0 is the batch dimension, dimensions 1 through n represent a single
                input).
                call_model_args - Other arguments used to call and run the model.
                expected_keys - List of keys that are expected in the output. For this
                method (IG2), the expected keys are
                REP_LAYER_VALUES - Output of representation layer with respect to the input.

            call_model_args: The arguments that will be passed to the call model
            function, for every call of the model.
            x_baseline: References used in path search and integration. Support multiple 
                        references. Shape [B,C,H,W]
            x_steps: Iteration number of gradient descent-based path search 
            Step_size: Step size of gradient descent-based path search 
            clip_min_max: Bounds for searched points, depending on datasets
            return_path: Wheater return GradPath for integration
        """
        assert  len(baselines.shape) - len(x_value.shape) == 1
        x_value = np.asarray([x_value], dtype=np.float32)
        baselines = np.asarray(baselines, dtype=np.float32)
        x_value = np.repeat(x_value, baselines.shape[0], axis=0)

        #GradPath search
        print('GradPath search...')
        path = self.Get_GradPath(x_value, baselines, call_model_function, call_model_args, 
                                 steps, step_size, clip_min_max)
        np.testing.assert_allclose(x_value, path[0], rtol=0.01)

        # Integrate gradients on GradPath, simliar to IG
        print('Integrate gradients on GradPath...')
        attr = np.zeros_like(x_value, dtype=np.float32)
        x_old = x_value
        for i, x_step in enumerate(path[1:]):
            call_model_output = call_model_function(
                x_old,
                call_model_args=call_model_args,
                expected_keys=[INPUT_OUTPUT_GRADIENTS])

            gradient = call_model_output[INPUT_OUTPUT_GRADIENTS]

            # IG^2 =  \frac{\partial f(\gamma^{G}(\frac{j}{k}))}{\partial x_i} \times
            # \frac{\partial \Vert \tilde{f}(\gamma^{G}(\frac{j}{k})) - \tilde{f}(x^r) \Vert_2^2 }{ \partial x_i}
            # \times \frac{\eta}{W_j}} Eq. 3 in paper
            attr += (x_old - x_step) * gradient
            x_old = x_step

        if return_path:
            return np.mean(attr,axis=0), path
        else:
            return np.mean(attr,axis=0)

    def Get_GradPath(self,
              x_value,
              baselines, #reference
              call_model_function,
              call_model_args=None,
              steps=201,
              step_size=256.0,
              clip_min_max=[0, 255],
              ):
        """" Iteratively search the counterfactuals based on gradient descent, 
             output GradPath for integration.
        Args:
            call_model_args: REP_DISTANCE_GRADIENTS - Gradients w.r.t. distance between representations 
                    of input and reference
        """
        
        # Calculate the layer representation of baselines
        data = call_model_function(baselines,
            call_model_args=call_model_args,
            expected_keys=[REP_LAYER_VALUES])
        call_model_args.update({'layer_baseline':data[REP_LAYER_VALUES].detach(),})

        #  Iteratively search in sample space to close the reference rep
        delta = np.zeros_like(x_value)
        path = []
        path.append(x_value)

        for i in range(steps):
            data = call_model_function(x_value + delta,
                call_model_args=call_model_args,
                expected_keys=[REP_DISTANCE_GRADIENTS])
            
            grad = data[REP_DISTANCE_GRADIENTS]
            loss = data['loss']

            # grad = normalize_by_pnorm(torch.Tensor(np.array(grad)))
            # delta = delta + batch_multiply(step_size, grad).numpy()
            grad = normalize_by_2norm(grad)
            delta = delta + grad * step_size
            delta = np.clip(x_value + delta, clip_min_max[0], clip_min_max[1]
                            ) - x_value

            x_adv = x_value + delta
            if i % 100 == 0:
                print(f'{i} iterations, rep distance Loss {loss}')
            path.append(x_adv)

        return path
