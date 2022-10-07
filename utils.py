import numpy as np
import matplotlib.pyplot as plt
import scipy 
import matplotlib.pyplot as plt
from lipschitz_normalizers import infinity_norm_normalization, two_to_infinity_norm_normalization
import tensorflow as tf
import deel
from deel.lip.initializers import SpectralInitializer
from deel.lip.layers import SpectralDense, SpectralConv2D, LipschitzLayer, Condensable, InvertibleDownSampling, FrobeniusDense
from deel.lip.activations import MaxMin, GroupSort, GroupSort2, FullSort
from deel.lip.model import Sequential as DeelSequential
from deel.lip.activations import PReLUlip, GroupSort, FullSort
from lipschitz_normalizers import NormalizedDense, SpectralInfConv2D
from tensorflow.keras.layers import InputLayer, Dense, Conv2D
import wandb
from deel.lip.normalizers import reshaped_kernel_orthogonalization


def compute_norm(vec):
    square_norm = tf.reduce_sum(vec ** 2., axis=tuple(range(1, len(vec.shape))), keepdims=True)
    return square_norm ** 0.5

def check_grads(model, seeds, plot_wandb, condense=False):
    if condense:
        print("WARNING: check_grads performs model.condense()")
        model.condense()
    
    table = wandb.Table(columns=['Singular', 'min-2-inf', 'max-2-inf', 'min-inf', 'max-inf'])
    for layer in model.layers:
        if isinstance(layer, SpectralDense) or isinstance(layer, FrobeniusDense) or isinstance(layer, NormalizedDense) or isinstance(layer, SpectralConv2D) or isinstance(layer, SpectralInfConv2D) or isinstance(layer, Dense) or isinstance(layer, Conv2D):
            if isinstance(layer, NormalizedDense) or isinstance(layer, SpectralInfConv2D):
                kernel = layer.normalizer_fun(layer.kernel, layer.inf_norm_bounds)
            elif isinstance(layer, FrobeniusDense):
                kernel = layer.kernel / tf.norm(layer.kernel, axis=layer.axis_norm) * layer._get_coef()
            elif isinstance(layer, Dense):
                kernel = layer.kernel
            else:
                kernel = reshaped_kernel_orthogonalization(
                            layer.kernel,
                            layer.u,
                            layer._get_coef(),
                            layer.niter_spectral,
                            layer.niter_bjorck,
                            layer.beta_bjorck)[0]
                
            reshaped = kernel.numpy().reshape((-1,kernel.shape[-1]))
            singular = scipy.linalg.svdvals(reshaped)
            inf2norm = tf.reduce_sum(reshaped**2, axis=0, keepdims=True) ** 0.5
            infnorm = tf.reduce_sum(tf.math.abs(reshaped), axis=0, keepdims=True)
            datum = [f"{np.max(singular)}", f"{np.min(inf2norm)}", f"{np.max(inf2norm)}", f"{np.min(infnorm)}", f"{np.max(infnorm)}"]
            table.add_data(*datum)
            print(f"S={datum[0]} 2-inf=[{datum[1],datum[2]}] inf=[{datum[3],datum[4]}] 2=[{np.min(singular),np.max(singular)}]")
    if plot_wandb:
        wandb.log({"Kernel Norms": table})

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(seeds)
        y = model(seeds, training=True)
        grad  = tape.gradient(y, seeds)
        grad_norm = compute_norm(grad)
        
    idx = np.argmax(grad_norm)
    print(idx, grad_norm[idx])
    seed = seeds[idx,:]
    seed = seed[np.newaxis,:]
    # print(seed)
    
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(seed)
        y = model(seed, training=True)
    grad_seed  = tape.gradient(y, seed)
    grad_seed_norm = compute_norm(grad_seed)
    
    print("GradSeedNorm:", grad_seed_norm)
    if plot_wandb:
        wandb.log({"GradSeedNorm": float(np.squeeze(grad_seed_norm.numpy()))})
    
    eps = 1e-2
    seed_next = seed + eps * grad_seed
    
    z_next, z = seed_next, seed
    dx = seed_next - seed
    df = dx
    print("LLC = Local Lipschitz Constant")
    print("Input, LLC wrt Input, LLC wrt previous layer")
    datum = ["Input", f"{np.sum(dx**2)**0.5}", f"{np.sum(dx**2)**0.5}"]
    data = [datum]
    print("Input: ", datum[0], datum[1], datum[2])
    for i, layer in enumerate(model.layers):
        z_next = layer(z_next, training=True)
        z  = layer(z, training=True)
        dd = df
        df = z_next - z
        datum = [str(type(layer)), f"{np.sum(df**2)**0.5 / np.sum(dx**2)**0.5:.2f}", f"{np.sum(df**2)**0.5 / np.sum(dd**2)**0.5:.2f}"]
        print(i, datum[0], datum[1], datum[2])
        data.append(datum)
    datum = ["Output", f"{np.sum(df**2)**0.5}", f"{np.sum(df**2)**0.5}"]
    data.append(datum)
    print("Output: ", datum[0], datum[1], datum[2])
    
    print(np.min(grad_norm), np.max(grad_norm))
    plt.hist(np.squeeze(grad_norm))
    if plot_wandb:
        wandb.log({"gradients": wandb.Histogram(np.squeeze(grad_norm))})
        columns = ["Layer name", "Global Lipschitz", "Local Lipschitz"]
        table =  wandb.Table(data=data, columns=columns)
        wandb.log({"Lipschitz": table})

    plt.show()