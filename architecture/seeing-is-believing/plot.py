"""
Plot utilities to create similar figure as the one in the paper
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch

def plot_input(model, input, name):
    """
    This is modified, based on the reference implementation
    https://github.com/KindXiaoming/BIMT/blob/main/mnist_3.5.ipynb
    """
    assert len(input.shape) == 2, "expected single input channel"
    fig=plt.figure(figsize=(10,20), clear=True)
    ax = fig.add_subplot(projection='3d')
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 4, 1]))
    """
    Plots the layers
    """
    for index, layer in enumerate(model.layer_modules):
        input_shape = layer.input_dimension
        output_shape = layer.output_dimension

        is_final_layer = (index + 1) == len(model.layer_modules)
        print(f"layer {input_shape}, is_final_layer {is_final_layer}")
        in_coordinates = layer.coordinates.in_coordinates
        print(in_coordinates.shape)
        if index == 0:
            ax.scatter(
                in_coordinates[:,0].detach().numpy(), 
                in_coordinates[:,1].detach().numpy(),
                [0]*input_shape, 
                s=5, 
                alpha=0.5, 
                c=input.detach().numpy()[:,::-1].reshape(-1,)
            )
        else:    
            ax.scatter(
                in_coordinates[:,0].detach().numpy(), 
                in_coordinates[:,1].detach().numpy(),
                [index]*input_shape, 
                s=5, 
                alpha=0.5, 
                color="black"
            )
        # we also plot the output
        if is_final_layer:
            output_coordinates = layer.coordinates.out_coordinates
            print(output_coordinates)
            ax.scatter(
                output_coordinates[:,0].detach().numpy(), 
                output_coordinates[:,1].detach().numpy(),
                [index + 1]*output_shape, 
                s=5, 
                alpha=0.5, 
                color="black"
            )
            # annotate the classes
            for i in range(10):
                ax.text(output_coordinates[i,0], output_coordinates[i,1], 3.05, "{}".format(i))#, c="black")
    """
    Plots the connections
    """
    for ii in range(len(model.layer_modules)):
        layer = model.layer_modules[ii]
        p = layer.linear_layer.weight.clone()
        p_shp = p.shape
        p = p / torch.abs(p).max()

        for i in range(p_shp[0]):
            if i % 20 == 0:
                print(f"Layer {ii}, index: {i}")
            for j in range(p_shp[1]):
                out_xy = layer.coordinates.out_coordinates[i].detach().numpy()
                in_xy = layer.coordinates.in_coordinates[j].detach().numpy()
                plt.plot([out_xy[0], in_xy[0]], [out_xy[1], in_xy[1]], [ii+1,ii], lw=1*np.abs(p[i,j].detach().numpy()), color="blue" if p[i,j]>0 else "red")

    ax.set_zlim(-0.1, 9)
    ax.set_axis_off()
    plt.savefig(f"{name}.png", bbox_inches='tight')

