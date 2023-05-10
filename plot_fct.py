import matplotlib.pyplot as plt 
import numpy as np
import torch
plt.style.use('dark_background')



def plot_test(l_im,cmap="coolwarm"):

    fig,axes=plt.subplots(nrows=len(l_im),ncols=3,figsize=(35,15)) 

    clim=(100,-100)

    for n,line in enumerate(l_im):
        [x,gen,y] = line
        im = axes[n,0].imshow(torch.squeeze(x).cpu().numpy(),cmap=cmap)
        im = axes[n,1].imshow(torch.squeeze(gen).cpu().numpy(),cmap=cmap)
        im = axes[n,2].imshow(torch.squeeze(y).cpu().numpy(),cmap=cmap)

    
    cols=['sat','generated','model']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    fig.tight_layout()
    plt.show()



