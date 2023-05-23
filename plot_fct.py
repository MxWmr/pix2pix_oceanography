import matplotlib.pyplot as plt 
import numpy as np
import torch
plt.style.use('dark_background')



def plot_test(l_im,date,save_path,cmap="coolwarm",save=True):

    fig,axes=plt.subplots(nrows=len(l_im),ncols=3,figsize=(35,15)) 

    clim=(100,-100)

    for n,line in enumerate(l_im):
        [x,gen,y] = line
        x =  torch.squeeze(x).cpu().numpy()
        gen = torch.squeeze(gen).cpu().numpy()
        y = torch.squeeze(y).cpu().numpy()
        m = min(np.amin(x),np.amin(gen),np.amin(y))
        M = max(np.amax(x),np.amax(gen),np.amax(y))
        im = axes[n,0].imshow(x,cmap=cmap,vmin=m,vmax=M)
        im = axes[n,1].imshow(gen,cmap=cmap,vmin=m,vmax=M)
        im = axes[n,2].imshow(y,cmap=cmap,vmin=m,vmax=M)

    
    cols=['sat','generated','model']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    fig.tight_layout()
    if save:
        plt.savefig(save_path+date+'images.png')
    plt.show()




def plot_diff(l_im,date,save_path,cmap="coolwarm",save=True):

    fig,axes=plt.subplots(nrows=len(l_im),ncols=3,figsize=(35,15)) 

    clim=(100,-100)

    for n,line in enumerate(l_im):
        [x,gen,y] = line
        x =  torch.squeeze(x).cpu().numpy()
        gen = torch.squeeze(gen).cpu().numpy()
        y = torch.squeeze(y).cpu().numpy()
        diff_x = gen-x
        diff_y = y-gen
        diff = y-x
        m = min(np.amin(diff_x),np.amin(diff_y),np.amin(diff))
        M = max(np.amax(diff_x),np.amax(diff_y),np.amax(diff))
        im = axes[n,0].imshow(diff_x,cmap=cmap,vmin=m,vmax=M)
        im = axes[n,2].imshow(diff_y,cmap=cmap,vmin=m,vmax=M)
        im = axes[n,1].imshow(diff,cmap=cmap,vmin=m,vmax=M)


    
    cols=['gen_sat','diff mod-sat','model-gen']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    fig.tight_layout()
    if save:
        plt.savefig(save_path+date+'diffs.png')
    plt.show()

