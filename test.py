
# import modules
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from architecture import *
from data_load import prepare_loaders
from plot_fct import *



if torch.cuda.is_available():
    device = "cuda" 
else:
    raise('No GPU !')

date = '04_12_11:47_'

# load data
Path = 
data_path = ""
data = torch.load(data_path + "SSH_MERCATOR_1%3.pt")


# data -= data.min(1, keepdim=True)[0]
# data /= data.max(1, keepdim=True)[0]


data = torch.unsqueeze(data,1)


# prepare data
batch_size=32
train_loader,valid_loader,test_loader = prepare_loaders(data,batch_size=batch_size)

# create model
model = NN()


saved_path = Path+'Save/'+date+'model.pth'
model.load_state_dict(torch.load(saved_path))
model = model.to(device)



# Test
mean,std, l_im = model.test(test_loader,device, get_im=[15,58,245])


print(mean)
print(std)

with open('test_result.txt', 'a') as f:
    f.write('\n'+date+'\n')
    f.write(str(mean)+'\n')
    f.write(str(std)+'\n')

    f.close()

# plot some results

plot_test(l_im)