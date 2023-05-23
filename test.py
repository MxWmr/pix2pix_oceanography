 import torch
from architecture import *
from datetime import datetime
import numpy as np 
from plot_fct import *
from data_load import *


save_path = '/usr/home/mwemaere/neuro/pix2pix/Save/'

data_path = '/usr/home/mwemaere/neuro/Data2/'


test_sat = torch.load(data_path + 'test_ssh_sat.pt')[:,:,:,:88]
test_mod = torch.load(data_path + 'test_ssh_mod.pt')[:,:,:,:88]

test_loader = ConcatData([test_sat,test_mod],shuffle=False)


G = Generator()

D = Discriminator()
   
   
device = 'cpu'
date = '05_22_16:31_'

D.load_state_dict(torch.load(save_path+date+'discr.pth'))
G.load_state_dict(torch.load(save_path+date+'gen.pth'))

crit = RMSELoss()

l_im,m_rmse = test_gen(D,G,test_loader,device,crit,get_im=[25,186,245])

plot_test(l_im,date,save_path)
plot_diff(l_im,date,save_path)