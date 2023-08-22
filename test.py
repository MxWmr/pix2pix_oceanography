import torch
from architecture import *
from datetime import datetime
import numpy as np 
from plot_fct import *
from data_load import *


save_path = '/usr/home/mwemaere/neuro/pix2pix/Save/'

data_path = '/usr/home/mwemaere/neuro/Data2/'


test_sattm1 = torch.load(data_path + 'test_ssh_sat.pt')[:-2,:,:,:88]
test_satt1 = torch.load(data_path + 'test_ssh_sat.pt')[1:-1,:,:,:88]
test_sattp1 = torch.load(data_path + 'test_ssh_sat.pt')[2:,:,:,:88]

test_ssttm1 = torch.load(data_path + 'test_sst.pt')[:-2,:,:,:88]
test_sstt1 = torch.load(data_path + 'test_sst.pt')[1:-1,:,:,:88]
test_ssttp1 = torch.load(data_path + 'test_sst.pt')[2:,:,:,:88]

test_mod = torch.load(data_path + 'test_ssh_mod.pt')[:,:,:,:88]

test_loader = ConcatData([test_sattm1,test_satt1,test_sattp1,test_ssttm1,test_sstt1,test_ssttp1,test_mod],shuffle=False)



G = Generator()

D = Discriminator()
   
   
device = 'cpu'
date = '08_21_14:29_'

D.load_state_dict(torch.load(save_path+date+'discr.pth'))
G.load_state_dict(torch.load(save_path+date+'gen.pth'))

crit = RMSELoss()

l_im,m_rmse,l_year,map_pred = test_gen(D,G,test_loader,device,crit,get_im=[25,186,245],year=True)

# plot_test(l_im,date,save_path)
# plot_diff(l_im,date,save_path)
# plot_year(l_year,date,save_path)
print_map(map_pred,date,save_path)