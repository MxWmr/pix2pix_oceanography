import torch
from architecture import *
from datetime import datetime
import numpy as np 
from plot_fct import *
from data_load import *

if torch.cuda.is_available():
    device = "cuda:0" 
else:
    raise('No GPU !')

date = datetime.now().strftime("%m_%d_%H:%M_")

save_path = '/usr/home/mwemaere/neuro/pix2pix/Save/'


data_path = '/usr/home/mwemaere/neuro/Data2/'



train_loader = Dataset(98,96,data_path,'ssh_sat_','ssh_mod_','sst_',batch_size=16)


valid_sattm1 = torch.load(data_path + 'valid_ssh_sat.pt')[:-2,:,:,:88]
valid_satt1 = torch.load(data_path + 'valid_ssh_sat.pt')[1:-1,:,:,:88]
valid_sattp1 = torch.load(data_path + 'valid_ssh_sat.pt')[2:,:,:,:88]

valid_ssttm1 = torch.load(data_path + 'valid_sst.pt')[:-2,:,:,:88]
valid_sstt1 = torch.load(data_path + 'valid_sst.pt')[1:-1,:,:,:88]
valid_ssttp1 = torch.load(data_path + 'valid_sst.pt')[2:,:,:,:88]

valid_mod = torch.load(data_path + 'valid_ssh_mod.pt')[:,:,:,:88]

valid_loader = ConcatData([valid_sattm1,valid_satt1,valid_sattp1,valid_ssttm1,valid_sstt1,valid_ssttp1,valid_mod],shuffle=False)



G = Generator()

D = Discriminator()




bce_crit = nn.BCELoss()
l1_crit = nn.L1Loss()

optim_gen = torch.optim.Adam(G.parameters(), lr=5e-3)
optim_discr = torch.optim.Adam(D.parameters(), lr=5e-3)
#scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim_gen, lr_lambda=custom_scheduler)
scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_gen,factor=0.1,patience=5)
scheduler_discr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_gen,mode='min',factor=0.1,patience=5)

n_epochs = 200
l1_lambda = 1

train_gan(D,G,train_loader,valid_loader,n_epochs,device,bce_crit,l1_crit,optim_gen,optim_discr,scheduler_gen,scheduler_discr,discr_cheat=1,l1_lambda=l1_lambda,valid=True)


torch.save(G.state_dict(), save_path+date+'gen.pth')
torch.save(D.state_dict(), save_path+date+'discr.pth')


