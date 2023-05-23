import torch
from architecture import *
from datetime import datetime
import numpy as np 
from plot_fct import *
from data_load import *

if torch.cuda.is_available():
    device = "cuda:1" 
else:
    raise('No GPU !')

date = datetime.now().strftime("%m_%d_%H:%M_")

save_path = '/usr/home/mwemaere/neuro/pix2pix/Save/'


data_path = '/usr/home/mwemaere/neuro/Data2/'



train_loader = Dataset(100,270,data_path,'ssh_sat_','ssh_mod_',batch_size=8)


test_sat = torch.load(data_path + 'test_ssh_sat.pt')[:,:,:,:88]
test_mod = torch.load(data_path + 'test_ssh_mod.pt')[:,:,:,:88]

test_loader = ConcatData([test_sat,test_mod],shuffle=False)


valid_sat = torch.load(data_path + 'valid_ssh_sat.pt')[:,:,:,:88]
valid_mod = torch.load(data_path + 'valid_ssh_mod.pt')[:,:,:,:88]

valid_loader = ConcatData([valid_sat,valid_mod],shuffle=False)


G = Generator()

D = Discriminator()


bce_crit = nn.BCELoss()
l1_crit = nn.MSELoss()

optim_gen = torch.optim.Adam(G.parameters(), lr=1e-4,betas=(0.5,0.999))
optim_discr = torch.optim.Adam(D.parameters(), lr=1e-5,betas=(0.5,0.999))

n_epochs = 150
l1_lambda = 0.3

train_gan(D,G,train_loader,valid_loader,n_epochs,device,bce_crit,l1_crit,optim_gen,optim_discr,discr_cheat=2,l1_lambda=l1_lambda,valid=True)


torch.save(G.state_dict(), save_path+date+'gen.pth')
torch.save(D.state_dict(), save_path+date+'discr.pth')


