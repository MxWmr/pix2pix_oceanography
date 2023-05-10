import torch
from architecture import *
from datetime import datetime
import numpy as np 
from plot_fct import *
from data_load import *

if torch.cuda.is_available():
    device = "cuda" 
else:
    raise('No GPU !')

date = datetime.now().strftime("%m_%d_%H:%M_")

save_path = '/usr/home/mwemaere/neuro/pix2pix/Save/'


data_path = '/usr/home/mwemaere/neuro/Data2/'



train_loader = Dataset(200,95,data_path,'ssh_sat_','ssh_mod_',batch_size=8)


test_sat = torch.load(data_path + 'test_ssh_sat.pt')[:,:,:,:88]
test_mod = torch.load(data_path + 'test_ssh_mod.pt')[:,:,:,:88]
test_loader = ConcatData([test_sat,test_mod],shuffle=True)


G = Generator()

D = Discriminator()

if True:   #train
    bce_crit = nn.BCELoss()
    l1_crit = nn.L1Loss()

    optim_gen = torch.optim.Adam(G.parameters(), lr=1e-4,betas=(0.5,0.999))
    optim_discr = torch.optim.Adam(D.parameters(), lr=1e-4,betas=(0.5,0.999))

    n_epochs = 80

    train_gan(D,G,train_loader,n_epochs,device,bce_crit,l1_crit,optim_gen,optim_discr)


    torch.save(G.state_dict(), save_path+date+'gen.pth')
    torch.save(D.state_dict(), save_path+date+'discr.pth')


if False:    #load & test
    device = 'cpu'
    date = '05_09_16:12_'

    D.load_state_dict(torch.load(save_path+date+'discr.pth'))
    G.load_state_dict(torch.load(save_path+date+'gen.pth'))

    rmse = RMSELoss()
    l_im = test_gen(D,G,test_loader,device,rmse,get_im=True)
    plot_test(l_im)
