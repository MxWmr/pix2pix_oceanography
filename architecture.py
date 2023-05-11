import torch
from torch import nn
import torch.functional as F
import math as mt
import numpy as np
import optuna
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def cnn_block(in_channels,out_channels,kernel_size,stride=1,padding='same', first_layer = False):

   if first_layer:
       return nn.Sequential(
       nn.Conv2d(in_channels,in_channels,kernel_size,stride=stride,padding=padding),
       nn.Conv2d(in_channels,out_channels,3,padding=padding))
   else:
       return nn.Sequential(
           nn.Conv2d(in_channels,in_channels,kernel_size,stride=stride,padding=padding),
           nn.Conv2d(in_channels,out_channels,3,padding=padding),
           nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5))

def tcnn_block(in_channels,out_channels,kernel_size,stride=1,padding='same',output_padding=0, first_layer = False):
    if first_layer:
       return nn.Sequential(
        nn.ConvTranspose2d(in_channels,in_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding),
        nn.ConvTranspose2d(in_channels,out_channels,3,stride=1,padding=padding,output_padding=output_padding))

    else:
       return nn.Sequential(
           nn.ConvTranspose2d(in_channels,in_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding),
           nn.ConvTranspose2d(in_channels,out_channels,3,stride=1,padding=padding,output_padding=output_padding),
           nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5))


class Generator(nn.Module):


    def __init__(self,n_chan=16):
        super(Generator,self).__init__()
        self.e1 = cnn_block(1,n_chan,4,2,1, first_layer = True)
        self.e2 = cnn_block(n_chan,n_chan*2,4,2,1)
        self.e3 = cnn_block(n_chan*2,n_chan*4,4,2,1, first_layer=True)


        self.d1 = tcnn_block(n_chan*4,n_chan*2,4,2,1)
        self.d2 = tcnn_block(n_chan*2*2,n_chan,4,2,1)
        self.d3 = tcnn_block(n_chan*2,1,4,2,1, first_layer=True)

        self.sig = nn.Sigmoid()


    def forward(self,x):
        x0 = self.e1(x)
        x1 = self.e2(nn.SiLU()(x0))
        x2 = self.e3(nn.SiLU()(x1))

        x3 = torch.cat([self.d1(nn.SiLU()(x2)),x1],1)
        x4 = torch.cat([self.d2(nn.SiLU()(x3)),x0],1)
        x5 = self.d3(nn.SiLU()(x4))

        return self.sig(x5)



class Discriminator(nn.Module):
 def __init__(self,n_chan=32):#input : 72x88
   super(Discriminator,self).__init__()
   self.conv1 = cnn_block(1*2,n_chan,4,2,1, first_layer=True) # 36x44
   self.conv2 = cnn_block(n_chan,n_chan*2,4,2,1)# 18x22
   self.conv3 = cnn_block(n_chan*2,1,4,2,1, first_layer=True) # 9x11
   self.sigmoid = nn.Sigmoid()



 def forward(self, x, y):

    O = torch.cat([x,y],dim=1)
    O = nn.SiLU()(self.conv1(O))
    O = nn.SiLU()(self.conv2(O))
    O = self.conv3(O)

    return self.sigmoid(O)




def train_gan(D,G,train_loader,n_epochs,device,bce_crit,l1_crit,optim_gen,optim_discr,l1_lambda=10,verbose=True):

    D.to(device)
    G.to(device)

    if verbose:
        tbw = SummaryWriter()

    
    for ep in range(n_epochs):
        print('epoch: {}'.format(ep+1))
        ite=0
        l_lossd = []
        l_lossg = []

        for i,(x,y) in tqdm(enumerate(train_loader)):
            x,y = x.to(device),y.to(device)

            b_size = x.shape[0]

            if b_size == 0:
                continue
            sat_class = torch.zeros(b_size,1,9,11).to(device)
            mod_class = torch.ones(b_size,1,9,11).to(device)

            ## Train discriminator

            D.zero_grad()

            pred_true = D(y,x)

            real_gan_loss = bce_crit(pred_true,mod_class)

            fake = G(x)

            pred_fake = D(fake.detach(),x)

            fake_gan_loss = bce_crit(pred_fake,sat_class)
            
            D_loss = real_gan_loss + fake_gan_loss

            l_lossd.append(D_loss.item())

            if mt.isnan(D_loss.item()):
                print(b_size)
                print(sat_class)

            D_loss.backward()
            optim_discr.step()


            ## Train generator

            G.zero_grad()

            pred_fake  = D(fake,x)

            fake_gan_loss = bce_crit(pred_fake+1e-10,mod_class)

            l1_loss = l1_crit(fake,y)

            G_loss = fake_gan_loss + l1_lambda*l1_loss

            l_lossg.append(G_loss.item())

            G_loss.backward()
            optim_gen.step()

            if verbose and ep%4 == 0 and i==10:
                tbw.add_image("input",x[-4])
                tbw.add_image("generated",fake[-4])
                tbw.add_image("target",y[-4])


        if verbose:
            mean_lossd = np.array(l_lossd).mean()
            tbw.add_scalar("loss discriminator",mean_lossd,ep)

            mean_lossg = np.array(l_lossg).mean()
            tbw.add_scalar("loss generator",mean_lossg,ep)



    if verbose:
        tbw.close()


def test_gen(D,G,test_loader,device,crit,get_im=False):
    D,G = D.to(device),G.to(device)
    l_im = []
    l_rmse = []

    correct = 0
    total = 0

    with torch.no_grad():

        for i,(x,y) in enumerate(test_loader):

            x,y = x.to(device),y.to(device)

            b_size = x.shape[0]
            if b_size == 0:
                continue

            gen = G(x)

            l_rmse.append(crit(gen,y))

            pred = D(gen,x)

            if torch.mean(pred)<0.5:
                correct+=1
            total+=1

            if get_im and i in [10,145,240]:
                l_im.append([x,gen,y])

    d_perf = correct/total*100 
    m_rmse = np.array(l_rmse).mean()

    print('discriminator accuracy: {}%'.format(d_perf))
    print('mean RMSE on the test set: {}'.format(m_rmse))

    return l_im,m_rmse








class Generator2(nn.Module):


    def __init__(self,n_chan=16):
        super(Generator,self).__init__()
        self.e1 = cnn_block(1,n_chan,4,2,1, first_layer = True)
        self.e2 = cnn_block(n_chan,n_chan*2,4,2,1)
        self.e3 = cnn_block(n_chan*2,n_chan*4,4,2,1, first_layer=True)


        self.d1 = tcnn_block(n_chan*4,n_chan*2,4,2,1)
        self.d2 = tcnn_block(n_chan*2*2,n_chan,4,2,1)
        self.d3 = tcnn_block(n_chan*2,1,4,2,1, first_layer=True)

        self.sig = nn.Sigmoid()


    def forward(self,x):
        x0 = self.e1(x)
        x1 = self.e2(nn.SiLU()(x0))
        x2 = self.e3(nn.SiLU()(x1))

        x3 = torch.cat([self.d1(nn.SiLU()(x2)),x1],1)
        x4 = torch.cat([self.d2(nn.SiLU()(x3)),x0],1)
        x5 = self.d3(nn.SiLU()(x4))

        return self.sig(x5)









































