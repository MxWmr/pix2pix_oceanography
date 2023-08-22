import torch
from torch import nn
import torch.functional as F
import math as mt
import numpy as np
import optuna
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class conv_block(nn.Module):
    def __init__(self, in_c,out_c,kernel_size=3, padding='same',activ=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_c,momentum=0.1,eps=1e-5)         
        self.factiv = nn.SELU()
        self.activ=activ
    def forward(self, inputs):
        x = self.conv1(inputs)    
        x = self.conv2(x)
        x = self.factiv(x)  
        if self.activ:
            x = self.factiv(x)  
            x = self.bn(x)
        return x



class encoder_block(nn.Module):
    def __init__(self, in_c, out_c,kernel_size=5, padding=2,first_layer=False):
        super().__init__()
        self.conv = conv_block(in_c, out_c,kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p



class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        
        self.w_g = nn.Sequential(
        nn.Conv2d(f_g, f_g,kernel_size=1, stride=1,padding='same', bias=True),
        nn.Conv2d(f_g, f_int,kernel_size=1, stride=1,padding='same', bias=True),
        nn.BatchNorm2d(f_int)
        )
        
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_l,kernel_size=1, stride=1,padding='same', bias=True),
            nn.Conv2d(f_l, f_int,kernel_size=1, stride=1,padding='same', bias=True),                    
            nn.BatchNorm2d(f_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1,kernel_size=1, stride=1,padding='same',bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.factiv = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.factiv(g1+x1)
        psi = self.psi(psi)
        
        return psi*x



class Generator(nn.Module):


    def __init__(self,n_chan=64):
        super(Generator,self).__init__()

        # downsample
        self.conv0 = nn.Conv2d(6, n_chan, kernel_size=3, padding='same')
        self.conv1 = conv_block(n_chan,n_chan)
        self.pool = nn.MaxPool2d(2,padding=0)

        self.conv2 = conv_block(n_chan,n_chan)
        self.conv3 = conv_block(n_chan,n_chan)

        # bottleneck
        self.conv4 = conv_block(n_chan,n_chan)
        self.conv5 = conv_block(n_chan,n_chan)
        self.conv51 = conv_block(n_chan,n_chan)
        self.conv52 = conv_block(n_chan,n_chan)
        self.conv53 = conv_block(n_chan,n_chan)
        self.conv54 = conv_block(n_chan,n_chan)

        


        # upsample
        self.up = nn.Upsample(scale_factor=2,mode='bicubic')
        self.att1 = AttentionBlock(f_g=n_chan, f_l=n_chan, f_int=n_chan//2)

        self.convout1 = conv_block(n_chan*2,1,activ=False)

        self.conv6 = conv_block(n_chan*2,n_chan)
        self.conv7 = conv_block(n_chan,n_chan)

        self.att2 = AttentionBlock(f_g=n_chan, f_l=n_chan, f_int=n_chan//2)
        self.conv8 = conv_block(n_chan*2,n_chan)
        self.conv9 = conv_block(n_chan,n_chan)    

        self.convout2 = conv_block(n_chan,1,activ=False)

        #out
        self.conv10 = conv_block(n_chan,8)
        self.conv11 = conv_block(8,1,activ=False)     




    def forward(self,x):


        x = self.conv0(x)
        x = self.conv1(x)
        x1 = self.pool(x)

        x = self.conv2(x1)
        x = self.conv3(x)
        x2 = self.pool(x)

        x = self.conv4(x2)
        x = self.conv5(x)
        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.conv54(x)

        #x2 = self.att2(g=x,x=x2)
        x = torch.cat([x,x2],axis=1)

        xout1 = self.convout1(x)
        xout1up = self.up(xout1)


        x = self.up(x)
        x = self.conv6(x)
        x = self.conv7(x)
    
        #x1 = self.att2(g=x,x=x1)
        x = torch.cat([x,x1],axis=1)

        x = self.conv8(x)
        x = self.conv9(x)

        
        xout2 = self.convout2(x)
        xout2 = xout2+xout1up
        xout2up = self.up(xout2)

        x = self.up(x)
        x = self.conv10(x)
        xout3 = self.conv11(x)+xout2up

        return (xout1,xout2,xout3)




class Discriminator(nn.Module):
 def __init__(self,n_chan=32):#input : 72x88
   super(Discriminator,self).__init__()
   self.conv01 = conv_block(2,16)
   self.conv02 = conv_block(16,32)
   self.conv1 = encoder_block(32,n_chan*2) # 36x44
   self.conv2 = encoder_block(n_chan*2,n_chan*4) # 18x22
   self.conv3 = encoder_block(n_chan*4,n_chan*2) # 9x11
   self.conv03 = conv_block(n_chan*2,n_chan)
   self.conv04 = conv_block(n_chan,1)

   self.sigmoid = nn.Sigmoid()



 def forward(self, x, y):

    O = torch.cat([x,y],dim=1)
    O = self.conv01(O)
    O = self.conv02(O)
    s,O = self.conv1(O)
    s,O = self.conv2(O)
    s,O = self.conv3(O)
    O = self.conv03(O)
    O = self.conv04(O)
    return self.sigmoid(O)




def train_gan(D,G,train_loader,valid_loader,n_epochs,device,bce_crit,l1_crit,optim_gen,optim_discr,scheduler_gen,scheduler_discr,discr_cheat=1,gen_cheat=1,l1_lambda=10,valid=False,verbose=True):
    """
    Inputs
    D: discriminator model (torch.nn object)
    G: generator model (torch.nn object)
    train_loader: torch dataloader for training data (sat,mod) format
    valid_loader: torch dataloader for validation data (sat,mod) format
    n_epochs: number of epochs fro trainintg (int)
    device: device on which the training goes ("cuda:0","cuda:1","cpu")
    bce_crit: criterion of binary cross entropy (torch.optim object)
    l1_crit:  L1 criterion (torch.nn object)
    optim_gen: optimizator for the generator (torch.optim object)
    optim_discr: optimizator for the discriminator (torch.optim object)
    discr_cheat: number of training of the discriminator for a point of data (int)
    gen_cheat: number of training of the generator for a point of data (int)
    l1_lambda: coefficient of mixing L1 loss with BCE loss for generator training BCELoss+l1_lambda*L1Loss (float) 
    valid: activation of validation (boolean)
    verbose: activation of tensorboard (boolean)

    """
    D.to(device)
    G.to(device)

    if verbose:
        tbw = SummaryWriter()

    if valid:
        rmse= RMSELoss()
        [mean_mod,std_mod] = torch.load('/usr/home/mwemaere/neuro/Data2/mean_std_mod.pt')
    
    pool1 = torch.nn.AvgPool2d(2,stride=(2,2))
    pool2 = torch.nn.AvgPool2d(4,stride=(4,4))  


    for ep in range(n_epochs):
        print('epoch: {}'.format(ep+1))
        ite=0
        l_lossd = []
        l_lossg = []
        l_loss_l1 = []
        l_loss_bce = []
        l_mean_d = []
        l_mean_g = []

        for i,(sattm1,satt,sattp1,ssttm1,sstt,ssttp1,mod) in enumerate(train_loader):

            sattm1,satt,sattp1,ssttm1,sstt,ssttp1,mod = sattm1.to(device),satt.to(device),sattp1.to(device),ssttm1.to(device),sstt.to(device),ssttp1.to(device),mod.to(device)

            b_size = satt.shape[0]
            if b_size == 0:
                continue

            x = torch.cat([sattm1,satt,sattp1,ssttm1,sstt,ssttp1],axis=1)

            sat_class = torch.zeros(b_size,1,9,11).to(device)
            mod_class = torch.ones(b_size,1,9,11).to(device)

            ## Train discriminator
            for k in range(discr_cheat):
                D.zero_grad()

                pred_true = D(mod,satt)

                real_gan_loss = bce_crit(pred_true,mod_class)

                fake = G(x)[2]

                pred_fake = D(fake.detach(),satt)

                fake_gan_loss = bce_crit(pred_fake,sat_class)
                
                D_loss = real_gan_loss + fake_gan_loss

                l_lossd.append(D_loss.item())

                if mt.isnan(D_loss.item()):
                    print(b_size)
                    print(sat_class)

                D_loss.backward()
                optim_discr.step()


            ## Train generator
            for k in range(gen_cheat):
                fake = G(x)[2]

                G.zero_grad()

                pred_fake  = D(fake,satt)

                fake_gan_loss = bce_crit(pred_fake+1e-10,mod_class)

                l1_loss = l1_crit(fake,mod)

                G_loss = fake_gan_loss + l1_lambda*l1_loss

                l_lossg.append(G_loss.item())
                l_loss_bce.append(fake_gan_loss.item())
                l_loss_l1.append(l1_loss.item())

                G_loss.backward()
                optim_gen.step()

        
        if valid:
            l_valid = []

            with torch.no_grad():
                correct,total = 0,0

                for i,(sattm1,satt,sattp1,ssttm1,sstt,ssttp1,mod) in enumerate(valid_loader):

                    sattm1,satt,sattp1,ssttm1,sstt,ssttp1,mod = sattm1.to(device),satt.to(device),sattp1.to(device),ssttm1.to(device),sstt.to(device),ssttp1.to(device),mod.to(device)

                    b_size = satt.shape[0]
                    if b_size == 0:
                        continue

                    x = torch.cat([sattm1,satt,sattp1,ssttm1,sstt,ssttp1],axis=1)

                    y_pred = G(x)[2]

                    l_valid.append(rmse(y_pred*std_mod+mean_mod,mod*std_mod+mean_mod).item())

                    pred = D(y_pred,satt)
                    if torch.mean(pred).item()<0.5:
                        correct+=1
                    total+=1

                valid_mean = np.array(l_valid).mean()
                tbw.add_scalar("validation RMSE (m)",valid_mean,ep)
                tbw.add_scalar("validation discr score (%)",correct/total*100,ep)
                print('RMSE: {}m'.format(valid_mean))
                print('Discr score: {}%'.format(correct/total*100))




        scheduler_gen.step(valid_mean)
        scheduler_discr.step(valid_mean)




        if verbose:
            mean_lossd = np.array(l_lossd).mean()
            l_mean_d.append(mean_lossd)
            tbw.add_scalar("loss discriminator",mean_lossd,ep)

            mean_lossg = np.array(l_lossg).mean()
            tbw.add_scalar("loss generator",mean_lossg,ep)

            mean_loss_bce = np.array(l_loss_bce).mean()
            l_mean_g.append(mean_loss_bce)
            tbw.add_scalar("loss gen BCE",mean_loss_bce,ep)

            mean_loss_l1 = np.array(l_loss_l1).mean()
            tbw.add_scalar("loss gen L1",mean_loss_l1,ep)
        


    if verbose:
        tbw.close()


def test_gen(D,G,test_loader,device,crit,get_im=False,year=False):
    D,G = D.to(device),G.to(device)
    l_im = []
    l_rmse = []
    l_rmse2 = []
    l_year = []
    l_month = []
    map_pred = torch.zeros(1,1,9,11)

    [mean_mod,std_mod] = torch.load('/usr/home/mwemaere/neuro/Data2/mean_std_mod.pt')
    #[mean_sat,std_sat] = torch.load('/usr/home/mwemaere/neuro/Data2/mean_std_sat.pt')

    correct = 0
    total = 0

    with torch.no_grad():

        for i,(sattm1,satt,sattp1,ssttm1,sstt,ssttp1,mod) in enumerate(test_loader):

            sattm1,satt,sattp1,ssttm1,sstt,ssttp1,mod = sattm1.to(device),satt.to(device),sattp1.to(device),ssttm1.to(device),sstt.to(device),ssttp1.to(device),mod.to(device)

            b_size = satt.shape[0]
            if b_size == 0:
                continue

            x = torch.cat([sattm1,satt,sattp1,ssttm1,sstt,ssttp1],axis=1)

            gen = G(x)[2]
            rm = crit(gen*std_mod+mean_mod,mod*std_mod+mean_mod)
            l_rmse.append(rm)
            l_rmse2.append(crit(gen*std_mod+mean_mod,satt*std_mod+mean_mod))

            pred = D(gen,satt)

            map_pred+=pred

            if torch.mean(pred)<0.5:
                correct+=1
            total+=1

            if year:
                l_month.append(rm)

            if year and i+1 in [31,59,90,120,151,181,212,243,273,304,334,364]:
                l_year.append(np.array(l_month).mean())
                l_month = []

            #if get_im and i in [10,145,240]:
                #l_im.append([x,gen,y])

                

    d_perf = correct/total*100 
    map_pred=(torch.ones(1,1,9,11)-map_pred/total)*100
    m_rmse = np.array(l_rmse).mean()
    m_rmse2 = np.array(l_rmse2).mean()

    print('discriminator accuracy: {}%'.format(d_perf))
    print('mean RMSE with target on the test set: {:.3f} m'.format(m_rmse))
    print('mean RMSE with input on the test set: {:.3f} m'.format(m_rmse2))
    return l_im,m_rmse,l_year,map_pred





class RMSELoss(torch.nn.Module):
    def __init__(self,coeff=1):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.coeff = coeff
        
    def forward(self,yhat,y):
        return self.coeff*torch.sqrt(self.mse(yhat,y))






def custom_scheduler(epoch):
    if epoch<20:
        return 1
    elif epoch<80:
        return mt.exp(-0.05)
    elif epoch <100: 
        return mt.exp(-0.15)
    else:
        return mt.exp(-0.5)








































