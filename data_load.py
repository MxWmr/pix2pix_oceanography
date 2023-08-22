import torch
import math as mt
import numpy as np



class Dataset(torch.utils.data.Dataset):
    def __init__(self,l_files,n_files,path,file_name_sat,file_name_mod,file_name_sst,batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.file_name_sat = file_name_sat
        self.file_name_mod = file_name_mod
        self.file_name_sst = file_name_sst
        self.l_files = l_files
        self.path = path
        self.n_files = n_files

    def __len__(self):
        return (self.l_files//self.batch_size + 1)*self.n_files

    def __getitem__(self,i):
        len  = (self.l_files//self.batch_size + 1)*self.n_files
        
        i_f = i//(self.l_files//self.batch_size + 1)
        i_2 = i % (self.l_files//self.batch_size +1)

        if i >= len:
            raise IndexError()

        d_sat = torch.load(self.path+self.file_name_sat+str(i_f)+'.pt')[:,:,:,:88]
        d_mod = torch.load(self.path+self.file_name_mod+str(i_f)+'.pt')[:,:,:,:88]
        d_sst = torch.load(self.path+self.file_name_sst+str(i_f)+'.pt')[:,:,:,:88]

        if self.batch_size*(i_2+1) <= self.l_files-2:
            sattm1 = d_sat[i_2*self.batch_size:self.batch_size*(i_2+1)]
            satt = d_sat[i_2*self.batch_size+1:self.batch_size*(i_2+1)+1]
            sattp1 = d_sat[i_2*self.batch_size+2:self.batch_size*(i_2+1)+2]

            ssttm1 = d_sst[i_2*self.batch_size:self.batch_size*(i_2+1)]
            sstt = d_sst[i_2*self.batch_size+1:self.batch_size*(i_2+1)+1]
            ssttp1 = d_sst[i_2*self.batch_size+2:self.batch_size*(i_2+1)+2]

            mod = d_mod[i_2*self.batch_size+1:self.batch_size*(i_2+1)+1]
            
        else:
            sattm1 = d_sat[i_2*self.batch_size:-2]
            satt = d_sat[i_2*self.batch_size+1:-1]
            sattp1 = d_sat[i_2*self.batch_size+2:]

            ssttm1 = d_sat[i_2*self.batch_size:-2]
            sstt = d_sat[i_2*self.batch_size+1:-1]
            ssttp1 = d_sat[i_2*self.batch_size+2:]

            mod = d_mod[i_2*self.batch_size+1:-1]

        return tuple([sattm1,satt,sattp1,ssttm1,sstt,ssttp1,mod])    


class ConcatData(torch.utils.data.Dataset):
    def __init__(self,datasets,shuffle=False,batch_size=1):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size

        if shuffle:
            n = len(datasets[0])
            id_rd = torch.randperm(n)
            for d in self.datasets:
                d = d[list(id_rd)]

    def __getitem__(self,i):
        self.datasets[0][(i+1)*self.batch_size]
        return tuple(d[i*self.batch_size:(i+1)*self.batch_size] for d in self.datasets)


    def __len__(self):
        return min(int(len(d)/self.batch_size) for d in self.datasets)


