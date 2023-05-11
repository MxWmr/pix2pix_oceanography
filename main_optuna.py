import torch
from architecture import *
from datetime import datetime
import numpy as np 
from plot_fct import *
from data_load import *
import optuna
from optuna.trial import TrialState


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


def objective(trial):

    device = "cuda" 
    batch_size = trial.suggest_int("batch_size",6,128,log=True)
    train_loader = Dataset(200,95,data_path,'ssh_sat_','ssh_mod_',batch_size=batch_size)
    
    G = Generator()
    D = Discriminator()

    bce_crit = nn.BCELoss()
    l1_crit = nn.L1Loss()

    lr1 = trial.suggest_float("lr_gen",1e-7,1e-3,log=True)
    lr2 = trial.suggest_float("lr_discr",1e-7,1e-3,log=True)

    optim_gen = torch.optim.Adam(G.parameters(), lr=lr1,betas=(0.5,0.999))
    optim_discr = torch.optim.Adam(D.parameters(), lr=lr2,betas=(0.5,0.999))

    #n_epochs = trial.suggest_int("epoch",30,150)
    n_epochs = 40
    l1_lambda = trial.suggest_int("l1_lambda",1,150)

    train_gan(D,G,train_loader,n_epochs,device,bce_crit,l1_crit,optim_gen,optim_discr,l1_lambda=l1_lambda,verbose=False)

    crit = nn.L1Loss()
    device = "cpu"
    l_im,m_rmse = test_gen(D,G,test_loader,device,crit)

    return m_rmse


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)    

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    with open('test_result.txt', 'a') as f:
        f.write('\n'+date+'\n')
        f.write(str(accu)+'\n')
        for key, value in trial.params.items():
            f.write("    {}: {}  \n".format(key, value))

        f.close()



