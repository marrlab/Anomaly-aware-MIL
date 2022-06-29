import torch
import torch.utils.data as data_utils
import torch.optim as optim
import numpy as np
import cv2
import sys
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from sklearn.covariance import MinCovDet
from model import MILModel
from DataLoader import DataLoader
from HDataLoader import HealthyLoader
from sklearn.mixture import GaussianMixture

'''
Parameters
'''
init_with = "new"
epochs = 300
lr = 0.0005

cuda = torch.cuda.is_available()
torch.manual_seed(int(sys.argv[1]) * int(sys.argv[3]))
if cuda:
    torch.cuda.manual_seed(int(sys.argv[1]) * int(sys.argv[3]))
loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
foldno = int(sys.argv[1]) # fold number (a value between 0 and 2)
model_name = sys.argv[2]   # name of model (a string name)
run_number = sys.argv[3]  # run number (avalue between 0 and 5)
healthy_data = []
precision = torch.eye(500)
center = torch.ones(500)
global_step = 1
'''
end of parameters
'''

cb = DataLoader(foldno, train=True)
n_train = cb.__len__()
train_loader = data_utils.DataLoader(cb, batch_size=1, shuffle=True, **loader_kwargs)

hcb = HealthyLoader(foldno, train=True)
n_h_train = hcb.__len__()
train_H_loader = data_utils.DataLoader(hcb, batch_size=1, shuffle=True, **loader_kwargs)

if init_with == "new":
    model = MILModel()
elif init_with == "latest":
    model = MILModel().load_latest()
if cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)


def trainMILSIL(epoch, global_step, Healthy=0, GMM=0):
    '''
    Training pipeline.
    
    Inputs:
    - epoch (epoch number)
    - global_step (itteration number embeded for tensorboard charts)
    - Healthy (flag: to specify when GMM should get updated)
    - GMM (contains fitted GMM information)
    
    Outputs:
    - GMM (only if Healthy=1)
    - train_mil_loss (training loss for the epoch)
    - global_step (itteration number embeded for tensorboard charts)
    '''
    global precision
    global center
    global healthy_data
    model.train()
    train_loss = 0.
    train_sil_loss = 0.
    train_mil_loss = 0.

    sil_ratio = torch.tensor(1 * (1 - .05) ** epoch)

    if Healthy == 0:
        for batch_idx, (feats, _, label) in enumerate(train_loader):
            feats = feats.float().squeeze()
            sil_target = label.repeat((feats.shape[0], 1))
            if cuda:
                feats = feats.cuda()
                label = label.cuda()
                sil_target = sil_target.cuda()
                sil_ratio = sil_ratio.cuda()

            feats, label, sil_target, sil_ratio = Variable(feats), Variable(label), Variable(sil_target), Variable(sil_ratio)
            optimizer.zero_grad()

            Y_prob, A, D, Y_sil_prob, M = model.forward_MILSIL(feats, GMM=GMM) # --> model update to use GMM
            mil_bce_loss, sil_loss = model.calculate_loss_MILSIL(Y_prob, Y_sil_prob, label, sil_target)
            loss = (1 - sil_ratio) * mil_bce_loss + sil_ratio * sil_loss

            writer.add_scalar('MIL_Loss/train', mil_bce_loss.data, global_step)
            writer.add_scalar('SIL_Loss/train', sil_loss.data, global_step)
            
            global_step += 1
            train_loss += loss.data
            train_mil_loss += mil_bce_loss.data
            train_sil_loss += sil_loss.data
            # backward pass
            loss.backward()
            # step
            optimizer.step()
        # calculate loss and error for epoch
        train_loss /= len(train_loader)
        train_sil_loss /= len(train_loader)
        train_mil_loss /= len(train_loader)
        print(
            'Epoch: {}, Loss: {:.4f}, MIL Loss: {:.4f}, SIL Loss: {:.4f}, SIL Ratio: {:.4f}'.format(epoch,
                                                                                                    train_loss.cpu().numpy(),
                                                                                                    train_mil_loss.cpu().numpy(),
                                                                                                    train_sil_loss.cpu().numpy(),
                                                                                                    sil_ratio.cpu().numpy())
        )

        return train_mil_loss, global_step
    
    # Healthy==1 -> GMM fitting on healthy data
    else:
        Hcells = []
        for batch_idx, (feats, _, label) in enumerate(train_H_loader):

            feats = feats.float().squeeze()
            sil_target = label.repeat((feats.shape[0], 1))

            if cuda:
                feats = feats.cuda()
                label = label.cuda()
                sil_target = sil_target.cuda()
                sil_ratio = sil_ratio.cuda()

            feats, label, sil_target, sil_ratio = Variable(feats), Variable(label), Variable(sil_target), Variable(
                sil_ratio)

            H = model.forward_MILSIL(feats, Healthy=1, GMM=0)
            Hcells.append((H.cpu().data.numpy()))

        Hcells = np.concatenate(Hcells, axis=0)
        GMM = GaussianMixture(n_components=1, covariance_type='full', random_state=0, init_params='random')
        GMM.fit(Hcells)
        print('GMM converged: ', GMM.converged_)

        return GMM



if __name__ == "__main__":
    
    '''
    Each 5 epoch only healthy data (HDataLoader) are passed through the network and GMM gets updated.
    For other cases mixed data containing healthy and disorderd cases (DataLoader) are passed through the MIL for training.
    If the loss is not decreased after 5 epochs early stop happens.
    '''
    
    print('Start Training')
    epoch = 0
    end = 0
    writer = SummaryWriter()
    global_step = 0
    while epoch < epochs:
        if epoch % 5 == 0:          
            GMM = trainMILSIL(epoch, global_step, Healthy=1)
        loss, global_step = trainMILSIL(epoch, global_step, Healthy=0, GMM=GMM)
        if loss < 0.0003:
            end -= - 1
        if end == 5:
            print("training finished, low loss reached.")
            break
        if epoch % 10 == 0:
            model.save(run_number + "-" + model_name + "-" + str(foldno))

        epoch -= - 1
    writer.close()
    model.save(run_number + "-" + model_name + "-" + "-" + str(foldno))
