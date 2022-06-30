import torch
import torch.utils.data as data_utils
from torch.autograd import Variable
from tqdm import tqdm
import os
import sys
import numpy as np
import pickle
import gzip
from scipy.spatial import distance

sys.path.append("..")
from DataLoader import DataLoader
from HDataLoader import HealthyLoader
from sklearn.mixture import GaussianMixture

Classes = []
datafeaturesfolder = "/storage/groups/qscd01/workspace/ario.sadafi/MIL-RBC-DS-MIXED/"

with open(datafeaturesfolder + "classes.txt") as f:
    clsdata = f.readlines()
    for cls in clsdata:
        Classes.append(cls.strip("\n"))

cuda = torch.cuda.is_available()
torch.manual_seed(1)
if cuda:
    torch.cuda.manual_seed(1)


Model_name = "MILSIL"
for fold in range(3):

    # loading data
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    test_loader = data_utils.DataLoader(DataLoader(foldnumber=fold, train=False),
                                        batch_size=1, shuffle=False, **loader_kwargs)
    # loading healthy data
    H_loader = data_utils.DataLoader(HealthyLoader(foldnumber=fold, train=True),
                                     batch_size=1, shuffle=False, **loader_kwargs) 

    model_runs = []
    for run in range(5):
        # loading model
        model_name = [x for x in [x for x in os.listdir("Model") if not x.startswith(".") and x.split("-")[1] == Model_name]
                      if x.split("-")[0] == str(run) and x.split("-")[3] == str(fold)][-1]
        print(model_name)
        model = torch.load(os.path.join("Model", model_name), map_location=torch.device('cpu'))
        if cuda:
            model.cuda()
        model.eval()

        # fit GMM
        Hcells = []
        for batch_idx, (feats, scimg, bag_labels, sample_name) in tqdm(enumerate(H_loader)):
            if cuda:
                feats = feats.cuda()
            feats = Variable(feats.squeeze())

            H = model.forward_MILSIL(feats, Helthy=1, GMM=0)
            Hcells.append((H.cpu().data.numpy()))
        Hcells = np.concatenate(Hcells, axis=0)
        GMM = GaussianMixture(n_components=1, covariance_type='full', random_state=0, init_params='random')
        GMM.fit(Hcells)
        print('GMM converged: ', GMM.converged_)

        bag_labels = []

        pred = []
        gt = []

        for batch_idx, (feats, scimg, bag_labels, sample_name) in tqdm(enumerate(test_loader)):
            if cuda:
                feats = feats.cuda()
            feats = Variable(feats.squeeze())

            out = model.forward_MILSIL(feats, GMM=GMM)
            pred.append(out[0].cpu().data.numpy())
            gt.append(bag_labels.cpu().data.int().numpy()[0])

            print(str(batch_idx) + "/" + str(len(test_loader)))

        model_runs.append([pred, gt])

    print("saving to file....")
    with gzip.open("evals/" + Model_name + "-test-" + str(fold) + ".pkl", "wb") as f:
        pickle.dump(model_runs, f)
    print('Fold:', fold, ' is done')

print("test is finished")
