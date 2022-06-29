'''
This part of code is written due to the data structure that our red blood cell data is saved in our server.
To adapt this code to your applicaiton you need to re write this file. 
In this case, you will need to override __len__ and __getitem__ functions to be able to use pytorch Dataloader function. 
'''

import numpy as np
import os
import pickle
import gzip
import cv2
from torch.utils.data import Dataset

m = 7
datafeaturesfolder = "/storage/groups/qscd01/workspace/ario.sadafi/MIL-RBC-DS-MIXED/"


class HealthyLoader(Dataset):

    def __init__(self, foldnumber, train=True, H=0):
        self.train = train
        self._load_classes()
        self.patientlist = self._load_folds(foldnumber)
        self.current_fold = foldnumber

        dataset_file_name = ("train" if train else "test") + "-" + str(foldnumber) + ".pkl"

        if not os.path.exists(os.path.join(datafeaturesfolder,dataset_file_name)):
            sample_files = [x for x in os.listdir(os.path.join(datafeaturesfolder, "all"))
                      for patient in self.patientlist
                      if x.split("_")[0] + "_" + x.split("_")[1] == patient]

            self.sample_list, self.data_list = self._analyze_sample_files(sample_files)

            with gzip.open(os.path.join(datafeaturesfolder,dataset_file_name), "wb") as f:
                pickle.dump([self.sample_list, self.data_list], f)
        else:
            print("loading healthy dataset...")
            with gzip.open(os.path.join(datafeaturesfolder,dataset_file_name), "rb") as f:
                self.sample_list, self.data_list = pickle.load(f)
        print("loading healthy is done")
        self.H_data = []
        for i in range(len(self.data_list)):
            if self.data_list[self.sample_list[i]]["label"] == 'Control':
                self.H_data.append(self.data_list[self.sample_list[i]])
          
        

    def __len__(self):
        return len(self.H_data)


    def __getitem__(self, index):
        data = self.H_data[index]
        label = self.classes.index(data["label"])

        if self.train:
            return data["features"], data["scimg"], label
        else:
            return data["features"], data["scimg"], label, self.sample_list[index].split('.')[0]


    def _analyze_sample_files(self, sample_files):
        sample_list = []
        data_list = {}
        for sample_file in sample_files:
            with gzip.open(os.path.join(datafeaturesfolder, "all", sample_file)) as f:
                data = pickle.load(f)
            print(sample_file)
            
            features = None
            scCells = None
            for d in data["data"]:
                feats = d["feats"]
                if feats is [] or feats.size == 0:
                    continue

                feats = np.rollaxis(feats, 3, 1)
                feats = 2. * (feats - np.min(feats)) / np.ptp(feats) - 1
                img = d["image"]
                m = 15
                cells = None
                for roi in d["rois"]:
                    h, w, _ = img.shape
                    roi = [max(0, roi[0] - m), max(0, roi[1] - m), min(h, roi[2] + m), min(w, roi[3] + m)]
                    cell = img[roi[0]:roi[2], roi[1]:roi[3]]
                    cell = cv2.resize(cell, (64, 64))
                    cell = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
                    cell = cv2.normalize(cell, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    cell = np.expand_dims(cell, axis=0)
                    if cells is None:
                        cells = cell
                    else:
                        cells = np.append(cells, cell, axis=0)

                if features is None:
                    features = feats
                    scCells = cells
                else:
                    features = np.append(features, feats, axis=0)
                    scCells = np.append(scCells, cells, axis=0)
            if features is not None:
                data_list[sample_file] = {
                    "label": data["meta"]["label"],
                    "features": features,
                    "scimg": scCells
                }
                sample_list.append(sample_file)
        return sample_list, data_list

    def _load_classes(self):
        self.classes = []
        with open(datafeaturesfolder + "classes.txt") as f:
            clsdata = f.readlines()
            for cls in clsdata:
                self.classes.append(cls.strip("\n"))

    def _load_folds(self, foldnumber):

        with open(datafeaturesfolder + "folds.pkl", "rb") as f:
            folds = pickle.load(f)

        train_list, test_list = folds[foldnumber]
        if self.train:
            return train_list
        else:
            return test_list

    def _get_classes(self):
        return self.classes

    def _correct_label_list(self, label_list):
        newlist = []
        for label in label_list:
            newlist.append(self.classes.index(label))

        return newlist

