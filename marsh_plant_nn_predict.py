import numpy as np
import cv2

import torch
import torch.nn as nn
from marsh_plant_dataset import MarshPlant_Dataset

N_CLASSES = 7
THRESHOLD_SIG = 0.5
batch_size = 32
bShuffle = False
num_workers = 8


model_path = './modeling/saved_models/ResNet101_marsh_plants_20190415.torch'

model = torch.load(model_path)
model.eval()
sigfunc = nn.Sigmoid()


test_infile  = 'marsh_data_all_test.txt'
test_data  = MarshPlant_Dataset(test_infile)
data_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = bShuffle, num_workers = num_workers)


cpu = torch.device("cpu")
gpu = torch.device("cuda")


pred = np.empty((0,N_CLASSES), int)
ann  = np.empty((0,N_CLASSES), int)

with torch.no_grad():
    for it, batch in enumerate(data_loader):
        output = model(batch['X'].to(gpu)).to(cpu)

        sig = sigfunc(output)
        sig = sig.detach().numpy()
        this_pred = sig > THRESHOLD_SIG;
        print(this_pred.shape)
        pred = np.append(pred, this_pred.astype(int), axis = 0)
        #print(pred.shape)

        this_ann = batch['Y'].to(cpu).detach().numpy()  #take off gpu, detach from gradients
        ann = np.append(ann, this_ann.astype(int), axis = 0)


np.savetxt('pred.txt',pred, fmt='%i', delimiter='\t')
np.savetxt('ann.txt' ,ann , fmt='%i', delimiter='\t')
