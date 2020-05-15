# evaluate performance of CNN classifier by checking activations and what is confused with what. 
import numpy as np
import cv2

import torch
import torch.nn as nn
from marsh_plant_dataset import MarshPlant_Dataset


outfile = 'performance_evaluation.txt'
N_CLASSES = 7 #9
THRESHOLD_SIG = 0.5
batch_size = 32
bShuffle = False
num_workers = 8

Pennings_Classes = ['Salicornia','Spartina','Limonium','Borrichia','Batis','Juncus','None']
#Pennings_Classes = [ 'Spartina','Juncus', 'Salicornia','Batis','Borrichia','Limonium','Soil' ,'other','Unknown' ]
# load and run CNN on test dataset


# evaluate performance of CNN classifier
import numpy as np
import cv2

import torch
import torch.nn as nn
from marsh_plant_dataset import MarshPlant_Dataset


outfile = 'performance_evaluation.txt'
N_CLASSES = 7 #9
THRESHOLD_SIG = 0.5
batch_size = 32
bShuffle = False
num_workers = 8

Pennings_Classes = ['Salicornia','Spartina','Limonium','Borrichia','Batis','Juncus','None']
#Pennings_Classes = [ 'Spartina','Juncus', 'Salicornia','Batis','Borrichia','Limonium','Soil' ,'other','Unknown' ]
# load and run CNN on test dataset

model_path = './modeling/saved_models/ResNet101_marsh_plants_juncus_10192019.torch'

model = torch.load(model_path)
model.eval()
sigfunc = nn.Sigmoid()

#test_infile  = 'marsh_percent_cover_test.txt'
test_infile = 'marsh_data_all_test.txt'

test_data  = MarshPlant_Dataset(test_infile)
data_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = bShuffle, num_workers = num_workers)


cpu = torch.device("cpu")
gpu = torch.device("cuda")


pred = np.empty((0,N_CLASSES), int)
ann  = np.empty((0,N_CLASSES), int)
sigs = np.empty((0,N_CLASSES), int)

with torch.no_grad():
    for it, batch in enumerate(data_loader):
        output = model(batch['X'].to(gpu)).to(cpu)

        sig = sigfunc(output)
        sig = sig.detach().numpy()
        sigs= np.append(sigs, sig, axis = 0)
        this_pred = sig > THRESHOLD_SIG;
        #print(this_pred.shape)
        pred = np.append(pred, this_pred.astype(int), axis = 0)
        #print("Predictions'sigmoid in Batch size 32")
        #print(sig)
        this_ann = batch['Y'].to(cpu).detach().numpy()  #take off gpu, detach from gradients
        #print("Labels in batch size 32")
        #print(this_ann) 
        ann = np.append(ann, this_ann.astype(int), axis = 0)


#np.savetxt('pred.txt',pred, fmt='%i', delimiter='\t')
#np.savetxt('ann.txt' ,ann , fmt='%i', delimiter='\t')

# evaluate performance on test dataset
#n_samples, c = ann.shape

ptot = np.sum(ann,axis=0)
ntot = np.sum(np.logical_not(ann), axis=0)

tp = np.logical_and(ann,pred)  #true positivies
tn = np.logical_and(np.logical_not(ann), np.logical_not(pred)) #true negatives
fp = np.logical_and(np.logical_not(ann), pred)  #false positivies
fn = np.logical_and(ann, np.logical_not(pred)) #false negatives

#totals
tpt = np.sum(tp,axis=0)
tnt = np.sum(tn,axis=0)
fpt = np.sum(fp,axis=0)
fnt = np.sum(fn,axis=0)

#rates
tpr = tpt/ptot #true positive rate
tnr = tnt/ntot #true negative rate
fpr = fpt/ntot #false positive rate
fnr = fnt/ptot #false negative rate

x,y = sigs.shape
#when spartina is True
print("Sigmoids.....labels....predictions")
for i in range(x):
    if(pred[i,5] or ann[i,5]):  
        print(sigs[i,:])
        print(ann[i,:])
        print(pred[i,:])
        print("........")

#fout = open(outfile,'w')
#for i in range(N_CLASSES):
#    fout.write('%s\t' % Pennings_Classes[i])
#    fout.write('%d\t%d\t%d\t%d\t' % (tpt[i], tnt[i],fpt[i],fnt[i]) )
#    fout.write('%f\t%f\t%f\t%f\t' % (tpr[i], tnr[i],fpr[i],fnr[i]) )
#    fout.write('\n')

#fout.close()
