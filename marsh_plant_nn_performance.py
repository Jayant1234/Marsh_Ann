# evaluate performance of CNN classifier
import numpy as np
import cv2
import torch
import torch.nn as nn
from marsh_plant_dataset import MarshPlant_Dataset_pc, MarshPlant_Dataset_pa
from sklearn.metrics import confusion_matrix
import itertools
from torchvision import transforms
import matplotlib.pyplot as plt

class Performer(object):
	def __init__(self, data_type="pa", modelname="resnet", batch_size=32,transform=None):
		self.outfile ='performance_bozo.txt'
		self.THRESHOLD_SIG = 0.5
		self.batch_size = batch_size
		self.bShuffle = False
		self.num_workers = 8
		self.modelname=modelname
		self.data_type=data_type
		self.transform=transform
		if(data_type=="pa"): 
			self.N_CLASSES = 7
			self.test_infile = 'marsh_data_all_test.txt'
			self.Pennings_Classes = ['Salicornia','Spartina','Limonium','Borrichia','Batis','Juncus','None']
			self.test_data  = MarshPlant_Dataset_pa(self.test_infile,transform=self.transform)
		elif(data_type=='pc'):
			self.N_CLASSES = 9
			self.test_infile  = 'marsh_percent_cover_test.txt'
			self.Pennings_Classes = [ 'Spartina','Juncus', 'Salicornia','Batis','Borrichia','Limonium','Soil' ,'other','Unknown' ]
			self.test_data  = MarshPlant_Dataset_pc(self.test_infile,transform=self.transform)
			

		
		self.model_path = './modeling/saved_models/'+modelname+'_'+data_type+'.torch'
		model = torch.load(self.model_path)
		#print(model)
		model.eval()
		sigfunc = nn.Sigmoid()
		
		data_loader = torch.utils.data.DataLoader(self.test_data, batch_size = self.batch_size, shuffle = self.bShuffle, num_workers = self.num_workers)

		cpu = torch.device("cpu")
		gpu = torch.device("cuda")

		pred = np.empty((0,self.N_CLASSES), int)
		ann  = np.empty((0,self.N_CLASSES), int)
		sigs = np.empty((0,self.N_CLASSES), int)

		with torch.no_grad():
			for it, batch in enumerate(data_loader):
				output = model(batch['X'].to(gpu)).to(cpu)

				
				if(self.data_type=='pa'):
					sig = output.detach().numpy()
					sigs= np.append(sigs, sig, axis = 0)
					this_pred=np.zeros_like(sig)
					this_pred = sig > self.THRESHOLD_SIG;

				elif(self.data_type=='pc'): 
					sig = sigfunc(output)
					sig = sig.detach().numpy()
					sigs= np.append(sigs, sig, axis = 0)
					this_pred=np.zeros_like(sig)
					this_pred=(sig == sig.max(axis=1)[:,None]).astype(int)
					
				
					
				#print(this_pred)
				pred = np.append(pred, this_pred.astype(int), axis = 0)
				#print("Predictions'sigmoid in Batch size 32")
				#print(sig)
				this_ann = batch['Y'].to(cpu).detach().numpy()  #take off gpu, detach from gradients
				#print("Labels in batch size 32")
				#print(this_ann) 
				ann = np.append(ann, this_ann.astype(int), axis = 0)

		#print(pred)
		#np.savetxt('pred.txt',pred, fmt='%i', delimiter='\t')
		#np.savetxt('ann.txt' ,ann , fmt='%i', delimiter='\t')

		# evaluate performance on test dataset
		#n_samples, c = ann.shape
		self.ann=ann
		self.pred=pred

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
		tpr = tpt/ptot #true positive rate (also called recall)
		tnr = tnt/ntot #true negative rate
		fpr = fpt/ntot #false positive rate
		fnr = fnt/ptot #false negative rate
		ind_precision = tpt/(tpt+fpt) #individual precision 

		#total total for precision, recall and F1 score
		tptt = np.sum(tp)
		tntt = np.sum(tn)
		fptt = np.sum(fp)
		fntt = np.sum(fn)
		precision = tptt/(tptt+fptt)
		recall= tptt/(tptt+fntt) # both these precision and recalls should be same, what is going on here??
		f_score= (2*precision*recall)/(precision+recall)
		accuracy= (tptt+fptt)/(tptt+fptt+fntt+tntt)
		fout = open(self.outfile,'a')
		fout.write('%s\n'%self.model_path)
		fout.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % ("TruePos","TrueNeg","FalsePos","FalseNeg","Recall","TrueNegRt","FalsePosRt","FalseNegRt","Precision"))
		total_precision=0.0
		total_recall=0.0
		for i in range(self.N_CLASSES):
			fout.write('%s\t' % self.Pennings_Classes[i])
			fout.write('%d\t%d\t%d\t%d\t' % (tpt[i], tnt[i],fpt[i],fnt[i]) )
			fout.write('%f\t%f\t%f\t%f\t%f\t' % (tpr[i], tnr[i],fpr[i],fnr[i],ind_precision[i]) )
			fout.write('\n')
			total_precision+=ind_precision[i]
			total_recall+=tpr[i]
		tot_f_score= (2*total_precision*total_recall)/((total_precision+total_recall)*self.N_CLASSES)
		strr=""+"precision="+str(precision)+"\n"+"recall="+str(recall)+"\n"+"f-1 score="+str(f_score)+"\n"+"total_recall"+str(total_recall/self.N_CLASSES)+"total_precision"+str(total_precision/self.N_CLASSES)+"\n total_f_1score= "+str(tot_f_score)+"\n ________________________________________________"+"\n"
		print(strr)
		fout.write('%s'%strr)
		#self.make_confusion_matrix(self.pred,self.ann,self.data_type,fout)
		fout.close()
	def make_confusion_matrix(self,pred,ann,data_type,fout):
		"""Save image with confusion matrix in case the data_type is pc.
		 implement and save plot from this : https://datascience.stackexchange.com/questions/40067/confusion-matrix-three-classes-python
		 implement this for multi-label and multi-class confusion matrix : https://stackoverflow.com/questions/53886370/multi-class-multi-label-confusion-matrix-with-sklearn
		"""
		if(data_type=='pc'): 

			#TODO: Covert Ann and pred to lists with labels.
			x,y=ann.shape
			y_true=[]
			y_pred=[]
			for i in range(x): 
				for j in range(self.N_CLASSES): 
					if(ann[i][j]==1):
						y_true.append(self.Pennings_Classes[j])
				for j in range(self.N_CLASSES): 
					if(pred[i][j]==1):
						y_pred.append(self.Pennings_Classes[j])
						break
					elif(j==8):
						y_pred.append(self.Pennings_Classes[8])
			print(len(y_true))
			print(len(y_pred))
			#print(pred)
			#TODO: Save the image file in results.
			cnf_matrix = confusion_matrix(y_true, y_pred,labels=self.Pennings_Classes)
			np.set_printoptions(precision=2)

			# Plot non-normalized confusion matrix
			plt.figure()
			self.plot_confusion_matrix(fout,cnf_matrix, classes=self.Pennings_Classes,title='Confusion matrix, without normalization')
		elif(data_type=='pa'): 
			self.multi_label_confusion_matrix(ann,pred,fout)

	def plot_confusion_matrix(self,fout, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			#fout.write("Normalized confusion matrix\n")
			#fout.write('Confusion matrix, without normalizatio\n')
		#fout.write(cm.tostring())
		#fout.write('\n')
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)
		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt),
                	horizontalalignment="center",
                 	color="white" if cm[i, j] > thresh else "black")

		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.tight_layout()
		plt.savefig('./Results/'+self.modelname+self.data_type+'.png', bbox_inches='tight',dpi=600)
	def multi_label_confusion_matrix(self,ann,pred,fout): 
		y_true = ann
		y_pred = pred
		labels = self.Pennings_Classes


		conf_mat_dict={}

		for label_col in range(len(labels)):
			y_true_label = y_true[:, label_col]
			y_pred_label = y_pred[:, label_col]
			conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)
		for label, matrix in conf_mat_dict.items():
			print("Confusion matrix for label {}:".format(label))
			print(matrix)

if __name__ == "__main__":
	model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn92', pretrained=True)
			
	image_dim=(512,512)
	modellist=['dpn','densenet','resnext','resnet','inception','pyramid']
	for modelname in modellist: 
		if modelname=='pyramid': 
			image_dim=(224,224)
		transform_test = transforms.Compose([
    	transforms.Resize(image_dim),
    	transforms.ToTensor(), 
    	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		#performer=Performer(data_type="pa",modelname="ResNet101_row50_2020",transform=transform_test)	 
		performer=Performer(data_type="pa",modelname=modelname,transform=transform_test)
