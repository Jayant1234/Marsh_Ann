import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from modeling.backbone.resnet import ResNet101
from marsh_plant_dataset import MarshPlant_Dataset_pa, MarshPlant_Dataset_pc
from RN101_newtop import RN101_newtop
from marsh_plant_nn_performance import Performer
import PyramidNet_model.PyramidNet as prn
import torch.utils.data.distributed
import torch.distributed as dist
from Attention_Augmented.attention_augmented_wide_resnet import Wide_ResNet
from neatcnn import NeatCNN 

class Trainer(object):
	def __init__(self, data_type="pa", modelname="resnet", batch_size_top =16,batch_size_all=4,epochs_top=20,epochs_all=30):
		self.batch_size_top = batch_size_top
		self.batch_size_all = batch_size_all
		self.data_type=data_type
		self.modelname=modelname
		if(data_type=="pa"):
			self.N_CLASSES = 7
			self.train_infile = 'marsh_data_all_train.txt'
			self.val_infile  = 'marsh_data_all_val.txt'
		elif(data_type=='pc'):
			self.N_CLASSES = 9
			self.train_infile='marsh_percent_cover_train'
			self.val_infile="marsh_percent_cover_val.txt"
			
		self.log_file = open("marsh_plant_nn_training_logfile.txt","w")

		self.epochs_top = epochs_top #30 number of epochs to train the top of the model
		self.epochs_all = epochs_all #20 number of epochs to train the entire model
		self.lr_top = 1e-4 # learning rate for training the top of your model
		self.lr_all = 1e-5 # learning rate to use when training the entire model
		self.model_path = './modeling/saved_models/'+modelname+'_'+data_type+'.torch'

###
# DataLoader Setup
###
	def setup_dataloader(self,dataset_dict,batch_size, bShuffle,num_workers,samplers):
		dataloaders = {}
		for key in dataset_dict:
			if samplers is not None:
				dataloaders[key] =  torch.utils.data.DataLoader(dataset_dict[key], batch_size = batch_size, shuffle = bShuffle, num_workers = num_workers,sampler=samplers[key])
			else: 
				dataloaders[key] =  torch.utils.data.DataLoader(dataset_dict[key], batch_size = batch_size, shuffle = bShuffle, num_workers = num_workers)
		return dataloaders

###
# Model Setup
###

	def setup_model(self):
		if(modelname=="resnet"):
			pretrained_model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=16)
			resnet_bottom = torch.nn.Sequential(*list(pretrained_model.children())[:-1]) # remove last layer (fc) layer
			model = RN101_newtop(base_model = resnet_bottom, num_classes = self.N_CLASSES)
		elif(modelname=='densenet'):
			model = models.densenet121(pretrained=True)
			#densenet_bottom = torch.nn.Sequential(*list(pretrained_model.children())[:-1]) # remove last layer (fc) layer
			#pretrained_model.classifier = nn.Linear(1024, num_classes)
			#model = RN101_newtop(base_model = densenet_bottom, num_classes = self.N_CLASSES)
			num_ftrs = model.classifier.in_features
			model.classifier = nn.Linear(num_ftrs,self.N_CLASSES)
		elif(modelname=='inception'):
			model= models.inception_v3(pretrained=True)
			#set_parameter_requires_grad(model_ft, feature_extract)
			# Handle the auxilary net
			num_ftrs = model.AuxLogits.fc.in_features
			model.AuxLogits.fc = nn.Linear(num_ftrs, self.N_CLASSES)
			# Handle the primary net
			num_ftrs = model.fc.in_features
			model.fc = nn.Linear(num_ftrs,self.N_CLASSES)
		elif(modelname=='resnext'): 
			self.batch_size_all=4 # got cuda error
			model= models.resnext101_32x8d(pretrained=True)
			num_ftrs = model.fc.in_features
			model.fc = nn.Linear(num_ftrs,self.N_CLASSES)
		elif(modelname=='pyramid'): 
			model= prn.PyramidNet(dataset='imagenet', depth=101, alpha=360, num_classes=1000, bottleneck=True,pretrain=True) #input imagenet args
			num_ftrs = model.fc.in_features
			#model = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
			model.fc = nn.Linear(num_ftrs,self.N_CLASSES)
		elif(modelname=='dpn'):
			model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn92', pretrained=True)
			num_chs = model.classifier.in_channels#*self.batch_size_all
			model.classifier = nn.Conv2d(num_chs, self.N_CLASSES, kernel_size=1, bias=True)
		elif(modelname=='aawide'): 
			model = Wide_ResNet(depth=10, widen_factor=5, dropout_rate=0.3, num_classes=self.N_CLASSES, shape=32)
			self.lr_all= 1e-3
			self.epochs_all=100
		elif(modelname=='neat'):
			model = NeatCNN(num_classes=self.N_CLASSES,channel_size=256,group_size=2,depth=1,width=1,residual=False)
			self.epochs_all=50
			self.batch_size_all=4
		elif(modelname=='neater'):
			model = NeatCNN(num_classes=self.N_CLASSES,channel_size=256,group_size=2,depth=3,width=1,residual=True)
			self.epochs_all=50
			self.batch_size_all=4
		return model

# helper functions
	def count_parameters(self,model):
		return sum(p.numel() for p in model.parameters() if p.requires_grad)


	def train(self,model, dataloaders, criterion, optimizer, num_epochs, scheduler,  best_acc=0 ):
		for epoch in range(num_epochs):
			for phase in ['train','val']:
				if phase == 'train':
					model.train()
				else:
					model.eval()

				running_loss = 0.0
				running_tp = 0
				running_pos = 0
				running_samples = 0

				for it, batch in enumerate(dataloaders[phase]):
					inputs  = batch['X'].cuda()#to(device)
					target = batch['Y'].cuda()#to(device)
					#print("input's size")
					#print(inputs.size())
					#input_var = torch.autograd.Variable(inputs)
					#print("variable'ssize")
					#print(input_var.size())

					optimizer.zero_grad()  #zero gradients

					with torch.set_grad_enabled(phase =='train'):  #track gradients for backprop in training phase
						if(self.modelname=='inception' and phase=='train'):
							output, aux_output = model(inputs)
							sigfunc = nn.Sigmoid()
							sig = sigfunc(output)
							if(self.data_type=='pc'):
								target=target.long()
								#print(target)
								label=torch.max(target, 1)[1]
								loss1 = criterion(output, label)
								loss2 = criterion(aux_output, label)
								loss = loss1 + 0.4*loss2
							else: 
								loss1 = criterion(output, target)
								loss2 = criterion(aux_output, target)
								loss = loss1 + 0.4*loss2  # evaluate loss
							
						else : 
							output = model(inputs)  # pass in image series
							#_, preds = torch.max(output,1)
							sigfunc = nn.Sigmoid()
							sig = sigfunc(output)

							if(self.data_type=='pc'):
								target=target.long()
								#print(target)
								label=torch.max(target, 1)[1]
								loss = criterion(output, label)  # evaluate loss
							else: 
								loss = criterion(output, target)  # evaluate loss

						if phase == 'train':
							loss.backward()  # update the gradients
							optimizer.step()  # update sgd optimizer lr

					# compare model output and annotations  - more easily done with numpy
					target_np = target.to("cpu").detach().numpy()  #take off gpu, detach from gradients
					target_np = target_np.astype(int)
					sig_np    = sig.to("cpu").detach().numpy()
					pred = sig_np > 0.5;
					pred = pred.astype(int)
					corr = np.equal(target_np, pred)
					tp = np.where(target_np == 1, corr, False )  #true positives
					fp = np.where(target_np == 0, np.logical_not(corr), False)  #false positives
					tn = np.where(target_np == 0, corr, False )  #true negatives
					fn = np.where(target_np == 1, np.logical_not(corr), False)  # false negatives
					#print(np.sum(tp))
					#print(fp)
				#    print(sig_np.shape)

					running_loss += loss.item() * inputs.size(0)
					running_tp += np.sum(tp)
					running_pos += np.sum(target_np)
					running_samples += inputs.size(0)

					if it % 50 == 0:
						avg_loss= running_loss/(running_samples)
						tp_rate = running_tp/running_pos
						#print("Epoch:", epoch, "Iteration:", it, "Average Loss:", avg_loss, "true_pos_rate", tp_rate)  # print the running/average loss, iterat starts at 0 thus +1
						print('Epoch: {}, Iteration: {}, Loss: {:.4f}, TruePos_rate: {:.4f}'.format(epoch, it,avg_loss,tp_rate))

				if scheduler:
					scheduler.step()
					print("Learning rate :")
					print(scheduler.get_lr())
					
				# save model if it is the best model on val set
				epoch_loss = running_loss/running_samples
				epoch_acc = running_tp/running_pos
				print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
				self.log_file.write('epoch\t{}\tphase\t{}\tLoss\t{:.4f}\tAcc\t{:.4f}\n'.format(epoch, phase, epoch_loss, epoch_acc))

				if phase == 'val' and epoch_acc > best_acc:
					best_acc = epoch_acc
					torch.save(model, self.model_path)# save state dict, this method is bound to break. 

		return best_acc



#for batch in data_loader:
#    output = model(batch['X'].to(device))
#    loss = criterion(output, batch['Y'].to(device))
#    #print(batch['Y'])
#    print(loss)


if __name__ == "__main__":
	print("PyTorch Version: ",torch.__version__)	
	#inputs 
	data_type="pc"
	modellist=['dpn']#,'neat' 'aawide','resnext','densenet','resnet','inception','pyramid','dpn']
	image_dim=(512,512)

	distributed=False
	
	
	for modelname in modellist: 

		trainer = Trainer(data_type=data_type,modelname=modelname)
		if(modelname=="pyramid" ): #or modelname=='dpn'): 
			image_dim=(224,224)
		if(modelname=="aawide" ): 
			image_dim=(32,32)
		#if(modelname=="dpn"): 
			#distributed=True
		if distributed :
			dist.init_process_group(backend='gloo', init_method='env://',
                                rank=0,world_size=4)
		
		transform_train = transforms.Compose([
		transforms.Resize(image_dim),transforms.ToTensor(), 
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

		transform_test = transforms.Compose([
	    	transforms.Resize(image_dim),
	    	transforms.ToTensor(), 
	    	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	
		transform_val = transforms.Compose([
	    	transforms.Resize(image_dim),
	    	transforms.ToTensor(), 
	    	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

	
	
		if(data_type=="pa"): 
			train_data = MarshPlant_Dataset_pa(trainer.train_infile,transform=transform_train)
			val_data  = MarshPlant_Dataset_pa(trainer.val_infile,transform=transform_val)
			datasets = {'train' : train_data, 'val' : val_data}
			criterion = nn.BCEWithLogitsLoss().cuda()
			if distributed: 
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
				val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
				samplers={'train' : train_sampler, 'val' : val_sampler}
		if(data_type=="pc"): 
			train_data = MarshPlant_Dataset_pc(trainer.train_infile,transform=transform_train)
			val_data  = MarshPlant_Dataset_pc(trainer.val_infile,transform=transform_val)
			datasets = {'train' : train_data, 'val' : val_data}
			criterion = nn.CrossEntropyLoss().cuda()#change to cross entropy here. 
			if distributed: 
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
				val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
				samplers={'train' : train_sampler, 'val' : val_sampler}

		bShuffle = True
		num_workers = 8
		if not distributed: 
			samplers=None
		dataloaders_top = trainer.setup_dataloader(datasets, trainer.batch_size_top, bShuffle, num_workers,samplers=samplers)
		dataloaders_all = trainer.setup_dataloader(datasets, trainer.batch_size_all, bShuffle, num_workers,samplers=samplers)
	
		#setup model and training parameters
	    
		model = trainer.setup_model()
		print(model)
		model.cuda()
		if distributed: 
			model = torch.nn.parallel.DistributedDataParallel(model)
		
		# Train Top (fc) layer
		# Freeze bottom of model so we are only training the top linear layer
		for param in model.parameters():
			param.requires_grad = False
		if(modelname=='densenet' or modelname=='dpn'or modelname=='neat' or  modelname=='neater'): 
			params_to_optimize_in_top = list(model.classifier.parameters())
		else: 
			params_to_optimize_in_top = list(model.fc.parameters())
		
		for param in params_to_optimize_in_top:
			param.requires_grad = True
		
		optimizer_top = torch.optim.Adam(params_to_optimize_in_top, lr = trainer.lr_top)
		lr_scheduler_top = torch.optim.lr_scheduler.StepLR(optimizer_top, step_size=10, gamma=0.5)
		#lr_scheduler_top = None
		print("Training Top:", trainer.count_parameters(model), "Parameters")
		
		b_acc = trainer.train(model, dataloaders_top, criterion, optimizer_top, trainer.epochs_top, scheduler = lr_scheduler_top, best_acc=0)
		print('Finished training top, best acc {:.4f}'.format(b_acc))
		
	
		for param in model.parameters():
			param.requires_grad = True
		print("Training All:", trainer.count_parameters(model), "Parameters")
	
		# Optimizer for Entire Network
		optimizer_all = torch.optim.Adam(model.parameters(), lr = trainer.lr_all)
		lr_scheduler_all = torch.optim.lr_scheduler.StepLR(optimizer_all, step_size=10, gamma=0.8)
		#lr_scheduler_all = None
		# Train on All
	
		b_acc = trainer.train(model, dataloaders_all, criterion, optimizer_all, trainer.epochs_all, scheduler = lr_scheduler_all, best_acc=b_acc)
		print('Finished training bottom, best acc {:.4f}'.format(b_acc))
	
		performer=Performer(data_type=data_type,modelname=modelname,transform=transform_test)
		print("Finished Performer class on test")