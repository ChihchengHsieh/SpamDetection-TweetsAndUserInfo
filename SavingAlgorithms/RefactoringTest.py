#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from urllib.parse import urlparse
import os
import numpy as np
from IPython.display import clear_output, display
import time
import pandas as pd
from Constants import Constants, specialTokenList, specialTokens
from All_Models import SSCL, GatedCNN, SelfAttnModel
from utils import getSampler
from LoadData import loadingTweetsAndUserInfoData
import torch
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader




'''
TODO:

1. Mardularize 
2. Pickle the output
3. Training


'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




#Using inplace method for the dat

class args(object):

    # Data
    
    
    dataset = "HSpam14"
    full_data = True
    usingWeightRandomSampling = True
    vocab_size = 8000 # if we create the new vocab size, we have to do the new preprocess again
    validation_portion = 0.05
    test_portion = 0.04
    random_seed = 64
    
    pickle_name = "UserAndTweetsFullPickleData"+ str(vocab_size) + "Vocab.txt"
    pickle_name_beforeMapToIdx = "UserAndTweetsFullPickleDatabeforeMapToIdx.txt"

    

    

    ##### Arch

    ## GatedCNN arch

    GatedCNN_embedingDim = 128
    GatedCNN_convDim = 64
    GatedCNN_kernel = 3
    GatedCNN_stride = 1
    GatedCNN_pad = 1
    GatedCNN_layers = 8
    GatedCNN_dropout = 0.1
        
    ## SSCL arch
    
    
    SSCL_embedingDim = 512
    SSCL_RNNHidden = 256
    SSCL_CNNDim = 256
    SSCL_CNNKernel = 5
    SSCL_CNNDropout = 0.1
    SSCL_LSTMDropout = 0.1
    SSCL_LSTMLayers = 1
    
    ## Attn arch

    SelfAttn_LenMaxSeq = 280 # Default, will be changed Later

    # These Two has to be the same
    SelfAttn_WordVecDim = 128
    SelfAttn_ModelDim = 128
    
    SelfAttn_FFInnerDim = 256
    SelfAttn_NumLayers = 3
    SelfAttn_NumHead = 4
    SelfAttn_KDim = 64
    SelfAttn_VDim = 64
    SelfAttn_Dropout = 0.1
    
    
    ## MultiTask Model
    
    FC_hidden = 16
    
    
    # Training params
    
    batch_size = 64
    L2 = 0.1
    threshold = 0.5
    lr = 0.002
    n_epoch = 50

    # If using Adam
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    
    earlyStopStep = 2000 # Set None if we don't want it
    earlyStopEpoch = 1 #

    # Logging the Training
    val_freq = 50
    val_steps = 3
    log_freq = 10
    model_save_freq = 1
    model_name = 'TestingModel'
    model_path = './'+ dataset +'_Log/' + model_name + '/Model/'
    log_path = './' + dataset +'_Log/' + model_name + '/Log/'
    
args.device = device

# Create the path for saving model and the log
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)
    


# In[ ]:


training_dataset, validation_dataset, test_dataset, tweets_text = loadingTweetsAndUserInfoData(args)


# In[ ]:


from Trainers import MultiTaskTrainer


# In[ ]:


args.numberOfSpammer = sum([t[-1] for t in training_dataset])
args.numberOfNonSpammer = len(training_dataset)-args.numberOfSpammer
args.len_max_seq = training_dataset[0][2]

print("Number of Spammer: ", args.numberOfSpammer)
print("Number of NonSpammer: ", args.numberOfNonSpammer)


# In[ ]:


if args.usingWeightRandomSampling:
    sampler = getSampler(training_dataset)
else:
    sampler = None

train_loader = DataLoader(
    training_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, sampler = sampler)
valid_loader = DataLoader(
    validation_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

trainer = MultiTaskTrainer(SelfAttnModel, args).to(device)

print("Number of Parameters in this Model: ",trainer.num_all_params())
print("Using device: ", device)

scheduler = optim.lr_scheduler.StepLR(trainer.optim, 2000, gamma=0.85)
# trainer.optim.param_groups[0]['lr']=
allStep = 0
epoch = 0


# In[ ]:


print("Model Structure: \n", trainer.model)


# In[ ]:


while epoch < args.n_epoch:
    for i, (text, extra_info, length, label) in enumerate(train_loader):
        
        trainer.train()
        text, extra_info, length, label = text.to(device), extra_info.to(device), length.to(device), label.to(device)
        
        if trainer.optim.param_groups[0]['lr'] >= 0.00001:
            scheduler.step()
        start_t = time.time()
#         trainer.train_step(((text, extra_info),length), label)
        trainer.train_step(((text, extra_info),None), label)
        

        end_t = time.time()
        allStep += 1
        print('| Epoch [%d] | Step [%d] | lr [%.6f] | Loss: [%.4f] | Acc: [%.4f] | Time: %.1fs' %
              (epoch, allStep, trainer.optim.param_groups[0]['lr'], trainer.loss.item(), trainer.accuracy.item(),
               end_t - start_t))

#         if trainer.accuracy.item() > 0.95: # Stop early
#             raise StopIteration
        if allStep % args.log_freq == 0:
            trainer.plot_train_hist(args.model_name)
            
        
        if args.earlyStopStep:
            if allStep >= args.earlyStopStep:
                    raise StopIteration
        

        if allStep % args.val_freq == 0:

            for _ in range(args.val_steps):
                trainer.eval()
                stIdx = np.random.randint(
                    0, len(validation_dataset) - args.batch_size)
                v_text, v_extra_info, v_len, v_label = validation_dataset[stIdx: stIdx +
                                                       args.batch_size]
                v_text, v_extra_info, v_len, v_label = v_text.to(device), v_extra_info.to(device), v_len.to(device), v_label.to(device)
                start_t = time.time()
#                 trainer.test_step(((v_text, v_extra_info),v_len), v_label)
                trainer.test_step(((v_text, v_extra_info),None), v_label)
                
                end_t = time.time()
                print('| Epoch [%d] | Validation | Step [%d] |  Loss: [%.4f] | Acc: [%.4f] | Time: %.1fs' %
                      (epoch, allStep, trainer.loss.item(), trainer.accuracy.item(), end_t - start_t))
            trainer.calculateAverage()
            clear_output()
            print("TrainConfusion Matrix: \n")
            display(pd.DataFrame(trainer.cms['Train'][-1]))
            print("ValConfusion Matrix: \n")
            display(pd.DataFrame(trainer.cms['Val'][-1]))
            trainer.plot_all(args.model_name)
            
            
     # After every Epoch, if can be moved

    epoch += 1
    trainer.model_save(epoch)


    if args.earlyStopEpoch:
        if epoch >= args.earlyStopEpoch:
            raise StopIteration


# In[ ]:


test_text, test_extra_info, test_length, test_label  =  zip(test_dataset[0:])
test_text, test_extra_info, test_length, test_label  = test_text[0].to(device), test_extra_info[0].to(device), test_length[0].to(device), test_label[0].to(device)

trainer.eval()
test_loss, test_accuracy, test_cm = trainer.test_step(((test_text, test_extra_info),test_length), test_label)

print("The Test Loss: ", test_loss.item())
print("The Test Accuracy: ", test_accuracy.item())
print("Test Confusion Matrix: \n", test_cm)

