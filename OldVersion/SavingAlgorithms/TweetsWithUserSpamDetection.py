import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from urllib.parse import urlparse
import os
import numpy as np
from IPython.display import clear_output, display
import time
import pandas as pd
from Trainers import MultiTaskTrainer
from Constants import Constants, specialTokenList, specialTokens
from All_Models import SSCL, GatedCNN, SelfAttnModel
from utils import getSampler
from LoadData import loadingTweetsAndUserInfoData
import torch
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader


%matplotlib inline


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
    MultiTask_FCHidden = 16
    textModel_outDim = 16
    infoModel_outDim = 16
    combine_dim = 32

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
    
    
    
    # Training params
    
    batch_size = 64
    L2 = 0.1
    threshold = 0.5
    lr = 0.0002
    n_epoch = 50

    # If using Adam
    
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    
    earlyStopStep = None # Set None if we don't want it
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
    

training_dataset, validation_dataset, test_dataset, tweets_text = loadingTweetsAndUserInfoData(args)


args.numberOfSpammer = sum([t[-1] for t in training_dataset])
args.numberOfNonSpammer = len(training_dataset)-args.numberOfSpammer
args.len_max_seq = training_dataset[0][2]

print("Number of Spammer: ", args.numberOfSpammer)
print("Number of NonSpammer: ", args.numberOfNonSpammer)



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

print("Model Structure: \n", trainer.model)


while epoch < args.n_epoch:
    for i, (text, extra_info, length, label) in enumerate(train_loader):
        
        trainer.train()
        text, extra_info, length, label = text.to(device), extra_info.to(device), length.to(device), label.to(device)
        
#         if trainer.optim.param_groups[0]['lr'] >= 0.0001:
#             scheduler.step()
        start_t = time.time()
#         trainer.train_step(((text, extra_info),length), label)
        trainer.train_step(((text, extra_info),None), label)
        

        end_t = time.time()
        allStep += 1
        print('| Epoch [%d] | Step [%d] | lr [%.6f] | Loss: [%.4f] | Acc: [%.4f] | Time: %.1fs' %
              (epoch, allStep, trainer.optim.param_groups[0]['lr'], trainer.loss.item(), trainer.accuracy.item(),
               end_t - start_t))

#         if trainer.accuracy.item() > 0.95: # Stop early
#             break
        if allStep % args.log_freq == 0:
            trainer.plot_train_hist(args.model_name)
            
        
        if args.earlyStopStep:
            if allStep >= args.earlyStopStep:
                break
        

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
            break


test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
test_accs = []
test_cms = []
trainer.eval()
for i, (test_text, test_extra_info, test_length, test_label) in enumerate(test_loader):
    test_text, test_extra_info, test_length, test_label = test_text.to(device), test_extra_info.to(device), test_length.to(device), test_label.to(device)
    test_loss, test_accuracy, test_cm = trainer.test_step(((test_text, test_extra_info),test_length), test_label)
    test_accs.append(test_accuracy)
    test_cms.append(test_cm)
print("Test Accuracy: ", torch.mean(torch.tensor(test_accs)))
print("Test Confusion Matrix: \n", sum(test_cms))