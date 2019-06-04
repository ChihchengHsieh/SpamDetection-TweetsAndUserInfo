from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from All_Models import MultiTaskModel

import matplotlib


class TkMultiTaskTrainer(nn.Module):
    '''
    Contain some useful training function to facilitate the training

    Two important training factors are here:

    1. Optimizer
    2. Loss function 
    '''

    def __init__(self, model, args, textbox, window):
        super(TkMultiTaskTrainer, self).__init__()

        self.textbox = textbox
        self.window = window

        self.model = MultiTaskModel(model, args)

        ## Init the optimizer
        self.optim = optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.L2)

        if args.usingWeightRandomSampling:
            pos_weight = None
        else:
            pos_weight = torch.tensor(
                args.numberOfNonSpammer/args.numberOfSpammer)

        self.threshold = args.threshold
        self.log_path = args.log_path
        self.model_path = args.model_path
        self.model_name = args.model_name

        # Init the Loss function
        self.Loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.hist = defaultdict(list)
        self.cms = defaultdict(list)
        self.confusion_matrics = []

        self.apply(self.weight_init)

    def forward(self, input, label):

        '''
        Perform forward step for generating the loss, accuracy and confusion matrix.
        '''

        if type(input) is tuple:
            input, lengths = input
        else:
            lengths = None

        self.pred = self.model(input, lengths)
        self.label = label
        loss = self.Loss(self.pred.squeeze(1), label)

        accuracy = torch.mean(
            ((torch.sigmoid(self.pred) > self.threshold).squeeze(-1) == label.byte()).float())

        cm = confusion_matrix(label.cpu().numpy(),
                              (torch.sigmoid(self.pred) > self.threshold).cpu().numpy())

        return loss, accuracy, cm

    def train_step(self, input, label):

        '''
        Training step that enables the gradient propagation.
        '''

        self.optim.zero_grad()

        self.loss, self.accuracy, self.cm = self.forward(input, label)

        self.hist["Temp_Train_Loss"].append(self.loss.item())
        self.hist["Temp_Train_Accuracy"].append(self.accuracy.item())
        self.hist["Train_Loss"].append(self.loss.item())
        self.hist["Train_Accuracy"].append(self.accuracy.item())
        self.cms["Train"].append(self.cm)
        self.cms["Train"] = self.cms["Train"][-10:]
        self.loss.backward()
        # clip_grad_norm(self.model.parameters(), 0.25)
        self.optim.step()

        return self.loss, self.accuracy, self.cm

    def test_step(self, input, label, validation=True):

        '''
        Test step that disable the gradient propagation. Therefore, the weight will not be changed in this step.
        '''
        self.loss, self.accuracy, self.cm = self.forward(input, label)

        if validation:
            self.hist["Temp_Val_Loss"].append(self.loss.item())
            self.hist["Temp_Val_Accuracy"].append(self.accuracy.item())
            self.hist["Val_Loss"].append(self.loss.item())
            self.hist["Val_Accuracy"].append(self.accuracy.item())
            self.cms["Val"].append(self.cm)
            self.cms["Val"] = self.cms["Val"][-10:]
        else:
            self.hist["Temp_Test_Loss"].append(self.loss.item())
            self.hist["Temp_Test_Accuracy"].append(self.accuracy.item())
            self.hist["Test_Loss"].append(self.loss.item())
            self.hist["Test_Accuracy"].append(self.accuracy.item())
            self.cms["Test"].append(self.cm)
            self.cms["Test"] = self.cms["Test"][-10:]

        return self.loss, self.accuracy, self.cm

    def calculateAverage(self,):
        '''
        Calculate the average loss and acuracy
        '''
        temp_keys = deepcopy(list(self.hist.keys()))
        for name in temp_keys:
            if 'Temp' in name:
                self.hist["Average" + name[4:]
                          ].append(sum(self.hist[name])/len(self.hist[name]))
                self.hist[name] = []

    def plot_train_hist(self, step):
        '''
        Plot the training history
        '''

        fig = plt.figure(figsize=(20, 10))
        num_loss = 2
        i = 0
        for name in self.hist.keys():
            if 'Train' in name and not "Temp" in name and not "Average" in name:
                i += 1
                fig.add_subplot(num_loss, 1, i)
                plt.plot(self.hist[name], label=name)
                plt.xlabel('Number of Steps', fontsize=8)
                plt.ylabel(name, fontsize=8)
                plt.title(name, fontsize=8, fontweight="bold")
                plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        fig.savefig(self.log_path+"Train_Loss&Acc_Hist_"+str(step)+".png")

        return fig

    def plot_all(self, step=None):

        '''
        Plot the training and validation history
        '''

        fig = plt.figure(figsize=(20, 10))
        for name in self.hist.keys():
            if "Average" in name:
                if 'Loss' in name:
                    plt.subplot(211)
                    plt.plot(self.hist[name], marker='o', label=name)
                    plt.ylabel('Loss', fontsize=8)
                    plt.xlabel('Number of epochs', fontsize=8)
                    plt.title('Loss', fontsize=8, fontweight="bold")
                    plt.legend(loc='upper left')
                if "Accuracy" in name:
                    plt.subplot(212)
                    plt.plot(self.hist[name], marker='o', label=name)
                    plt.ylabel('Accuracy', fontsize=8)
                    plt.xlabel('Number of epochs', fontsize=8)
                    plt.title('Accuracy', fontsize=8, fontweight="bold")
                    plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        if step is not None:
            fig.savefig(self.log_path + "All_Hist_"+str(step)+".png")

        return fig

    def model_save(self, step):

        '''
        Storing the weight of the trained model
        '''

        path = self.model_path + self.model_name+'_Step_' + str(step) + '.pth'
        torch.save({self.model_name: self.state_dict()}, path)
        self.textbox.insert("end", 'Model Saved\n')
        self.window.update_idletasks()

    def load_step_dict(self, step):

        '''
        Load the stored weight.
        '''

        path = self.model_path + self.model_name + \
            '_Step_' + str(step) + '.pth'
        self.load_state_dict(torch.load(
            path, map_location=lambda storage, loc: storage)[self.model_name])
        self.textbox.insert("end", 'Model Loaded\n')
        self.window.update_idletasks()

    def num_all_params(self,):
        '''
        Print how many parameters in the model
        '''
        return sum([param.nelement() for param in self.parameters()])

    def weight_init(self, m):

        '''
        Initialising the weihgt
        '''

        if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv1d]:
            nn.init.kaiming_normal_(m.weight, 0.2, nonlinearity='leaky_relu')
        elif type(m) in [nn.LSTM]:
            for name, value in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(value.data)
                if 'bias'in name:
                    value.data.normal_()
