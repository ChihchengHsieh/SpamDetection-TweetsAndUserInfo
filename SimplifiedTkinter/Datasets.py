import torch
from torch.utils.data import Dataset, DataLoader
        

class TweetsWithUserInfoDataset(Dataset):
    def __init__(self, text_p, X_ordered, text_len, y_ordered):
        super(TweetsWithUserInfoDataset, self).__init__()
        '''
        The Pytorch dataset for feeding the data into Neural Network.
        '''
        self.text_p = text_p
        self.X_ordered = X_ordered
        self.text_len = text_len
        self.y_ordered = y_ordered

    def __len__(self,):
        return len(self.y_ordered)

    def __getitem__(self, idx):
        return self.text_p[idx], self.X_ordered[idx], self.text_len[idx], self.y_ordered[idx]
