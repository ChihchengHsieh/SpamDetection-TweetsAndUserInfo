from urllib.parse import urlparse
import nltk
from nltk.tokenize import TweetTokenizer
import re
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as Data
import numpy as np
from Datasets import HSapm14Dataset, TweetsWithUserInfoDataset
from torch.utils.data import WeightedRandomSampler
from Constants import Constants


def mapIndexToWord(input, text):
    return [text.tokens[i] for i in input]

    
def CreateTweetsWithUserInfoDatatset(X, y):
    '''
    At these step two kind of the sequential data has to be coped with, twitter text and 
    '''
    text_len = torch.tensor(list(map(len, X['text'])))
    text_len, text_len_idx = text_len.sort(0, descending=True)

    temp_text = list(X['text'])
    # The efficiency here need to be improved
    text_ordered = [torch.LongTensor(temp_text[i]) for i in text_len_idx]
    # X_ordered = torch.FloatTensor([list(map(float, list(
    #     X.drop(['text'], axis=1).iloc[i.item(), :]))) for i in text_len_idx])
    X_ordered = torch.FloatTensor(np.array(X.drop(['text'], axis = 1))).index_select(0, text_len_idx)
    y_ordered = torch.FloatTensor(np.array(y)).index_select(0, text_len_idx)
    # y_ordered = torch.FloatTensor(np.array([y[i] for i in text_len_idx]))
    text_p = pad_sequence(text_ordered, batch_first=True)
    print('Dataset Construction')
    dataset = TweetsWithUserInfoDataset(text_p, X_ordered, text_len, y_ordered)
    return dataset


def getSampler(dataset):

    target = torch.tensor([label for _, _, _, label in dataset])
    class_sample_count = torch.tensor(
        [(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = np.array([weight[t.item()] for t in target.byte()])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler


def matchingURL(match):
    try:
        return urlparse(match.group()).netloc or match.group().split("/")[0]
    except:
        return match.group().split("/")[0]


def preprocessingInputData(input):
    tknzr = TweetTokenizer()
    ps = nltk.stem.PorterStemmer()
    allText = [i for i in input]
    preprocessedText = [[ps.stem(word) for word in tknzr.tokenize(re.sub(r'\d+', '', re.sub(r"http\S+|www.\S+", matchingURL,
                                                                                            sentence)).lower()) if word not in nltk.corpus.stopwords.words('english') and len(word) >= 3] for sentence in allText]
    return preprocessedText


def mapFromWordToIdx(input, text):
    '''
    Using text, so it will be changed once we update the text
    '''
    wholeText = []

    for s in input:
        sentence = [2]
        for w in s:
            if w in text.tokens:
                sentence.append(text.index(w))
            else:
                sentence.append(1)
        sentence.append(3)
        wholeText.append(sentence)

    return wholeText


def CreateDatatset(text, X, y):
    '''
    The input X is the idx so we can't get the original one from here?
    '''

    X_len = torch.tensor(list(map(len, X)))
    X_len, X_len_idx = X_len.sort(0, descending=True)

    text_ordered = [text[i] for i in X_len_idx]
    X_ordered = [torch.LongTensor(X[i]) for i in X_len_idx]
    y_ordered = [y[i] for i in X_len_idx]
    y_ordered = torch.FloatTensor(np.array(y_ordered))

    X_p = pad_sequence(X_ordered, batch_first=True)

    dataset = HSapm14Dataset(text_ordered, X_p, X_len, y_ordered)

    return dataset


####################### FOR MASKING #######################

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table).cuda()


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(
        0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask
