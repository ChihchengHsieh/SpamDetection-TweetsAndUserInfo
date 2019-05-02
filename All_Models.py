import torch
import torch.nn as nn
import gensim
from Constants import Constants
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from SubModels import Encoder
import numpy as np

# Load Google's pre-trained Word2Vec model.
# model = gensim.models.KeyedVectors.load_word2vec_format(
#     './GoogleNews-vectors-negative300.bin', binary=True)

# word2vector = torch.FloatTensor(model.vectors)


class MultiTaskModel(nn.Module):

    def __init__(self, textModel, args):
        super(MultiTaskModel, self).__init__()

        self.textModel = textModel(args)

        self.infoModel = nn.Sequential(
            nn.Linear(args.num_features-1, args.MultiTask_FCHidden*2),
            nn.BatchNorm1d(args.MultiTask_FCHidden*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.MultiTask_FCHidden*2, args.MultiTask_FCHidden*2),
            nn.BatchNorm1d(args.MultiTask_FCHidden*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.MultiTask_FCHidden*2, args.MultiTask_FCHidden),
            nn.BatchNorm1d(args.MultiTask_FCHidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.MultiTask_FCHidden, args.infoModel_outDim),
        )

        self.combineModel = nn.Sequential(
            nn.Linear(args.textModel_outDim +
                      args.infoModel_outDim, args.combine_dim),
            nn.BatchNorm1d(args.combine_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.combine_dim, 1)
            # nn.Linear(args.textModel_outDim +
            #           args.infoModel_outDim, 1),
        )

    def forward(self, input, length):

        text_input, extra_info = input

        text_out = self.textModel(text_input, length)

        # return text_out

        extra_info_out = self.infoModel(extra_info)

        return self.combineModel(torch.cat((text_out, extra_info_out), dim=1))


class SSCL(nn.Module):

    ''' The Model from paper '''

    def __init__(self, args):
        super(SSCL, self).__init__()

        self.embed = nn.Embedding(
            args.vocab_size, args.SSCL_embedingDim, Constants.PAD)

        self.cnn = nn.Sequential(
            nn.Conv1d(args.SSCL_embedingDim, args.SSCL_CNNDim,
                      args.SSCL_CNNKernel, 1, 2),
            nn.BatchNorm1d(args.SSCL_CNNDim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(args.SSCL_CNNDropout),
        )

        self.rnn = nn.LSTM(args.SSCL_CNNDim, args.SSCL_RNNHidden,
                           batch_first=True, dropout=args.SSCL_LSTMDropout, num_layers=args.SSCL_LSTMLayers)

        self.out_net = nn.Sequential(
            nn.Linear(args.SSCL_RNNHidden, args.textModel_outDim),
        )

        self.h0 = nn.Parameter(torch.randn(1, args.SSCL_RNNHidden))
        self.c0 = nn.Parameter(torch.randn(1, args.SSCL_RNNHidden))

#         self.apply(self.weight_init)

    def forward(self, input, lengths=None):

        B = input.size(0)

        emb_out = self.embed(input).transpose(1, 2)

        out = self.cnn(emb_out).transpose(1, 2)

        if not lengths is None:
            out = pack_padded_sequence(out, lengths, batch_first=True)
            out, hidden = self.rnn(
                out, (self.h0.repeat(1, B, 1), self.c0.repeat(1, B, 1)))
            out = pad_packed_sequence(out, batch_first=True)[0][:, -1, :]
        else:
            # out = self.rnn(out,(self.h0.repeat(1,B,1), self.c0.repeat(1,B,1)))[0][:, -1, :]
            # out = self.rnn(out)[0][:, -1, :]
            out = self.rnn(out)[0].sum(dim=1)

        out = self.out_net(out)

        return out

    def weight_init(self, m):

        if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
            nn.init.kaiming_normal_(m.weight, 0.2, nonlinearity='leaky_relu')
        elif type(m) in [nn.LSTM]:
            for name, value in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(value.data)
                if 'bias'in name:
                    value.data.normal_()


class GatedCNN(nn.Module):

    def __init__(self, args):
        super(GatedCNN, self).__init__()

        self.emb = nn.Embedding(args.vocab_size, args.GatedCNN_embedingDim)

        self.conv = nn.ModuleList([nn.Conv1d(args.GatedCNN_embedingDim, args.GatedCNN_convDim,
                                             args.GatedCNN_kernel, args.GatedCNN_stride, args.GatedCNN_pad)])
        self.conv.extend([
            nn.Sequential(
                nn.Conv1d(args.GatedCNN_convDim, args.GatedCNN_convDim, args.GatedCNN_kernel,
                          args.GatedCNN_stride, args.GatedCNN_pad),
                nn.BatchNorm1d(args.GatedCNN_convDim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(args.GatedCNN_dropout)
            )
            for _ in range(args.GatedCNN_layers)])
        self.conv_gate = nn.ModuleList([nn.Conv1d(
            args.GatedCNN_embedingDim, args.GatedCNN_convDim, args.GatedCNN_kernel, args.GatedCNN_stride, args.GatedCNN_pad)])
        self.conv_gate.extend([
            nn.Sequential(
                nn.Conv1d(args.GatedCNN_convDim, args.GatedCNN_convDim, args.GatedCNN_kernel,
                          args.GatedCNN_stride, args.GatedCNN_pad),
                nn.BatchNorm1d(args.GatedCNN_convDim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(args.GatedCNN_dropout),
            )
            for _ in range(args.GatedCNN_layers)])

        self.fc = nn.Sequential(
            nn.Linear(args.GatedCNN_convDim, args.textModel_outDim),
        )

    def forward(self, input, lengths=None):

        out_ = self.emb(input).transpose(1, 2)

        for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            out_temp = out_  # Prepare for Residule
            out_a = conv(out_)
            out_b = conv_gate(out_)
            out_ = out_a * torch.sigmoid(out_b)
            if out_temp.size()[1] == out_.size()[1]:
                out_ += out_temp  # Residule

        out_ = out_.sum(dim=-1)

        out_ = self.fc(out_)

        return out_

    def weight_init(self, m):

        if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Conv1d]:
            nn.init.kaiming_normal_(m.weight, 0.2, nonlinearity='leaky_relu')
        elif type(m) in [nn.LSTM]:
            for name, value in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(value.data)
                if 'bias'in name:
                    value.data.normal_()


class SelfAttnModel(nn.Module):
    '''
    Input: A sequence 
    Output: A Single unit output
    '''

    def __init__(self, args):
        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=args.vocab_size, len_max_seq=args.SelfAttn_LenMaxSeq,
            d_word_vec=args.SelfAttn_WordVecDim, d_model=args.SelfAttn_ModelDim, d_inner=args.SelfAttn_FFInnerDim,
            n_layers=args.SelfAttn_NumLayers, n_head=args.SelfAttn_NumHead, d_k=args.SelfAttn_KDim, d_v=args.SelfAttn_VDim,
            dropout=args.SelfAttn_Dropout)

        self.fc = nn.Linear(args.SelfAttn_ModelDim, args.textModel_outDim)

        assert args.SelfAttn_ModelDim == args.SelfAttn_WordVecDim, \
            'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, input, lengths=None):

        enc_output, *_ = self.encoder(input)
        out_ = self.fc(enc_output)

        return out_.sum(dim=1)
