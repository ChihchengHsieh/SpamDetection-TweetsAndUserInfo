import torch
import torch.nn as nn
import gensim
from Constants import Constants
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from SubModels import Encoder
import numpy as np


class MultiTaskModel(nn.Module):
    '''
    This model contain three NN.

    1. textModel
    2. infoModel
    3. combineModel

    And, this model will use these models according to the hyper-parameter (using_infoModel, using_textModel) setting
    '''
    def __init__(self, textModel, args):
        super(MultiTaskModel, self).__init__()

        ## If using the textModel
        if args.using_textModel:
            # the textModel will the model provided in the constructor argument
            self.textModel = textModel(args)
            self.customForward = self.usingTextForward


        ## If using the infoModel
        if args.using_infoModel:
            
            # This is the structure of infoModel
            self.infoModel = nn.Sequential(
                nn.Linear(args.num_features-1, args.MultiTask_FCHidden*2),
                nn.BatchNorm1d(args.MultiTask_FCHidden*2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(args.MultiTask_FCHidden*2,
                          args.MultiTask_FCHidden*2),
                nn.BatchNorm1d(args.MultiTask_FCHidden*2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(args.MultiTask_FCHidden*2, args.MultiTask_FCHidden),
                nn.BatchNorm1d(args.MultiTask_FCHidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(args.MultiTask_FCHidden, args.infoModel_outDim),
            )
            self.customForward = self.usingInfoForward

        if args.using_textModel and args.using_infoModel:
            ## If using both info and text models, the combine model will be used for concating the outputs from those models.
            self.combineModel = nn.Sequential(
                nn.Linear(args.textModel_outDim +
                          args.infoModel_outDim, args.combine_dim),
                nn.BatchNorm1d(args.combine_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(args.combine_dim, 1)
            )
            self.customForward = self.usingBothForward

    def forward(self, input, length=None):

        '''
        Feed Forward steps, three different cases are considered

        1. only using textModel
        2. only using infoModel
        3. using both info and text model
        '''

        return self.customForward(input, length)

    def usingTextForward(self, input, length):

        text_input, extra_info = input

        text_out = self.textModel(text_input, length)

        return text_out

    def usingInfoForward(self, input, length):

        text_input, extra_info = input

        extra_info_out = self.infoModel(extra_info)

        return extra_info_out

    def usingBothForward(self, input, length):

        text_input, extra_info = input

        text_out = self.textModel(text_input, length)

        extra_info_out = self.infoModel(extra_info)

        return self.combineModel(torch.cat((text_out, extra_info_out), dim=1))


class SSCL(nn.Module):

    ''' 
    The Model from paper 
    Stacked Sequential Covolutional LSTM.
    The basic concept of this model is to stack a LSTM layer above a CNN layer.
    
    '''

    def __init__(self, args):
        super(SSCL, self).__init__()


        ## This is the embedding layer for transforming the index to word vector.
        ## More details: https://pytorch.org/docs/stable/nn.html#embedding
        self.embed = nn.Embedding(
            args.vocab_size, args.SSCL_embedingDim, Constants.PAD)

        ## This is the CNN block
        '''
        The nn.Sequentail function is used for constructing the neural network
        This CNN block consist of:

        1. 1D-Covolution: https://pytorch.org/docs/stable/nn.html#torch.nn.functional.conv1d
        2. 1D-BatchNorm: https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm1d
        3. LeakyReLU: https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU
        4. Dropout: https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout

        ''' 
        self.cnn = nn.Sequential(
            nn.Conv1d(args.SSCL_embedingDim, args.SSCL_CNNDim,
                      args.SSCL_CNNKernel, 1, 2),
            nn.BatchNorm1d(args.SSCL_CNNDim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(args.SSCL_CNNDropout),
        )


        # LSTM: https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        self.rnn = nn.LSTM(args.SSCL_CNNDim, args.SSCL_RNNHidden,
                           batch_first=True, dropout=args.SSCL_LSTMDropout, num_layers=args.SSCL_LSTMLayers)

        
        # The final fully connected layer for maping the output to wanted dimension
        # Linear: https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        self.out_net = nn.Sequential(
            nn.Linear(args.SSCL_RNNHidden, args.textModel_outDim),
        )

        ## The trainable init h0 and c0 in LSTM.
        self.h0 = nn.Parameter(torch.randn(1, args.SSCL_RNNHidden))
        self.c0 = nn.Parameter(torch.randn(1, args.SSCL_RNNHidden))


    def forward(self, input, lengths=None):
        '''
        Feed Forward Step of this Neural Network (SSCL)
        '''

        B = input.size(0)

        # Pass through the embedding layer
        emb_out = self.embed(input).transpose(1, 2)

        # Pass through the CNN layer
        out = self.cnn(emb_out).transpose(1, 2)

        # Pass through the LSTM layer
        if not lengths is None:
            out = pack_padded_sequence(out, lengths, batch_first=True)
            out, hidden = self.rnn(
                out, (self.h0.repeat(1, B, 1), self.c0.repeat(1, B, 1)))
            out = pad_packed_sequence(out, batch_first=True)[0][:, -1, :]
        else:
            # out = self.rnn(out,(self.h0.repeat(1,B,1), self.c0.repeat(1,B,1)))[0][:, -1, :]
            # out = self.rnn(out)[0][:, -1, :]
            out = self.rnn(out)[0].sum(dim=1)

        # Map to the output dimesion
        out = self.out_net(out)

        return out


class GatedCNN(nn.Module):

    '''
    This model consit of multiple gated Convolution block 
    
    Gated Convolution: https://arxiv.org/pdf/1612.08083.pdf

    This CNN block in this model consist of:

    1. 1D-Covolution: https://pytorch.org/docs/stable/nn.html#torch.nn.functional.conv1d
    2. 1D-BatchNorm: https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm1d
    3. LeakyReLU: https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU
    4. Dropout: https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout


    '''


    def __init__(self, args):
        super(GatedCNN, self).__init__()



        ## The embedding layer
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

        # The final neural network for mapping the output to wanted dimension
        self.fc = nn.Sequential(
            nn.Linear(args.GatedCNN_convDim, args.textModel_outDim),
        )

    def forward(self, input, lengths=None):
        

        # Pass through the embedding layer
        out_ = self.emb(input).transpose(1, 2)


        # Pass through GatedCNN
        for i, (conv, conv_gate) in enumerate(zip(self.conv, self.conv_gate)):
            out_temp = out_  # Prepare for Residule
            out_a = conv(out_)
            out_b = conv_gate(out_)
            out_ = out_a * torch.sigmoid(out_b)
            if out_temp.size()[1] == out_.size()[1]:
                out_ += out_temp  # Residule

        # Sum all of the output in the dimension of length
        out_ = out_.sum(dim=-1)

        # Map the output to wanted dimension
        out_ = self.fc(out_)

        return out_


class SelfAttnModel(nn.Module):
    '''
    Input: A sequence 
    Output: A Single unit output


    Transformer Model: https://arxiv.org/pdf/1706.03762.pdf

    This model is the encodding part in the transfomer model.

    '''

    def __init__(self, args):
        super().__init__()


        ## The encoder part of the transformer
        self.encoder = Encoder(
            n_src_vocab=args.vocab_size, len_max_seq=args.SelfAttn_LenMaxSeq,
            d_word_vec=args.SelfAttn_WordVecDim, d_model=args.SelfAttn_ModelDim, d_inner=args.SelfAttn_FFInnerDim,
            n_layers=args.SelfAttn_NumLayers, n_head=args.SelfAttn_NumHead, d_k=args.SelfAttn_KDim, d_v=args.SelfAttn_VDim,
            dropout=args.SelfAttn_Dropout)

        # The final neural network for mapping the output to wanted dimension
        self.fc = nn.Linear(args.SelfAttn_ModelDim, args.textModel_outDim)

        assert args.SelfAttn_ModelDim == args.SelfAttn_WordVecDim, \
            'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, input, lengths=None):
        
        # Pass through the encoder of transformer
        enc_output, *_ = self.encoder(input)
        # Map to wanted output dim
        out_ = self.fc(enc_output)

        return out_.sum(dim=1) # Sum at the length dimension
