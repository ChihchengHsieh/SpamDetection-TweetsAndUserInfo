class Constants():
    ''' The Constants for the text '''
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    PAD_WORD = '<PAD>'
    UNK_WORD = '<UNK>'
    SOS_WORD = '<SOS>'
    EOS_WORD = '<EOS>'


specialTokens = {
    Constants.PAD_WORD: Constants.PAD,
    Constants.UNK_WORD: Constants.UNK,
    Constants.SOS_WORD: Constants.SOS,
    Constants.EOS_WORD: Constants.EOS,
}

specialTokenList = [Constants.PAD_WORD,
                    Constants.UNK_WORD,
                    Constants.SOS_WORD,
                    Constants.EOS_WORD]