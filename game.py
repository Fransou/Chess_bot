import numpy as np
import matplotlib.pyplot as plt
n_channels = 16


class ChessBoard:

    def __init__(self,) -> None:
        self.board = np.zeros((8,8,16))
        self.other_feat = [1,1,1,1,1] #castle,long,castle, who'sturn
        self.transcription = {
            "p": 0,
            "n": 1,
            "b": 2,
            "r": 3,
            "q": 4,
            "k": 5,
            "P": 6,
            "N": 7,
            "B": 8,
            "R": 9,
            "Q": 10,
            "K": 11,
        }
        self.revers_transcription = {
            0:"p",
            1:"n",
            2:"b",
            3:"r",
            4:"q",
            5:"k",
        }
    
    def reset(self):
        self.board = np.zeros((8,8,16))


    def translate(self,fen):
        self.reset()
        i=0
        j =7
        fen_splitted = fen.split(' ')
        for l in fen_splitted[0]:
            if l in self.transcription.keys():
                self.board[i, j, self.transcription[l]] = 1
                i+=1
            elif l == "/":
                j-=1
                i=0
            else:
                i+= int(l)

        if not 'K' in fen_splitted[2]:
            self.board[:,:,12] = 1
        if not 'Q' in fen_splitted[2]:
            self.board[:,:,13] = 1
        if not 'k' in fen_splitted[2]:
            self.board[:,:,14] = 1
        if not 'q' in fen_splitted[2]:
            self.board[:,:,15] = 1

