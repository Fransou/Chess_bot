import numpy as np
import matplotlib.pyplot as plt
import keras as k

from keras.models import Sequential
import keras.layers as layers
from keras import backend as K




class ChessBoard:

    def __init__(self,) -> None:
        self.board = np.zeros((8,8,7))
        self.other_feat = [1,1,1,1,1] #castle,long,castle, who'sturn
        self.transcription = {
            "p": 0,
            "n": 1,
            "b": 2,
            "r": 3,
            "q": 4,
            "k": 5,
            'enp':6
        }

    
    def reset(self):
        self.board = np.zeros((8,8,7))
        self.other_feat = [1,1,1,1,1]

    def translate(self,fen):
        self.reset()
        i=0
        j =7
        fen_splitted = fen.split(' ')
        for l in fen_splitted[0]:
            if l.lower() in self.transcription.keys():
                if l == l.lower():
                    self.board[i, j, self.transcription[l.lower()]] = 1
                else:
                    self.board[i, j, self.transcription[l.lower()]] = -1
                i+=1
            elif l == "/":
                j-=1
                i=0
            else:
                i+= int(l)

        self.other_feat[-1] = 2*int(fen_splitted[1]=='w')-1

        if not 'K' in fen_splitted[2]:
            self.other_feat[0] = 0
        if not 'Q' in fen_splitted[2]:
            self.other_feat[1] = 0
        if not 'k' in fen_splitted[2]:
            self.other_feat[2] = 0
        if not 'q' in fen_splitted[2]:
            self.other_feat[3] = 0
        if not fen_splitted[3] == '-':
            dic = np.array(['a','b','c','d','e','f','g','h'])
            self.board[np.where(dic == fen_splitted[3][0])[0],int(fen_splitted[3][1]),6] = 1




            

        
