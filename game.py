import numpy as np
import matplotlib.pyplot as plt

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
        self.revers_transcription = {
            0:"p",
            1:"n",
            2:"b",
            3:"r",
            4:"q",
            5:"k",
            6:'enp'
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

    def generate_fen(self):

        i=0
        j=7
        fen0 = ''
        

        for j in range(7,-1,-1):
            current_i = 0
            for i in range(8):
                if (self.board[i,j] == 0).all():
                    current_i +=1
                else:
                    b_w = self.board[i,j][np.argmax(np.abs(self.board[i,j]))]
                    piece = self.revers_transcription[np.argmax(np.abs(self.board[i,j]))]

                    if b_w == -1:
                        piece = piece.upper()

                    if current_i == 0:
                        fen0 = fen0 + piece
                    else:
                        fen0 = fen0 + str(current_i) + piece

                    current_i = 0
            if current_i>0:
                fen0 = fen0 + str(current_i)
            fen0 = fen0 + '/'

        fen = fen0[:-1] + ' '
        if self.other_feat[-1] == 1:
            fen = fen + 'w'
        else:
            fen = fen + 'b'
        
        fen = fen + ' ' + 'K'*self.other_feat[0]+'Q'*self.other_feat[1]+'k'*self.other_feat[2]+'q'*self.other_feat[3]
        if fen[-1] == ' ':
            fen = fen + '-'

        if (self.board[:,:,6] == 0).all():
            fen = fen + ' -'

        else:
            ind = np.argmax(np.abs(self.board[:,:,6]))
            dic = np.array(['a','b','c','d','e','f','g','h'])
            ind = dic[ind[0]] + str(ind[1])
            fen = fen + ' ' + ind
        
        fen = fen + ' 1 10'
        return fen

 



            

        
