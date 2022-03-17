import imp
import gym
from gym import spaces
import numpy as np
import chess
from game import *
from stockfish import Stockfish


stockfish = Stockfish(path="stockfish_13_win_x64_avx2/stockfish_13_win_x64_avx2")

class Chess_env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, fen = ""):
        super(Chess_env, self).__init__()    # Define action and observation space

        # They must be gym.spaces objects    # Example when using discrete actions:
        # Example for using image as input:
        if fen =="":
            self.board_feat = ChessBoard()
            self.board = chess.Board()
        else:
            self.board = chess.Board(fen)
            self.board_feat = ChessBoard()
            self.board_feat.translate(fen)

        self.current_step = 0
        self.reward = 0

    def _take_action(self, action):
        """Updates the env when the agent choses an action"""
        # Execute one time step within the environment
        self.board.push_san(str(action))

        self.board_feat.translate(self.board.fen())


    def reset(self):
        '''
        Resets the board, plays a random number of moves, and returns observation, multiplicator (1 if the model plays white, -1 else)'''        
        self.board = chess.Board()
        self.board_feat = ChessBoard()

        self.board_feat.translate(self.board.fen())
        self.current_step = 0

        # We play moves randomly

        n_init_moves = np.random.randint(0,10) * 2 +1
        for i in range(n_init_moves):
            if not self.get_possible_actions() == []:
                action = np.random.choice(self.get_possible_actions())
                self._take_action(action)
            else:
                self.reset()
        if self.get_possible_actions() == []:
            return self.reset()
        n_pieces = np.sum(self.board_feat.board[:,:,:12])

        while n_pieces > 6 or n_init_moves%2 == 1:
            stockfish.set_fen_position(self.board.fen())
            move = stockfish.get_best_move()
            n_pieces = np.sum(self.board_feat.board[:,:,:12])

            _,_,done,_ = self.step(move)

            n_init_moves += 1 

            if done:
                return self.reset()
        return self._next_observation(), 2*int(n_init_moves%2 ==0)-1

    def reset_board_feat(self):
        self.board_feat.translate(self.board.fen())


    def _next_observation(self):
        self.board_feat.translate(self.board.fen())
        return self.board_feat.board

    def step(self, action):
        """One play (2 step to come back to the same player)"""
        self._take_action(action)
        fen = self.board.fen().split(' ')[0]
        self.current_step += 1
        done = False
        reward = 0

        if self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition() or self.board.is_insufficient_material() or self.get_possible_actions() == []:
            done = True
        
        if self.board.is_checkmate():
            reward = 1
            done = True

        if not done:
            obs = self._next_observation()
        else:
            obs = np.zeros((8,8,n_channels))

        self.reset_board_feat()
        return obs, reward, done, {}


    def get_possible_actions(self):
        return list(self.board.legal_moves)

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(self.board)



    def generate_input_from_action(self, action):
        inter_board = chess.Board(self.board.fen())
        inter_board.push(action)

        self.board_feat.translate(inter_board.fen())

        return self.board_feat.board

