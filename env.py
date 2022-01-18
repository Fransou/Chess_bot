import imp
import gym
from gym import spaces
import numpy as np
import chess
from game import *
max_states = 80

class Chess_env(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Chess_env, self).__init__()    # Define action and observation space

        # They must be gym.spaces objects    # Example when using discrete actions:

        self.action_space = spaces.Discrete(max_states)    

        # Example for using image as input:

        self.observation_space = spaces.Tuple(
            (spaces.Box(low=0, high=255, shape=
                        (8, 8, n_channels), dtype=np.uint8),
            spaces.Box(low=0, high=1, shape =(4,1), dtype=np.uint8)
            )
        )
        self.reward_range = (-1, 1) 

        self.board_feat = ChessBoard()
        self.board = chess.Board()
        self.current_step = 0
        self.reward = 0

    def _take_action(self, action):
        """Updates the env when the agent choses an action"""
        # Execute one time step within the environment
        self.board.push(action)

        self.board_feat.translate(self.board.fen())


    def reset(self):
        '''
        Resets the board, plays a random number of moves, and returns observation, multiplicator (1 if the model plays white, -1 else)'''        
        self.board = chess.Board()
        self.board_feat = ChessBoard()

        self.board_feat.translate(self.board.fen())
        self.current_step = 0

        # We play moves randomly

        n_init_moves = np.random.randint(0,6)
        for i in range(n_init_moves):
            action = np.random.choice(self.get_possible_actions())
            self._take_action(action)

        return self._next_observation(), 2*int(n_init_moves%2 ==0)-1

    def reset_board_feat(self):
        self.board_feat.translate(self.board.fen())


    def _next_observation(self):
        """
        tuple : (obs0,obs1,multiplicator)

        obs0 : Board
        obs1 : Special features (can castle)
        multiplicator : scalar, whether it black's or white's turn
        """

        possible_actions = list(self.board.legal_moves)
        obs0 = []
        obs1 = []
        mult = []

        for action in possible_actions:
            obs = self.generate_input_from_action(action)
            obs0.append(obs[0])
            obs1.append(obs[1])
            mult.append(obs[2])

        for i in range(max_states - len(possible_actions)):
            obs0.append(np.zeros((8,8,n_channels)))
            obs1.append(np.zeros((4,)))
            mult.append([0])

        obs0 = np.array(obs0)
        obs1 = np.array(obs1)
        mult = np.array(mult)

        self.reset_board_feat()

        return [obs0,obs1,mult]

    def step(self, action):
        """One play (2 step to come back to the same player)"""
        self._take_action(action)
        fen = self.board.fen().split(' ')[0]
        self.current_step += 1
        done = False
        reward = 0

        
        reward = (fen.count('Q')- fen.count('q')) * 9/10 + 5/10 * (fen.count('R')- fen.count('r')) + 3/10 * (fen.count('B') + fen.count('N') - fen.count('n') - fen.count('b'))        

        if self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition() or self.board.is_insufficient_material() or self.get_possible_actions() == []:
            reward =0
            done = True
        
        if self.board.is_checkmate():
            reward += 10
            done = True


        if not done:
            obs = self._next_observation()
        else:
            obs = np.zeros((max_states,8,8,n_channels)),np.zeros((max_states,4,)),np.zeros((max_states,))
        self.reset_board_feat()
        return obs, reward/10, done, {}


    def get_possible_actions(self):
        return list(self.board.legal_moves)

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(self.board)



    def generate_input_from_action(self, action):
        inter_board = chess.Board(self.board.fen())
        inter_board.push(action)

        self.board_feat.translate(inter_board.fen())

        return self.board_feat.board, np.array(self.board_feat.other_feat[:-1]),  np.array([self.board_feat.other_feat[-1]])


    

    def create_pretraining_dataset(self,n_games, n_steps, prob):
        X_b = []
        X_of = []
        y = []

        list_fens = []

        for g in range(n_games):
            _ = self.reset()
            for step in range(n_steps):
                action = np.random.choice(list(self.board.legal_moves))
                _,_,done,_ = self.step(action)

                if prob>np.random.rand():
                    if not self.board.fen().split(' ')[0] in list_fens:
                        list_fens.append(self.board.fen().split(' ')[0])
                        obs = self._next_observation()
                        y.append(
                            np.sum(
                                np.sum(obs[0], axis=0) * (1-np.sign(np.abs(self.board_feat.board))),
                                axis=2
                            )
                        )
                        X_b.append(self.board_feat.board)
                        X_of.append(self.board_feat.other_feat[:4])
                        
                if done:
                    break

        return [np.array(X_b), np.array(X_of)], np.array(y)



