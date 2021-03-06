import imp
from os import stat
from time import time
from tkinter.tix import Tree
from turtle import clear
from xmlrpc.client import Boolean
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
from tensorflow.python.ops.gen_math_ops import mul
import tensorflow.keras.regularizers as regularizers
from env import *
from game import n_channels
from tqdm import tqdm
from time import time
from IPython.display import clear_output
from stockfish import Stockfish
from mcts import *

stockfish_v_elo = Stockfish(path=STOCKFISH_PATH)



import warnings
warnings.filterwarnings('ignore')

# https://arxiv.org/abs/2111.09259


def revert_board(board):
    """Changes the side of the board to exchange white's and black's piece."""
    b = board.numpy().copy()

    b[:,:4,:6] = board[:,8:3:-1,6:12]
    b[:,4:,:6] = board[:,3::-1,6:12]

    b[:,:4,6:12] = board[:,8:3:-1,:6]
    b[:,4:,6:12] = board[:,3::-1,:6]

    b[:,:,12:14] = board[:,:,14:16]
    b[:,:,14:16] = board[:,:,12:14]

    return tf.convert_to_tensor(b)

def revert_prediction(move):
    n_move = ''
    for i in range(0,len(move)):
        if i in [1,3]:
            n_move += str(8-int(move[i])+1)
        else:
            n_move += move[i]
    return n_move

def reverse_fen(fen):
    l = fen.split(' ')[0].split('/')[::-1]
    n_fen = ''
    for el in l:
        for i in range(len(el)):
            if el[i].upper() == el[i]:
                n_fen += el[i].lower()
            else:
                n_fen += el[i].upper()
        n_fen += '/'

    n_fen = n_fen[:-1] + ' '
    spfen = fen.split(' ')

    if spfen[1] == 'b':
        n_fen += 'w'
    else:
        n_fen += 'b'
    n_fen += ' '

    if 'k' in spfen[2]:
        n_fen += 'K'
    if 'q' in spfen[2]:
        n_fen += 'Q'
    if 'K' in spfen[2]:
        n_fen += 'k'
    if 'Q' in spfen[2]:
        n_fen += 'q'
    if '-' in spfen[2]:
        n_fen += '-'
    
    n_fen += ' '
    if spfen[3] == '-':
        n_fen += '-'
    else:
        n_fen += spfen[3][0] + str(8-int(spfen[3][1]) +1)

    n_fen += ' ' + spfen[-2] +' '+ spfen[-1]

    return n_fen    

class DeepQ():

    def __init__(self, env, dropout_rate=0.2, n_residual = 1, n_channels = 128):
        self.env = env

        fen_mirror = reverse_fen(self.env.board.fen())
        self.mirror_env = Chess_env(fen_mirror)

          # Experience replay buffers
        self.loss_q_history = []
        self.loss_v_history = []
        self.target_move = []

        self.running_reward = 0
        self.episode_count = 0
        self.frame_count = 0

        self.dropout_rate = dropout_rate

        self.head = self.create_head(n_residual, n_channels)
        self.head_target = self.create_head(n_residual, n_channels)

        self.target_model = self.create_q_model(self.head_target)  #This is the model against which our model will play
        self.model = self.create_q_model(self.head)

    def residual_block(self, x, n_channels):
        x_skip = x
        x = layers.Conv2D(n_channels,3, activation="linear",padding='same')(x)   
        x = layers.Activation(keras.activations.relu)(x)
        x = layers.Dropout(rate=self.dropout_rate)(x)

        x = layers.Conv2D(n_channels,3, activation="linear",padding='same')(x)   
        x = x + x_skip
        x = layers.Activation(keras.activations.relu)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dropout(rate=self.dropout_rate)(x)

        return x

    def create_head(self,n_residual, n_channel):
        inputs = layers.Input(shape=(8, 8, n_channels,))

        x = layers.Conv2D(n_channel,3, activation="linear",padding='same')(inputs)   
        x = layers.BatchNormalization()(x)
        x = layers.Activation(keras.activations.relu)(x)
        x = layers.Dropout(rate=self.dropout_rate)(x)

        for _ in range(n_residual):
            x = self.residual_block(x, n_channel)
        
        return keras.Model(inputs=inputs, outputs=x)

    def create_q_model(self,head):

        inputs = layers.Input(shape=(8, 8, n_channels,))

        x = head(inputs)

        out_policy = layers.Conv2D(256,1, activation="relu", padding='same')(x)
        out_policy = layers.Dropout(rate=self.dropout_rate)(out_policy)
        # 7 horizontal moves left and right, 7 vertical moves up and down, 7 diagonal NW, NE, SW, SE, 8 knight moves, 3 promotions
        out_policy = layers.Conv2D(73,1, activation="softmax", padding='same')(out_policy)

        out = layers.Conv2D(1,1, activation="relu")(x)
        out = layers.Dropout(rate=self.dropout_rate)(out)
        out = layers.Flatten()(out)
        out = layers.Dense(1, activation="tanh", use_bias = False)(out)

        out = out

        return keras.Model(inputs=inputs, outputs=[out,out_policy])

    def create_mask_output(self, moves):
        """Create masks to evaluate only the moves that are legal, applied to the policy output of the layer"""
        mask = np.zeros((8,8,73))
        moves = list(moves)
        for m in moves:
            m=str(m)
            dic = {['a','b','c','d','e','f','g','h'][i] : i for i in range(8)}
            x0 = dic[m[0]]
            x1 = dic[m[2]]
            y0 = int(m[1])-1
            y1 = int(m[3])-1
            #horizontal
            if y0 == y1:
                if x0>x1:
                    mask[x0,y0,x0-x1 + 7*0 -1] = 1
                else:
                    mask[x0,y0,x1-x0 + 7*1 -1] = 1
            #Vertical
            elif x0 == x1:
                if y0> y1:
                    mask[x0,y0,y0-y1 + 7*2 -1] = 1
                else:
                    mask[x0,y0,y1-y0 + 7*3 -1] = 1  
            #diag
            elif abs(x0-x1) == abs(y0-y1):
                #NW
                if x1-x0<0 and y1-y0>0:
                    mask[x0,y0,x0-x1 + 7*4 -1] = 1
                #NE
                elif x1-x0>0 and y1-y0>0:
                    mask[x0,y0,x1-x0 + 7*5 -1] = 1
                #SW 
                elif x1-x0<0 and y1-y0<0:
                    mask[x0,y0,x0-x1 + 7*6 -1] = 1 
                #SE
                elif x1-x0>0 and y1-y0<0:
                    mask[x0,y0,x1-x0 + 7*7 -1] = 1
            #Knights
            else:
                if x1-x0==1 and y1-y0==2:  
                    mask[x0,y0,7*8] = 1

                elif x1-x0==2 and y1-y0==1:  
                    mask[x0,y0,7*8+1] = 1 

                elif x1-x0==2 and y1-y0==-1:  
                    mask[x0,y0,7*8+2] = 1

                elif x1-x0==1 and y1-y0==-2:  
                    mask[x0,y0,7*8+3] = 1

                elif x1-x0==-1 and y1-y0==-2:  
                    mask[x0,y0,7*8+4] = 1  

                elif x1-x0==-2 and y1-y0==-1:  
                    mask[x0,y0,7*8+5] = 1 

                elif x1-x0==-2 and y1-y0==-1:  
                    mask[x0,y0,7*8+6] = 1 

                elif x1-x0==-1 and y1-y0==-2:  
                    mask[x0,y0,7*8+7] = 1  
            if m[-1] == 'n':
                if x0 == x1:
                    mask[x0,y0,8*8] = 1 
                elif x1-x0 == 1:
                    mask[x0,y0,8*8+1] = 1 
                elif x1-x0 == -1:
                    mask[x0,y0,8*8+2] = 1     
            if m[-1] == 'b':
                if x0 == x1:
                    mask[x0,y0,8*8+3] = 1 
                elif x1-x0 == 1:
                    mask[x0,y0,8*8+4] = 1 
                elif x1-x0 == -1:
                    mask[x0,y0,8*8+5] = 1 
            if m[-1] == 'r':
                if x0 == x1:
                    mask[x0,y0,8*8+6] = 1 
                elif x1-x0 == 1:
                    mask[x0,y0,8*8+7] = 1 
                elif x1-x0 == -1:
                    mask[x0,y0,8*8+8] = 1 
        return mask

    def translate_policy_ind_move(self,i,j,k, env = None):
        if env == None:
            env = self.env
        dic = ['a','b','c','d','e','f','g','h']
        j = j+1
        move = dic[i] + str(j)
        if k < 8*7:
            length_depl = k%7 +1
            if k//7 == 0:
                move += dic[i-length_depl] + str(j)
            if k//7 == 1:
                move += dic[i+length_depl] + str(j)
            if k//7 == 2:
                move += dic[i] + str(j-length_depl)  
            if k//7 == 3:
                move += dic[i] + str(j+length_depl)  
            if k//7 == 4:
                move += dic[i- length_depl] + str(j+length_depl) 
            if k//7 == 5:
                move += dic[i+ length_depl] + str(j+length_depl)            
            if k//7 == 6:
                move += dic[i- length_depl] + str(j-length_depl) 
            if k//7 == 7:
                move += dic[i+ length_depl] + str(j-length_depl) 
        elif k >= 8*7 and k <8*8:
            knights_move = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]
            move += dic[i+ knights_move[k-8*7][0]] + str(j + knights_move[k-8*7][1])

        if k ==8*8:
            move += dic[i] + str(j+1) + 'n'
        if k ==8*8+1:
            move += dic[i+1] + str(j+1) + 'n'
        if k ==8*8+2:
            move += dic[i-1] + str(j+1) + 'n'
        if k ==8*8+3:
            move += dic[i] + str(j+1) + 'b'
        if k ==8*8+4:
            move += dic[i+1] + str(j+1) + 'b'
        if k ==8*8+5:
            move += dic[i-1] + str(j+1) + 'b'
        if k ==8*8+6:
            move += dic[i] + str(j+1) + 'r'
        if k ==8*8+7:
            move += dic[i+1] + str(j+1) + 'r'
        if k ==8*8+8:
            move += dic[i-1] + str(j+1) + 'r'

        if move[-1] == '8' and env.board.piece_at(chess.parse_square(move[:2])).piece_type == 1:
            move += 'q'
        return move

    def predict_move_prob(self, envs = None, white = True, target = False, eps=0.01):
        if not white:
            for i,env in enumerate(envs):
                fen = env.board.fen()
                fen_mirror = reverse_fen(fen)
                env = Chess_env(fen_mirror)
                envs[i] = env
        inp = []
        mask = []
        for env_ in envs:
            inp.append(env_.board_feat.board)
            mask.append(self.create_mask_output(env_.board.legal_moves))

        inp = tf.convert_to_tensor(np.array(inp))
        mask = tf.convert_to_tensor(np.array(mask))
        if target:
            full_pred = self.target_model.predict(inp)
        else:
            full_pred = self.model.predict(inp)

        preds = full_pred[1]
        if white:
            q_value = full_pred[0]
        else : 
            q_value = -full_pred[0]

        preds = (preds+eps) * mask 
        outs_m = []
        outs_p = []
        for c in range(preds.shape[0]):
            outs_m.append([])
            outs_p.append([])

        indices = np.argwhere(preds>0.)
        sums = [np.sum(preds[c]) for c in range(preds.shape[0])]
        
        for c,i,j,k in indices:
            m = self.translate_policy_ind_move(i,j,k, env = envs[c])
            if not white:
                m = revert_prediction(m)
            outs_m[c].append(m)

            if not sums[c] == 0:
                outs_p[c].append(preds[c,i,j,k]/sums[c])
            else:
                outs_p[c].append(0)
        return outs_m, outs_p, q_value

    def predict_move_to_play(self, target = False, env = None, white = True):
        if env == None:
            env = self.env

        if not white:
            fen = env.board.fen()
            fen_mirror = reverse_fen(fen)
            self.mirror_env = Chess_env(fen_mirror)
            env = self.mirror_env
        
        inp = tf.convert_to_tensor(env.board_feat.board.reshape(-1,8,8,16))
        mask = self.create_mask_output(env.board.legal_moves)
        if target:
            preds = self.target_model.predict(inp)[1]
        else:
            preds = self.model.predict(inp)[1]

        preds = preds * mask
        _,i,j,k = np.unravel_index(np.argmax(preds), preds.shape)

        move = self.translate_policy_ind_move(i,j,k, env = env)

        if move[-1] == '8' and env.board.piece_at(chess.parse_square(move[:2])).piece_type == 1:
            move += 'q'
        if not white:
            move = revert_prediction(move)

        if mask[i,j,k] == 0:
            move = np.random.choice(list(self.env.board.legal_moves))

        return move

    def predict_move_to_play_MCTS(self, env=None, n_iterations = 300, white = True,):
        if env == None:
            env = self.env
        m = MCTS(self, env, {'num_simulations':n_iterations})
        if white:
            x = m.run(env.board_feat.board, 1)
        else:
            x = m.run(env.board_feat.board, -1)
        return x.select_action(0)

    def train(
            self, 
            max_epoch, 
            epsilon_random_frames = 20000,
            epsilon_greedy_frames = 1000000.0,
            epsilon = 1.,
            epsilon_min = 0.1,
            batch_size = 32,
            max_steps_per_episode = 100,
            learning_rate = 1e-2,
            MCTS_iterations = 300,
            update_target = 100,
            jupyter=True,
            name='model',
            elo = 800
        ):
        """
        Max epochs : maximum number of errors
        epsilon_random_frames : Number of frames to take random action and observe output
        epsilon_greedy_frames :Number of frames for exploration
        max_memory_length : Maximum replay length
        update_after_actions : Train the model after ? actions
        update_target_network : How often to update the target network
        epsilon : Epsilon greedy parameter
        epsilon min : Minimum epsilon greedy parameter
        gamma : Discount factor for past rewards
        batch_size : Number of games used in an epoch (sample 32 positions from each game)
        """

        env = self.env
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0)

        epsilon_interval = (
            epsilon - epsilon_min
        )  # Rate at which to reduce chance of random action being taken

        loss_function = tf.keras.losses.MeanSquaredError()
        loss_function_move = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        batch = {
            "state" : [],
            "reward" : [],
            "target_move": []
        }
        n_new_run = 0

        while self.episode_count<=max_epoch:  # Run until solved
            state,_ = env.reset()
            print(env.board)
            n_new_run += 1
            
            batch['state'].append([])
            batch['target_move'].append([])
            batch['reward'].append(0)
            current_wins = 0
            for _ in tqdm(range(1, max_steps_per_episode)):                
                self.frame_count += 1
                actions = env.get_possible_actions()

                # Use epsilon-greedy for exploration
                if self.frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                    # Take random action
                    action = np.random.randint(0,len(actions))
                    move = actions[action]
                else:
                    # Predict action Q-values
                    # From environment state
                    move = self.predict_move_to_play_MCTS(env= env, n_iterations= MCTS_iterations, white=True)


                # Decay probability of taking random action
                epsilon -= epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)

                # Apply the sampled action in our environment
                action = str(move)
                state_next, reward, done, _ = env.step(action)

                if not done:
                    actions = env.get_possible_actions()

                    stockfish_v_elo.set_elo_rating(elo)
                    stockfish_v_elo.set_fen_position(env.board.fen())
                    move = stockfish_v_elo.get_best_move()

                    state_next, reward, done, _ = env.step(move)    
                    if done:
                        print('Lost')
                        print(current_wins/len(batch['reward']))
                        reward = -1
                else:
                    print('Won')
                    current_wins += 1
                    reward = 1
                    print(current_wins/len(batch['reward']))
                    


                batch['state'][-1].append(state)
                batch['target_move'][-1].append(self.create_mask_output([action]))
                batch['reward'][-1] = reward
                state = state_next
                
                if done:
                    break


            if len(batch['reward']) >= batch_size and n_new_run%(batch_size//2) == 0:
                state_sample = []
                rewards_sample = []
                target_move_sample = []

                for i in range(batch_size):
                    indices = [k for k in range(len(batch['state'][i]))]
                    np.random.shuffle(indices)
                    indices = indices[:32]

                    state_sample += [batch['state'][i][j] for j in indices]
                    target_move_sample += [batch['target_move'][i][j] for j in indices]
                    rewards_sample += [batch['reward'][i]]*len(indices)
                    
                rewards_sample = np.array(rewards_sample)
                target_move_sample = np.array(target_move_sample).astype('float32')
                state_sample = np.array(state_sample)

                bs = state_sample.shape[0]

                final_reward = tf.convert_to_tensor(rewards_sample)
                # CHANGE TARGET TO BELLMAN EQU / Q_VALUE ....

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    tensor_ss = tf.convert_to_tensor(np.array(state_sample))
                    target_move_sample_t = tf.convert_to_tensor(target_move_sample)

                    pred = self.model(tensor_ss)

                    pred_move = tf.reshape(pred[1], (bs,-1))
                    target_move_sample_t = tf.reshape(target_move_sample_t, (bs,-1))
                    q_value = pred[0]
                    loss_v = loss_function_move(pred_move, target_move_sample_t)
                    loss_q = loss_function(final_reward, q_value)

                    loss = loss_q + loss_v
                # Backpropagation
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                self.model.save(name)
                loss_q = loss_q.numpy()
                self.loss_q_history.append(np.mean(loss_q))
                loss_v = loss_v.numpy()
                self.loss_v_history.append(np.mean(loss_v))


                if n_new_run>10*batch_size:
                    batch = {
                                "state" : [],
                                "reward" : [],
                                "target_move": []
                            }
                    n_new_run = 0

                # Log details
                template = "Episode {}, frame count {}"
                print(template.format(self.episode_count,self.frame_count))
                if jupyter:
                    clear_output()

                fig,axes = plt.subplots(1,2, figsize=(15,5))

                axes[0].plot(self.loss_q_history)
                axes[0].set_title('Policy Loss')

                axes[1].plot(self.loss_v_history)
                axes[1].set_title('Value Loss')

                plt.show()

            self.episode_count += 1
                        # update the the target network with new weights
            if self.episode_count % update_target == 0:
                self.target_model.set_weights(self.model.get_weights())

    def pre_train(
            self, 
            max_epoch, 
            batch_size = 32,
            max_steps_per_episode = 100,
            learning_rate = 1e-2,
            update_target = 100,
            random_best_action = 0.8,
            jupyter=True,
            n_top_move = 5,
            name='model',
            length_train_set=256*2,
            length_test_set = 64*2,
        ):
        """
        Max epochs : maximum number of errors
        epsilon_random_frames : Number of frames to take random action and observe output
        epsilon_greedy_frames :Number of frames for exploration
        max_memory_length : Maximum replay length
        update_after_actions : Train the model after ? actions
        update_target_network : How often to update the target network
        epsilon : Epsilon greedy parameter
        epsilon min : Minimum epsilon greedy parameter
        gamma : Discount factor for past rewards
        batch_size : Number of games used in an epoch (sample 32 positions from each game)
        """

        env = self.env
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0)

        loss_function = tf.keras.losses.MeanSquaredError()
        loss_function_move = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


        dataset_train = {
            "state" : [],
            "reward" : [],
            "target_move": []
        }
        dataset_test = {
            "state" : [],
            "reward" : [],
            "target_move": []
        }
        progress_bar = tqdm(range(length_train_set))

        while len(dataset_train['reward'])<length_train_set:
            state,_ = env.reset()
            for _ in range(1, max_steps_per_episode):
                actions = env.get_possible_actions()

                stockfish.set_fen_position(env.board.fen())
                reward = np.clip(stockfish.get_evaluation()['value']/500, -1, 1,)
                best_moves = [s['Move'] for s in stockfish.get_top_moves(n_top_move)]

                # Take random action
                action = np.random.randint(0,len(actions))
                move = actions[action]
                action = str(move)
                if np.random.random()<random_best_action:
                    action = stockfish.get_best_move_time(10)

                state_next, _, done, _ = env.step(action)

                if not done:
                    actions = env.get_possible_actions()
                    move = np.random.choice(actions)
                    if np.random.random()<random_best_action:
                        stockfish.set_fen_position(env.board.fen())
                        move = stockfish.get_best_move_time(10)
                    state_next, _, done, _ = env.step(move) 
                if done:
                    break
                if np.random.random() < 0.2:   
                    progress_bar.update(1)
                    dataset_train['state'].append(state)
                    dataset_train['target_move'].append(self.create_mask_output(best_moves))
                    dataset_train['reward'].append(reward)
                state = state_next


        progress_bar = tqdm(range(length_test_set))
        while len(dataset_test['reward'])<length_test_set:
            state,_ = env.reset()
            for _ in range(1, max_steps_per_episode):
                actions = env.get_possible_actions()

                stockfish.set_fen_position(env.board.fen())
                reward = np.clip(stockfish.get_evaluation()['value']/500, -1, 1,)
                best_moves = [s['Move'] for s in stockfish.get_top_moves(n_top_move)]

                # Take random action
                action = np.random.randint(0,len(actions))
                move = actions[action]
                action = str(move)
                if np.random.random()<random_best_action:
                    action = stockfish.get_best_move_time(10)

                state_next, _, done, _ = env.step(action)

                if not done:
                    actions = env.get_possible_actions()
                    move = np.random.choice(actions)
                    if np.random.random()<random_best_action:
                        stockfish.set_fen_position(env.board.fen())
                        move = stockfish.get_best_move_time(10)
                    state_next, _, done, _ = env.step(move) 
                if done:
                    break
                if np.random.random() < 0.05:   
                    progress_bar.update(1)
                    dataset_test['state'].append(state)
                    dataset_test['target_move'].append(self.create_mask_output(best_moves))
                    dataset_test['reward'].append(reward)
                state = state_next


        for _ in tqdm(range(max_epoch)):  # Run until solved
            print(f"Running episode { self.episode_count}/{max_epoch}")
            for idx in range(0,length_train_set, batch_size):
                state_sample = dataset_train['state'][idx:idx+batch_size]
                rewards_sample = dataset_train['reward'][idx:idx+batch_size]
                target_move_sample = dataset_train['target_move'][idx:idx+batch_size]
                    
                rewards_sample = np.array(rewards_sample)
                target_move_sample = np.array(target_move_sample).astype('float32')
                state_sample = np.array(state_sample)

                bs = state_sample.shape[0]

                final_reward = tf.convert_to_tensor(rewards_sample)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    tensor_ss = tf.convert_to_tensor(np.array(state_sample))
                    target_move_sample_t = tf.convert_to_tensor(target_move_sample)

                    pred = self.model(tensor_ss)

                    pred_move = tf.reshape(pred[1], (bs,-1))
                    target_move_sample_t = tf.reshape(target_move_sample_t, (bs,-1))
                    q_value = pred[0]

                    loss_v = loss_function_move(pred_move, target_move_sample_t)
                    loss_q = loss_function(final_reward, q_value)

                    loss = loss_q + loss_v
                # Backpropagation
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
                loss_q = loss_q.numpy()
                self.loss_q_history.append(np.mean(loss_q))
                loss_v = loss_v.numpy()
                self.loss_v_history.append(np.mean(loss_v))

            self.model.save(name)
            if jupyter:
                clear_output()

            fig,axes = plt.subplots(1,3, figsize=(15,5))

            axes[0].plot(self.loss_q_history)
            axes[0].set_title('Policy Loss')

            axes[1].plot(self.loss_v_history)
            axes[1].set_title('Value Loss')

            if len(self.loss_q_history)>20:
                axes[0].plot(
                    np.arange(len(self.loss_q_history))[20:],
                    np.mean(
                        np.array([self.loss_q_history[i:-20+i] for i in range(20)]),
                        axis = 0
                        )
                    )
                axes[1].plot(
                    np.arange(len(self.loss_v_history))[20:],
                    np.mean(
                        np.array([self.loss_v_history[i:-20+i] for i in range(20)]),
                        axis = 0
                        )
                    )

            # axes[2].plot([-1,1],[-1,1])
            
            gt = final_reward.numpy()
            ev = q_value.numpy()
            for x,y in zip(gt,ev):
                axes[2].scatter(x,y, color='r')
            axes[2].set_xlabel("Ground Truth evaluation")
            axes[2].set_ylabel("Model evaluation")
            plt.show()
            






