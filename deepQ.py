import imp
from os import stat
from time import time
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
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.n_action_history = []
        self.n_action_next_history = []
        self.loss_history = []
        self.mask_history = []
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
        x = layers.BatchNormalization()(x)
        x = layers.Activation(keras.activations.relu)(x)
        x = layers.Dropout(rate=self.dropout_rate)(x)


        x = layers.Conv2D(n_channels,3, activation="linear",padding='same')(x)   
        x = layers.BatchNormalization()(x)
        x = x + x_skip
        x = layers.Activation(keras.activations.relu)(x)
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
        out_policy = layers.Conv2D(73,1, activation="softmax", padding='same')(x)

        out = layers.Conv2D(1,1, activation="relu")(x)
        out = layers.Dropout(rate=self.dropout_rate)(out)
        
        out = layers.Flatten()(out)
        out = layers.Dense(1, activation="tanh")(out)

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

    def translate_policy_ind_move(self,i,j,k):
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
        return move

    def predict_move_prob(self, envs = None, white = True, target = False):
        if not white:
            for i,env in enumerate(envs):
                fen = env.board.fen()
                fen_mirror = reverse_fen(fen)
                self.mirror_env = Chess_env(fen_mirror)
                env = self.mirror_env
                envs[i] = env
        inp = []
        mask = []
        for env in envs:
            inp.append(env.board_feat.board)
            mask.append(self.create_mask_output(env.board.legal_moves))

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
        preds = preds * mask

        outs_m = []
        outs_p = []
        for channel in range(preds.shape[0]):
            out_move = []
            out_p = []
            norm = np.sum(preds[channel])
            indices = np.argwhere(preds[channel]>0)
            for i,j,k in indices:
                m = self.translate_policy_ind_move(i,j,k)
                if not white:
                    m = revert_prediction(m)
                out_move.append(m)
                out_p.append(preds[channel,i,j,k] / norm)

            if out_p == [] or np.sum(out_p) == 0:
                moves = env.board.legal_moves
                out_p = []
                out_move = []
                for m in moves:
                    m= str(m)
                    if not white:
                        m = revert_prediction(m)
                    out_move.append(m)
                    out_p.append(1/len(list(moves)))

            outs_m.append(out_move)
            outs_p.append(out_p)

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

        move = self.translate_policy_ind_move(i,j,k)

        if move[-1] == '8' and env.board.piece_at(chess.parse_square(move[:2])).piece_type == 1:
            move += 'q'
        if not white:
            move = revert_prediction(move)

        if mask[i,j,k] == 0:
            move = np.random.choice(list(self.env.board.legal_moves))

        return move

    def predict_move_to_play_MCTS(self,depth, env=None, n_iterations = 300, target = False, white = True,):

        if env == None:
            env = self.env
        fen = env.board.fen()
       
        envs = []
        for i in range(n_iterations):
            envs.append(Chess_env(fen))
        paths = []
        dones = [False for i in range(n_iterations)]
        q_val = [0] * n_iterations
        for k in range(depth):
            n_envs = []
            for i in range(n_iterations):
                n_envs.append(Chess_env(envs[i].board.fen()))
            outs_m, outs_p, q_value = self.predict_move_prob(n_envs, Boolean((white+k)%2), target = target)
            for i in range(n_iterations):
                if not dones[i]:
                    m = np.random.choice(outs_m[i], p = outs_p[i])
                    if k == 0:
                        paths.append(m)
                    _, _, done, _ = envs[i].step(m)
                    dones[i] = done
                    q_val[i] = q_value[i]


        q_val = np.array(q_val).reshape(-1)
        results = {m : [] for m in paths}
        for m, q_v in zip(paths, q_val):
            results[m].append(q_v)

        max_val = -np.inf
        for m in results.keys():
            val = np.mean(results[m])
            if val > max_val:
                move = m
                max_val = val
        return move




    def train(
            self, 
            max_epoch, 
            epsilon_random_frames = 20000,
            epsilon_greedy_frames = 1000000.0,
            max_memory_length = 5000,
            update_after_actions = 4,
            update_target_network = 1000,
            epsilon = 1.,
            epsilon_min = 0.1,
            gamma = 0.99,
            batch_size = 32,
            max_steps_per_episode = 100,
            learning_rate = 1e-2,
            MCTS_depth = 4,
            MCTS_iterations = 300,
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
        batch_size : Size of batch taken from replay buffer
        """

        env = self.env
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

        epsilon_interval = (
            epsilon - epsilon_min
        )  # Rate at which to reduce chance of random action being taken

        loss_function = keras.losses.MSE
        loss_function_move = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        
        while self.episode_count<=max_epoch:  # Run until solved
            state,_ = env.reset()
            episode_reward = 0
            for timestep in tqdm(range(1, max_steps_per_episode)):
                
                self.mask_history.append(self.create_mask_output(self.env.board.legal_moves))

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
                    move = self.predict_move_to_play_MCTS(MCTS_depth, n_iterations=MCTS_iterations,)


                # Decay probability of taking random action
                epsilon -= epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)

                # Apply the sampled action in our environment
                action = str(move)
                state_next, reward, done, _ = env.step(action)

                episode_reward = reward

                if not done:
                    actions = env.get_possible_actions()
                    if self.frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                    # Take random action
                        move = np.random.choice(actions)
                    else:
                        move = self.predict_move_to_play_MCTS(MCTS_depth, n_iterations=MCTS_iterations, target = True, white=False)
                    state_next, reward, done, _ = env.step(move)    
                    if done:
                        reward = -1
                if done:
                    print('reward', reward)             
                # Save actions and states in replay buffer
                self.action_history.append(action)
                self.state_history.append(state)
                self.state_next_history.append(state_next)
                self.done_history.append(done)
                self.rewards_history.append(reward)
                self.n_action_next_history.append(len(env.get_possible_actions()))
                self.target_move.append(self.create_mask_output([action]))

                state = state_next
                # Update every fourth frame and once batch size is over 32
                if self.frame_count % update_after_actions == 0 and len(self.done_history) > batch_size:

                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(self.done_history)), size=batch_size)

                    # Using list comprehension to sample from replay buffer
                    state_sample = [self.state_history[i] for i in indices]
                    state_next_sample = [self.state_next_history[i] for i in indices]
                    rewards_sample = np.array([self.rewards_history[i] for i in indices])
                    action_sample = [self.action_history[i] for i in indices]
                    target_move_sample = np.array([self.target_move[i] for i in indices]).astype('float32')
                    done_sample = tf.convert_to_tensor(
                        [float(self.done_history[i]) for i in indices]
                    )
                    mask_sample = [self.mask_history[i] for i in indices]
                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability

                    future_rewards = []
                    for s in state_next_sample:
                        s = s.reshape(-1,8,8,16)
                        res = self.target_model.predict([s]) 
                        future_rewards.append(res[0])

                    future_rewards = np.array(future_rewards).astype('float32')
                    future_rewards = tf.convert_to_tensor(future_rewards)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + gamma * tf.reduce_max(
                        future_rewards, axis=1
                    )

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(indices,1)
                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        tensor_ss = tf.convert_to_tensor(np.array(state_sample))
                        target_move_sample_t = tf.convert_to_tensor(target_move_sample)
                        pred = self.model(tensor_ss)
                        pred_move = tf.reshape(pred[1], (batch_size,-1))
                        target_move_sample_t = tf.reshape(target_move_sample_t, (batch_size,-1))
                        q_values = pred[0]
                        q_values = tf.reshape(q_values, (-1))
                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss_v = loss_function_move(pred_move, target_move_sample_t)
                        loss_q = loss_function(updated_q_values, q_action)
                        loss = loss_q + loss_v
                    # Backpropagation
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    self.loss_history.append(loss)
                if self.frame_count % update_target_network == 0:
                    # update the the target network with new weights
                    self.target_model.set_weights(self.model.get_weights())
                    # Log details
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(self.running_reward,self.episode_count,self.frame_count))
                    plt.plot(self.loss_history)
                    plt.show()
                # Limit the state and reward history
                if len(self.rewards_history) > max_memory_length:
                    del self.rewards_history[:1]
                    del self.state_history[:1]
                    del self.state_next_history[:1]
                    del self.action_history[:1]
                    del self.done_history[:1]
                    del self.target_move[:1]

                if done:
                    break
            # Update running reward to check condition for solving
            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > 100:
                del self.episode_reward_history[:1]

            self.running_reward = np.mean(self.episode_reward_history)

            self.episode_count += 1

            if self.running_reward > 40:  # Condition to consider the task solved
                print("Solved at episode {}!".format(self.episode_count))
                break


            print("Final configuration : \n", self.env.board)