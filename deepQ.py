import imp
from os import stat
from time import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
from tensorflow.python.ops.gen_math_ops import mul
import tensorflow.keras.regularizers as regularizers
from env import *
from game import n_channels

# https://arxiv.org/abs/2111.09259


def revert_board(board):
    """Changes the side of the board to exchange white's and black's piece."""
    b = board.copy()

    b[:,:4,:6] = board[:,8:3:-1,6:12]
    b[:,4:,:6] = board[:,3::-1,6:12]

    b[:,:4,6:12] = board[:,8:3:-1,:6]
    b[:,4:,6:12] = board[:,3::-1,:6]

    b[:,:,12:14] = board[:,:,14:16]
    b[:,:,14:16] = board[:,:,12:14]

    return b

def revert_prediction(move):
    n_move = ''
    for i in range(0,len(move)):
        if i%2==1:
            n_move += str(8-int(move[i]))
        else:
            n_move += move[i]
    return n_move





class DeepQ():

    def __init__(self, env, dropout_rate=0.2, n_residual = 3, n_channels = 128):
        self.env = env
        self.model = None
        self.target_model = None
          # Experience replay buffers
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.n_action_history = []
        self.n_action_next_history = []
        self.mask_history = []
        self.running_reward = 0
        self.episode_count = 0
        self.frame_count = 0

        self.dropout_rate = dropout_rate

        self.head = self.create_head(n_residual, n_channels)
        self.head_target = self.create_head(n_residual, n_channels)

        self.model_target = self.create_q_model(self.head_target)  #This is the model against which our model will play
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

    def create_mask_output(self, env= None):
        """Create masks to evaluate only the moves that are legal, applied to the policy output of the layer"""
        if env == None:
            env = self.env
        moves = list(self.env.board.legal_moves)
        mask = np.zeros((8,8,73))
        for m in moves:
            m=str(m)
            dic = {['a','b','c','d','e','f','g','h'][i] : i for i in range(8)}
            x0 = dic[m[0]]
            x1 = dic[m[2]]
            y0 = int(m[1])-1
            y1 = int(m[3])-1
            #horizontal
            if y0 == y1 and not (len(m) == 5 and m[-1] != 'q'):
                if x0>x1:
                    mask[x0-1,y0,x0-x1 + 7*0 -1] = 1
                else:
                    mask[x0-1,y0,x0-x1 + 7*1 -1] = 1
            #Vertical
            elif x0 == x1:
                if y0> y1:
                    mask[x0,y0,x0-x1 + 7*2 -1] = 1
                else:
                    mask[x0,y0,x0-x1 + 7*3 -1] = 1  
            #diag
            elif abs(x0-x1) == abs(y0-y1):
                #NW
                if x1-x0<0 and y1-y0>0:
                    mask[x0,y0,x0-x1 + 7*4 -1] = 1
                #NE
                elif x1-x0>0 and y1-y0>0:
                    mask[x0,y0,x0-x1 + 7*5 -1] = 1
                #SW 
                elif x1-x0<0 and y1-y0<0:
                    mask[x0,y0,x0-x1 + 7*6 -1] = 1 
                #SE
                elif x1-x0>0 and y1-y0<0:
                    mask[x0,y0,x0-x1 + 7*7 -1] = 1
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
                if y0 == y1:
                    mask[x0,y0,8*8] = 1 
                elif y1-y0 == 1:
                    mask[x0,y0,8*8+1] = 1 
                elif y1-y0 == -1:
                    mask[x0,y0,8*8+2] = 1     
            if m[-1] == 'b':
                if y0 == y1:
                    mask[x0,y0,8*8+3] = 1 
                elif y1-y0 == 1:
                    mask[x0,y0,8*8+4] = 1 
                elif y1-y0 == -1:
                    mask[x0,y0,8*8+5] = 1 
            if m[-1] == 'r':
                if y0 == y1:
                    mask[x0,y0,8*8+6] = 1 
                elif y1-y0 == 1:
                    mask[x0,y0,8*8+7] = 1 
                elif y1-y0 == -1:
                    mask[x0,y0,8*8+8] = 1 
        return mask

    def translate_policy_ind_move(self,i,j,k):
        dic = ['a','b','c','d','e','f','g','h']
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
            move += dic[i+ knights_move[k-8*7][0]] + str(j + i+ knights_move[k-8*7][1])

        if k ==8*8:
            move += dic(i) + str(j+1) + 'n'
        if k ==8*8+1:
            move += dic(i+1) + str(j+1) + 'n'
        if k ==8*8+2:
            move += dic(i-1) + str(j+1) + 'n'
        if k ==8*8+3:
            move += dic(i) + str(j+1) + 'b'
        if k ==8*8+4:
            move += dic(i+1) + str(j+1) + 'b'
        if k ==8*8+5:
            move += dic(i-1) + str(j+1) + 'b'
        if k ==8*8+6:
            move += dic(i) + str(j+1) + 'r'
        if k ==8*8+7:
            move += dic(i+1) + str(j+1) + 'r'
        if k ==8*8+8:
            move += dic(i-1) + str(j+1) + 'r'

        return move

    def predict_move_prob(self, inp):
        mask = self.create_mask_output()
        preds = self.model.predict(inp)[1]

        preds = preds * mask
        out = []

        norm = np.sum(preds)
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                for k in range(preds.shape[2]):
                    if preds[i,j,k]>0:
                        out.append([self.translate_policy_ind_move(i,j,k), preds[i,j,k] / norm])
        return out

    def predict_move_to_play(self, target = False, env = None, white = True):
        if env == None:
            env = self.env

        inp = env.feat_board.board
        mask = self.create_mask_output(env)
        if not white:
            mask = np.flip(mask, axis=1)

        if target:
            preds = self.target_model.predict(inp)[1]
        else:
            preds = self.model.predict(inp)[1]
        preds = preds * mask
        i,j,k = np.argmax(preds)
        return self.translate_policy_ind_move(i,j,k)

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
            max_steps_per_episode = 100
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
        model = self.model
        model_target = self.model_target
        env = self.env
        optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

        epsilon_interval = (
            epsilon - epsilon_min
        )  # Rate at which to reduce chance of random action being taken

        loss_function = keras.losses.Huber() 
        
        # TTTTTTOOOOOOOODDDDDOOOOO : MCTS pour la partie choisir le coup, puis..jy

        while self.episode_count<=max_epoch:  # Run until solved
            state,_ = env.reset()
            episode_reward = 0
            for timestep in range(1, max_steps_per_episode):
                
                self.mask_history.append(self.create_mask_output())

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
                    move = self.predict_move_to_play()


                # Decay probability of taking random action
                epsilon -= epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)

                # Apply the sampled action in our environment
                state_next, reward, done, _ = env.step(move)

                episode_reward = reward

                if not done:
                    actions = env.get_possible_actions()
                    if self.frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                    # Take random action
                        move = np.random.choice(actions)
                    else:
                        t_sn = tf.convert_to_tensor(np.array(state_next))
                        t_sn = revert_board(t_sn)
                        move = revert_prediction(self.predict_move_to_play(True, white=False))
                    state_next, reward, done, _ = env.step(move)    
                    if done:
                        reward = -1             
                # Save actions and states in replay buffer
                self.action_history.append(action)
                self.state_history.append(state)
                self.state_next_history.append(state_next)
                self.done_history.append(done)
                self.rewards_history.append(reward)
                self.n_action_next_history.append(len(env.get_possible_actions()))

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
                    done_sample = tf.convert_to_tensor(
                        [float(self.done_history[i]) for i in indices]
                    )
                    mask_sample = [self.mask_history[i] for i in indices]
                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability

                    future_rewards = []
                    for s in state_next_sample:
                        s = s.reshape(-1,8,8,16)
                        res = model_target.predict([s]) #### MTCS
                        future_rewards.append(res[0])
                    future_rewards = np.array(future_rewards).astype('float32')
                    future_rewards = tf.convert_to_tensor(future_rewards)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + gamma * tf.reduce_max(
                        future_rewards, axis=1
                    )

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample
                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample,1)
                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        tensor_ss = tf.convert_to_tensor(np.array(state_sample))
                        pred = model(tensor_ss)
                        q_v_move = tf.math.reduce_max(pred[1] * tf.convert_to_tensor(mask_sample, dtype='float32'))

                        q_values = pred[0] * q_v_move
                        
                            #Voire comment on a la loss en fonction des différents outputs
                        q_values = tf.reshape(q_values, (-1))
                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = loss_function(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if self.frame_count % update_target_network == 0:
                    # update the the target network with new weights
                    model_target.set_weights(model.get_weights())
                    # Log details
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(self.running_reward,self.episode_count,self.frame_count))

                # Limit the state and reward history
                if len(self.rewards_history) > max_memory_length:
                    del self.rewards_history[:1]
                    del self.state_history[:1]
                    del self.state_next_history[:1]
                    del self.action_history[:1]
                    del self.done_history[:1]

                if done:
                    break
            print('Update')
            # Update running reward to check condition for solving
            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > 100:
                del self.episode_reward_history[:1]

            self.running_reward = np.mean(self.episode_reward_history)

            self.episode_count += 1

            if self.running_reward > 40:  # Condition to consider the task solved
                print("Solved at episode {}!".format(self.episode_count))
                break