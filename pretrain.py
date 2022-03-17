from deepQ import *

if __name__ == '__main__':
    env = Chess_env()
    _ = env.reset()
    model = DeepQ(env, dropout_rate=0., n_channels=128, n_residual=15)
    model.pre_train(
            max_epoch= 200*32, 
            batch_size = 128,    
            max_steps_per_episode = 30, 
            learning_rate = 1e-4,
            jupyter=False,
            random_best_action=0.3,
            n_top_move= 30, 
            name='pretrained',
            length_train_set=256*50,
            length_test_set = 64*50,
        )