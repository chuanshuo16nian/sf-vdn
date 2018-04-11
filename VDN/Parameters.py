
Gamma = 0.99
Learning_rate = 0.00025
Epsilon = 0.27857
Final_epsilon = 0.1

Num_action = 5
Num_replay_memory = 50000
Num_start_training = 50000
Num_training = 5000000
Num_update = 5000
Num_batch = 32
Num_test = 10000
Num_skipFrame = 1
Num_stackFrame = 4
# Num_colorChannel =1
Num_plot_episode = 10

GPU_fraction = 0.2
Is_train = True
#Load_path = ''

Input_size = 25 * Num_stackFrame
# img_size = 80

# first_conv = [8, 8, Num_colorChannel *Num_stackFrame, 32]
# second_conv = [4, 4, 32, 64]
# third_conv = [3, 3, 64, 64]
# first_dense =  [10 * 10 * 64, 512]
first_dense = [Input_size, 128]
second_dense = [128, 64]
output_dense = [64, Num_action]
first_LSTM = [32, 32] ##############################
