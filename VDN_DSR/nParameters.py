class Parameters(object):
    Gamma = 0.99
    Learning_rate = 0.00025
    Epsilon = 1
    Final_epsilon = 0.1

    Num_action = 2
    Num_replay_memory = 100000
    Num_start_training = 100000
    Num_training = 10000000
    Num_rStep = 5000000
    Num_update = 10000
    Num_batch = 32
    Num_test = 10000
    Num_skipFrame = 1
    Num_stackFrame = 4
    # Num_colorChannel =1
    Num_plot_episode = 10
    Num_rdatabase = 1000
    r_batch_size = 1000


    GPU_fraction = 0.2
    Is_train = True
    #Load_path = ''

    Input_size = 25 * Num_stackFrame
    # img_size = 80
    #
    first_dense = [Input_size, 100]
    second_dense = [100, 100]
    decoder_first_dense = [100, 100]
    decoder_second_dense = [100, Input_size]
    reward_weight = [100, 1]

    fai_first_dense = [100, 100]
    fai_second_dense = [100, 100]
    fai_out_dense = [100, 100]
