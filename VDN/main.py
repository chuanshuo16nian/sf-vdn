import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import os

import Parameters

game = 'Fetch'
USE_GPU = False

class VDN(object):
    def __init__(self):

        self.algorithm = 'VDN'
        self.game_name = game

        self.progress = ''
        self.Num_action = 5 ######################

        self.Num_replay_episode = 500
        self.step_size = 6
        self.lstm_size = 32
        # self.flatten_size = 10 * 10 * 64

        self.episode_memory = []

        # Initialize parameters
        self.Num_Exploration = Parameters.Num_start_training
        self.Num_Training = Parameters.Num_training
        self.Num_Testing = Parameters.Num_test

        self.learning_rate = Parameters.Learnig_rate
        self.gamma = Parameters.Gamma

        self.first_epsilon = Parameters.Epsilon
        self.final_epsilon = Parameters.Final_epsilon
        self.epsilon = Parameters.Epsilon

        self.Num_plot_episode = Parameters.Num_plot_episode

        self.Is_train = Parameters.Is_train
        # self.Load_path = Parameters.Load_path

        self.step = 1
        self.score = 0
        self.episode = 0

        # Training time
        self.date_time = str(datetime.date.today()) + ' ' + \
                         str(datetime.datetime.now().hour) + '_' + \
                         str(datetime.datetime.now().minute)

        # parameters for skip and stack
        # self.state_set = []
        # self.Num_skipping = Parameters.Num_skipFrame
        # self.Num_stacking = Parameters.Num_stackFrame

        # parameters for experience replay
        self.Num_replay_memory = Parameters.Num_replay_memory
        self.Num_batch = Parameters.Num_batch
        self.replay_memory = []

        # parameters for target network
        self.Num_update_target = Parameters.Num_update

        # parameters for network
        self.input_size = Parameters.Input_size
        # self.Num_colorChannel = Parameters.Num_colorChannel

        self.first_dense = Parameters.first_dense
        self.first_LSTM =Parameters.first_LSTM
        self.second_dense = Parameters.second_dense
        self.output_dense = Parameters.output_dense

        self.GPU_fraction = Parameters.GPU_fraction

        # Variables for tensorboard
        self.loss = 0
        self.maxQ = 0
        self.score_board = 0
        self.maxQ_board = 0
        self.loss_board = 0
        self.step_old = 0

        # Initialize Network
        self.sess, self.saver, self.summary_placeholders, \
        self.update_ops, self.summary_op, self.summary_writer = self.init_sess()
        self.input_1,self.input_2, self.output, self.Q_1, self.Q_2 = self.network('network')
        self.input_target_1, self.input_target_2, self.output_target,self.Q_1_target, self.Q_2_target = self.network('target')
        self.train_step, self.action_target, self.y_target, self.loss_train = self.loss_and_train()


    def main(self):
        pass

    def init_sess(self):
        # initialize variables
        if USE_GPU:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.GPU_fraction
            sess = tf.InteractiveSession(config=config)
        else:
            sess = tf.InteractiveSession()

        # make folder to save model
        save_path = './saved_models/'+ self.game_name + '/' + self.date_time
        os.makedirs(save_path)

        # Summary for tensorboard
        summary_placeholdes, update_ops, summary_op = self.setup_summary()
        summary_writer = tf.summary.FileWriter(save_path, sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)

        # Load the file if the saved file exists
        saver = tf.train.Saver()
        # check save
        check_save = input('Load Model?(1=yes/2=no):')
        if check_save == 1:
            # restore variable from disk
            load_path = input('Please input the model path:')
            saver.restore(sess, load_path)
            print("Model restored")

            check_train = input("Prediction or training?(1=Inference/2=Training):")
            if check_train == 1:
                self.Num_Exploration = 0
                self.Num_Training = 0

        return sess, saver, summary_placeholdes, update_ops, summary_op, summary_writer

    def initialization(self, game_state):
        initialState = ''
        return initialState

    def skip_and_stack_frame(self, state):
        pass

    def get_progress(self):
        progress = ''
        if self.step <= self.Num_Exploration:
            progress = 'Exploring'
        elif self.step <= self.Num_Exploration + self.Num_Training:
            progress = 'Training'
        elif self.step <= self.Num_Exploration + self.Num_Training + self.Num_Testing:
            progress = 'Testing'
        else:
            progress = 'Finished'
        return progress

    # Resize the input into a long vector
    def reshape_input(self, state):
        pass

    # Code for tensorboard
    def setup_summary(self):
        episode_score = tf.Variable(0.)
        episode_MaxQ = tf.Variable(0.)
        episode_loss = tf.Variable(0.)

        tf.summary.scalar('Average Acore/' + str(self.Num_plot_episode) + 'episodes', episode_score)
        tf.summary.scalar('Average MaxQ/' + str(self.Num_plot_episode) + 'episodes', episode_MaxQ)
        tf.summary.scalar('Average Loss/' + str(self.Num_plot_episode) + 'episodes', episode_loss)

        summary_vars = [episode_score, episode_MaxQ, episode_loss]

        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op

    # Convolution
    # def conv2d(self, x, w, stride):
    #   return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

    # Get Variables
    # def conv_weight_variable(self, name, shape):
    #    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())

    def weight_variable(self, name, shape):
        return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

    def bias_variable(self, name, shape):
        return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

    def network(self, network_name):
        # input
        x_1 = tf.placeholder(tf.float32, shape=[None, 75 * 4])
        x_2 = tf.placeholder(tf.float32, shape=[None, 75 * 4])

        x_1_norm = (x_1 - (255.0 / 2)) / (255.0 / 2)
        x_2_norm = (x_2 - (255.0 / 2)) / (255.0 / 2)

        # cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_size, state_is_tuple=True)

        with tf.variable_scope(network_name):

            # fc1 weights
            w_fc1_1 = self.weight_variable(network_name + 'w_fc1_1', self.first_dense)
            b_fc1_1 = self.bias_variable(network_name + 'b_fc1_1', [self.first_dense[1]])
            w_fc1_2 = self.weight_variable(network_name + 'w_fc1_2', self.first_dense)
            b_fc1_2 = self.bias_variable(network_name + 'b_fc1_2', [self.first_dense[1]])
            # fc2 weight2
            w_fc2_1 = self.weight_variable(network_name + 'w_fc2_1', self.second_dense)
            b_fc2_1 = self.bias_variable(network_name + 'b_fc2_1', [self.second_dense[1]])
            w_fc2_2 = self.weight_variable(network_name + 'w_fc2_2', self.second_dense)
            b_fc2_2 = self.bias_variable(network_name + 'b_fc2_2', [self.second_dense[1]])
            # output layer weights
            w_out_1 = self.weight_variable(network_name + 'w_out_1', self.output_dense)
            b_out_1 = self.bias_variable(network_name + 'b_out_1', [self.output_dense[1]])
            w_out_2 = self.weight_variable(network_name + 'w_out_2', self.output_dense)
            b_out_2 = self.bias_variable(network_name + 'b_out_2', [self.output_dense[1]])

            # network
            h_fc1_1 = tf.nn.relu(tf.matmul(x_1_norm, w_fc1_1) + b_fc1_1)
            h_fc1_2 = tf.nn.relu(tf.matmul(x_2_norm, w_fc1_2) + b_fc1_2)

            h_fc2_1 = tf.nn.relu(tf.matmul(h_fc1_1, w_fc2_1) + b_fc2_1)
            h_fc2_2 = tf.nn.relu(tf.matmul(h_fc1_2, w_fc2_2) + b_fc2_2)

            Q_1 = tf.matmul(h_fc2_1, w_out_1) + b_out_1
            Q_2 = tf.matmul(h_fc2_2, w_out_2) + b_out_2
            # compute joint Q
            tmp1 = []
            l = tf.reshape(Q_1, [-1])
            for i in range(self.Num_action):
                tmp1.append(l)
            tmp1 = tf.transpose(tmp1, [1, 0])
            tmp1 = tf.reshape(tmp1, [-1, self.Num_action * self.Num_action])
            tmp2 = Q_2
            for i in range(self.Num_action - 1):
                tmp2 = tf.concat([tmp2, Q_2], 1)
            Q_joint = tf.add(tmp1, tmp2)

            return x_1, x_2, Q_joint, Q_1, Q_2

    def loss_and_train(self):
        action_target = tf.placeholder(tf.float32, shape=[None, self.Num_action * self.Num_action])
        y_target = tf.placeholder(tf.float32, shape=[None])

        y_prediction = tf.reduce_sum(tf.multiply(self.output, action_target), reduction_indices=1)
        Loss = tf.reduce_mean(tf.square (y_prediction - y_target))
        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(Loss)

        return train_step, action_target, y_target, Loss


    def select_action(self, stack_state_1, stack_state_2):
        action_1 = np.zeros([self.Num_action])
        action_1_index = 0
        action_2 = np.zeros([self.Num_action])
        action_2_index = 0
        # choose action
        if self.progress == 'Exploring':
            # choose random action
            action_1_index = random.randint(0, self.Num_action - 1)
            action_2_index = random.randint(0, self.Num_action - 1)
            action_1[action_1_index] = 1
            action_2[action_2_index] = 1
        elif self.progress == 'Training':
            if random.random() < self.epsilon:
                # choose random action
                action_1_index = random.randint(0, self.Num_action - 1)
                action_1[action_1_index] = 1
                action_2_index = random.randint(0, self.Num_action - 1)
                action_2[action_2_index] = 1
            else:
                # choose greedy action
                Q_1_value = self.Q_1.eval(feed_dict={self.input_1:[stack_state_1]})
                action_1_index = np.argmax(Q_1_value)
                action_1[action_1_index] = 1
                Q_2_value = self.Q_2.eval(feed_dict={self.input_2: [stack_state_2]})
                action_2_index = np.argmax(Q_2_value)
                action_2[action_2_index] = 1
                self.maxQ = np.max(Q_1_value + Q_2_value)

            # Decrease epsilon while training
            if self.epsilon > self.final_epsilon:
                self.epsilon -= self.first_epsilon/self.Num_Training
        elif self.progress == 'Testing':
            # choose greedy action
            Q_1_value = self.Q_1.eval(feed_dict={self.input_1: [stack_state_1]})
            action_1_index = np.argmax(Q_1_value)
            action_1[action_1_index] = 1
            Q_2_value = self.Q_2.eval(feed_dict={self.input_2: [stack_state_2]})
            action_2_index = np.argmax(Q_2_value)
            action_2[action_2_index] = 1
            self.maxQ = np.max(Q_1_value + Q_2_value)

            self.epsilon = 0

        return action_1, action_2

    def experience_replay(self, state, action, reward, next_state, terminal):
        # If replay memory is full, delete the oldest experience
        if len(self.replay_memory) >= self.Num_replay_memory:
            del self.replay_memory[0]

        self.replay_memory.append([state, action, reward, next_state, terminal])

    def update_target(self):
        # Get trainable variables
        trainable_variables = tf.trainable_variables()
        # network variables
        trainable_variables_network = [var for var in trainable_variables if var.name.startswith('network')]

        # target variables
        trainable_variables_target = [var for var in trainable_variables if var.name.startswith('target')]

        for i in range(len(trainable_variables_network)):
            self.sess.run(tf.assign(trainable_variables_target[i], trainable_variables_network[i]))

    def train(self, replay_memory):

        # Sample from replay memory
        minibatch = random.sample(replay_memory, self.Num_batch)

        # save the each batch data
        state_1_batch = [batch[0][0] for batch in minibatch]
        state_2_batch = [batch[0][1] for batch in minibatch]
        action_1_batch = [batch[1][0] for batch in minibatch]
        action_2_batch = [batch[1][1] for batch in minibatch]
        reward_batch = [batch[2] for batch in minibatch]
        next_state_1_batch = [batch[3][0] for batch in minibatch]
        next_state_2_batch = [batch[3][1] for batch in minibatch]
        terminal_batch = [batch[4] for batch in minibatch]
        action_batch = []
        for i in range(len(minibatch)):
            tmp = np.zeros([self.Num_action * self.Num_action])
            index = np.argmax(action_1_batch[i]) * self.Num_action + np.argmax(action_2_batch[i])
            tmp[index] = 1
            action_batch.append(tmp)


        # get y_prediction
        y_batch = []
        Q_batch = self.output_target.eval(feed_dict={self.input_target_1:next_state_1_batch,
                                                     self.input_target_2:next_state_2_batch})
        # get target values
        for i in range(len(minibatch)):
            if terminal_batch[i] == True:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch + self.gamma * np.max(Q_batch[i]))

        _, self.loss = self.sess.run([self.train_step, self.loss_train],
                                     feed_dict={self.action_target:action_batch,
                                                self.y_target:y_batch,
                                                self.input_1:state_1_batch,
                                                self.input_2:state_2_batch})

    def save_model(self):
        # Save the variables to disk.
        if self.step == self.Num_Exploration + self.Num_Training:
            save_path = self.saver.save(self.sess, 'saved_networks/' + self.game_name +
                                        '/' + self.date_time +  '_' + self.algorithm + "/model.ckpt")
            print("Model saved in file: %s" % save_path)

    def plotting(self, terminal):
        if self.progress != 'Exploring':
            if terminal:
                self.score_board += self.score

            self.maxQ_board += self.maxQ
            self.loss_board += self.loss

            if self.episode % self.Num_plot_episode == 0 and self.episode != 0 and terminal:
                diff_step = self.step - self.step_old
                tensorboard_info = [self.score_board / self.Num_plot_episode, self.maxQ_board / diff_step, self.loss_board / diff_step]

                for i in range(len(tensorboard_info)):
                    self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]:float(tensorboard_info[i])})
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.step)

                self.score_board = 0
                self.maxQ_board = 0
                self.loss_board = 0
                self.step_old = self.step
        else:
            self.step_old = self.step


    def if_terminal(self, game_state):
        # Show Progress
        print('Step: ' + str(self.step) + ' / ' +
              'Episode: ' + str(self.episode) + ' / ' +
              'Progress: ' + self.progress + ' / ' +
              'Epsilon: ' + str(self.epsilon) + ' / ' +
              'Score: ' + str(self.score))
        if self.progress != 'Exploring':
            self.episode += 1
        self.score = 0

        # If game is finished, initialize the state
        state = self.initialization(game_state)
        stacked_state = self.skip_and_stack_frame(state)

        return stacked_state

if __name__ == '__main__':
    agent = VDN()
    agent.main()
























