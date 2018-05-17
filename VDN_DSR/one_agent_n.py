import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from PIL import Image
import pylab
from nParameters import Parameters
from Fetch_3act_bigr import GameEnv

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
WIN = True
TEST = False
game = 'Fetch_2act_one'
USE_GPU = False

def to_gray(state):
    im = Image.fromarray(np.uint8(state * 255))
    im = im.convert('L')
    return pylab.array(im)

if WIN:
    base_path = 'F://bs//sf-vdn//VDN_DSR//saved_models//'
else:
    base_path = '/home/admin1/zp/VDN_DSR/saved_models/'


class VDN_DSR(object):
    def __init__(self):
        self.algorithm = 'VDN_DSR'
        self.game_name = game
        self.env = GameEnv()
        self.progress = ''
        self.Num_action = Parameters.Num_action
        self.Num_Exploration = Parameters.Num_start_training
        self.Num_Training = Parameters.Num_training
        self.Num_Testing = Parameters.Num_test
        self.Num_rStep = Parameters.Num_rStep
        self.learning_rate = Parameters.Learning_rate
        self.gamma = Parameters.Gamma
        self.first_epsilon = Parameters.Epsilon
        self.final_epsilon = Parameters.Final_epsilon
        self.epsilon = Parameters.Epsilon
        self.r_saved_flag = False
        self.step = 1
        self.estep = 1
        self.score = 0
        self.episode = 0
        self.date_time = str(datetime.date.today()) + ' ' + \
                         str(datetime.datetime.now().hour) + '_' + \
                         str(datetime.datetime.now().minute)
        self.state_set = []
        self.Num_skipping = Parameters.Num_skipFrame
        self.Num_stacking = Parameters.Num_stackFrame
        self.Num_replay_memory = Parameters.Num_replay_memory
        self.Num_rdatebase = Parameters.Num_rdatabase
        self.Num_batch = Parameters.Num_batch
        self.replay_memory = []
        self.reward_database = []
        self.r_batch_size = Parameters.r_batch_size
        self.Num_update_target = Parameters.Num_update # #############################
        self.input_size = Parameters.Input_size
        self.first_dense = Parameters.first_dense
        self.second_dense = Parameters.second_dense
        self.decoder_first_dense = Parameters.decoder_first_dense
        self.decoder_second_dense = Parameters.decoder_second_dense
        self.reward_weight = Parameters.reward_weight
        self.fai_first_dense = Parameters.fai_first_dense
        self.fai_second_dense = Parameters.fai_second_dense
        self.fai_out_dense = Parameters.fai_out_dense

        if USE_GPU:
            config = tf.ConfigProto()
            config.log_device_placement = False
            self.sess = tf.InteractiveSession(config=config)
        else:
            self.sess = tf.InteractiveSession()
        self.input_1, self.autoencoder_out_1, self.reward_estimator_1, self.fai_act1_1, self.fai_act2_1, \
        self.state_feature_1, self.fai_input_1, self.fai_for_q_1, self.q_out_1, self.reward_weight_1 = self.network('network')

        self.input_1_target, self.autoencoder_out_1_target, self.reward_estimator_1_target, self.fai_act1_1_target, self.fai_act2_1_target, \
        self.state_feature_1_target, self.fai_input_1_target, self.fai_for_q_1_target, self.q_out_1_target, self.reward_weight_1_target = self.network('target')

        self.fai_act1_1_goal, self.fai_act2_1_goal, self.train_fai_act1_1, self.train_fai_act2_1, self.loss_fai_1, \
        self.r_target, self.loss_r, self.loss_autoencoder_1, self.loss_sum, self.train_r_and_autoencoder = self.loss_and_train()
        self.saver = self.init_saver()

        self.sess.run(tf.global_variables_initializer())
        # check save
        if not TEST:
            check_save = input('Load Model?(1=yes/2=no):')
            if check_save == '1':
                # restore variable from disk
                load_path = input('Please input the model path:')
                self.saver.restore(self.sess, load_path)
                print("Model restored")

                check_train = input("Prediction or training?(1=Inference/2=Training):")
                if check_train == '1':
                    self.Num_Exploration = 0
                    self.Num_Training = 0

    def init_saver(self):
        saver = tf.train.Saver()
        return saver

    def reshape_input(self, state):
        out = np.reshape(state, [-1])
        return out

    def initialization(self):
        self.env.reset()
        agent1_state_pre, agent2_state_pre = self.env.get_states()
        agent1_state = to_gray(agent1_state_pre)
        agent2_state = to_gray(agent2_state_pre)
        agent1_state_reshape = self.reshape_input(agent1_state)
        agent2_state_reshape = self.reshape_input(agent2_state)
        for i in range(self.Num_skipping * self.Num_stacking):
            self.state_set.append([agent1_state_reshape, agent2_state_reshape])
        return [agent1_state_reshape, agent2_state_reshape]

    def skip_and_stack_frame(self, states):
        self.state_set.append(states)
        agent1_state_in = np.zeros([self.Num_stacking, self.input_size // self.Num_stacking])
        agent2_state_in = np.zeros([self.Num_stacking, self.input_size // self.Num_stacking])
        for i in range(self.Num_stacking):
            agent1_state_in[i, :] = self.state_set[-1 - i][0]
            agent2_state_in[i, :] = self.state_set[-1 - i][1]
        del self.state_set[0]
        agent1_state_in_rs = np.reshape(agent1_state_in, [-1])
        agent2_state_in_rs = np.reshape(agent2_state_in, [-1])
        return [agent1_state_in_rs, agent2_state_in_rs]

    def get_progress(self):
        progress = ''
        if self.step <= self.Num_Exploration:
            progress = 'Exploring'
        elif self.step <= self. Num_Exploration + self.Num_rStep:
            progress = 'RLearning'
        elif self.step <= self.Num_Exploration + self.Num_Training + self.Num_rStep:
            progress = 'Training'
        elif self.step <= self.Num_Exploration + self.Num_Training + self.Num_Testing + self.Num_rStep:
            progress = 'Testing'
        else:
            progress = 'Finished'
        return progress

    def weight_variable(self, name, shape):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, name, shape):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def fai_net(self, network_name, state_feature):
        fai_w_fc1 = self.weight_variable(network_name + 'fai_w_fc1', self.fai_first_dense)
        fai_b_fc1 = self.bias_variable(network_name + 'fai_b_fc1', self.fai_first_dense[1])

        fai_h_fc1 = tf.nn.relu(tf.matmul(state_feature, fai_w_fc1) + fai_b_fc1)

        fai_w_fc2 = self.weight_variable(network_name + 'fai_w_fc2', self.fai_second_dense)
        fai_b_fc2 = self.weight_variable(network_name + 'fai_b_fc2', self.fai_second_dense[1])

        fai_h_fc2 = tf.nn.relu(tf.matmul(fai_h_fc1, fai_w_fc2) + fai_b_fc2)

        fai_w_out = self.weight_variable(network_name + 'fai_w_out', self.fai_out_dense)
        fai_b_out = self.weight_variable(network_name + 'fai_b_out', self.fai_out_dense[1])

        fai_net_out = tf.matmul(fai_h_fc2, fai_w_out) + fai_b_out

        return fai_net_out

    def dsr_net(self, network_name):
        x = tf.placeholder(tf.float32, shape=[None, 25 * self.Num_stacking])
        x_norm = (x - (255.0 / 2)) / (255.0 / 2)
        with tf.variable_scope(network_name):
            w_fc1 = self.weight_variable(network_name + 'w_fc1', self.first_dense)
            b_fc1 = self.bias_variable(network_name + 'b_fc1', self.first_dense[1])

            ah1 = tf.nn.relu(tf.matmul(x_norm, w_fc1) + b_fc1)

            w_fc2 = self.weight_variable(network_name + 'w_fc2', self.second_dense)
            b_fc2 = self.bias_variable(network_name + 'b_fc2', self.second_dense[1])

            state_feature = tf.matmul(ah1, w_fc2) + b_fc2

            w_decoder1 = self.weight_variable(network_name + 'w_decoder1', self.decoder_first_dense)
            b_decoder1 = self.bias_variable(network_name + 'b_decoder1', self.decoder_first_dense[1])

            decoder_h1 = tf.nn.relu(tf.matmul(state_feature, w_decoder1) + b_decoder1)

            w_decoder_out = self.weight_variable(network_name + 'w_decoder_out', self.decoder_second_dense)
            b_decoder_out = self.bias_variable(network_name + 'b_decoder_out', self.decoder_second_dense[1])

            autoencoder_out = tf.matmul(decoder_h1, w_decoder_out) + b_decoder_out

            reward_weight = self.weight_variable(network_name + 'reward_weight', self.reward_weight)

            fai_for_q = tf.placeholder(tf.float32, shape=[None, 100])

            reward_estimator = tf.matmul(state_feature, reward_weight)

            q_out = tf.matmul(fai_for_q, reward_weight)

            fai_input = tf.placeholder(tf.float32, shape=[None, 100])
            fai_act1 = self.fai_net(network_name + 'act1', fai_input)
            fai_act2 = self.fai_net(network_name + 'act2', fai_input)
            # for i in range(self.Num_action):
            #     fai.append(self._fai_net(network_name + str(i), fai_input))

            return x, autoencoder_out, reward_estimator, fai_act1, fai_act2, state_feature, fai_input, fai_for_q, q_out, reward_weight

    def network(self, network_name):
        with tf.variable_scope(network_name):
            x_1, autoencoder_out_1, reward_estimator_1, fai_act1_1, fai_act2_1, state_feature_1, fai_input_1, fai_for_q_1, q_out_1, reward_weight_1 = self.dsr_net('agt1_dsrnet')

            return x_1, autoencoder_out_1, reward_estimator_1, fai_act1_1, fai_act2_1, \
                   state_feature_1, fai_input_1, fai_for_q_1, q_out_1, reward_weight_1

    def loss_and_train(self):

        fai_act1_1_target = tf.placeholder(tf.float32, shape=[None, 100])
        fai_act2_1_target = tf.placeholder(tf.float32, shape=[None, 100])
        loss_fai_act1_1 = tf.reduce_sum(tf.square(self.fai_act1_1 - fai_act1_1_target))
        loss_fai_act2_1 = tf.reduce_sum(tf.square(self.fai_act2_1 - fai_act2_1_target))
        loss_fai_1 = (loss_fai_act1_1 + loss_fai_act2_1)/self.Num_batch
        # train_fai_1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(loss_fai_1)
        train_fai_act1_1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(loss_fai_act1_1)
        train_fai_act2_1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(loss_fai_act2_1)

        r_target = tf.placeholder(tf.float32, shape=[None])
        loss_r = tf.reduce_mean(tf.square(r_target - self.reward_estimator_1))

        loss_autoencoder_1 = tf.reduce_mean(tf.square(self.input_1 - self.autoencoder_out_1))

        loss_sum = loss_autoencoder_1 + loss_r
        train_r_and_autoencoder = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(loss_sum)

        return fai_act1_1_target, fai_act2_1_target, train_fai_act1_1, train_fai_act2_1, loss_fai_1, \
                r_target, loss_r, loss_autoencoder_1, loss_sum, train_r_and_autoencoder

    def experience_replay(self, state, action, reward, next_state, terminal):
        # If replay memory is full, delete the oldest experience
        if len(self.replay_memory) >= self.Num_replay_memory:
            del self.replay_memory[0]
        self.replay_memory.append([state, action, reward, next_state, terminal])
        if reward != 0:
            if len(self.reward_database) >= self.Num_rdatebase:
                del self.replay_memory[0]
            self.reward_database.append([state, action, reward, next_state, terminal])

    def update_target(self):
        # Get trainable variables
        trainable_variables = tf.trainable_variables()
        # network variables
        trainable_variables_network = [var for var in trainable_variables if var.name.startswith('network')]

        # target variables
        trainable_variables_target = [var for var in trainable_variables if var.name.startswith('target')]

        for i in range(len(trainable_variables_network)):
            self.sess.run(tf.assign(trainable_variables_target[i], trainable_variables_network[i]))

    def train_r(self):

        # Train r branch
        if random.random() <= 0.9:
            minibatch = random.sample(self.replay_memory, self.r_batch_size)
        else:
            minibatch = random.sample(self.reward_database, self.r_batch_size)
        # Aneal rbatch size by factot 0.8
        if self.step % 50000 == 0:
            if self.r_batch_size > 1:
                self.r_batch_size = int(self.r_batch_size * 0.8)
            if self.r_batch_size < 1:
                self.r_batch_size = 1

        next_state_1_batch = []

        reward_batch = []
        for batch in minibatch:
            next_state_1_batch.append(batch[3][0])
            reward_batch.append(batch[2])
        _ = self.sess.run(self.train_r_and_autoencoder,feed_dict={self.input_1: next_state_1_batch,
                                                                  self.r_target: reward_batch})
        print('rrr')

    def train_fai(self):
        # Train fai branch
        minibatch = random.sample(self.replay_memory, self.Num_batch)
        act1_state_1_batch = []
        act2_state_1_batch = []
        act1_next_state_1_batch = []
        act2_next_state_1_batch = []
        terminal_batch = []
        for batch in minibatch:
            if batch[1][0] == 0:
                act1_state_1_batch.append(batch[0][0])
                act1_next_state_1_batch.append(batch[3][0])
            else:
                act2_state_1_batch.append(batch[0][0])
                act2_next_state_1_batch.append(batch[3][0])
            terminal_batch.append(batch[4])

        # get y_prediction for fai_act1
        act1_y1_batch = []
        if len(act1_state_1_batch) != 0:
            act1_next_state_feature_1_batch = self.state_feature_1.eval(feed_dict={self.input_1: act1_next_state_1_batch})
            act1_target_next_state_fai_act1_batch = self.fai_act1_1_target.eval(feed_dict={self.fai_input_1_target:act1_next_state_feature_1_batch})
            act1_target_next_state_fai_act2_batch = self.fai_act2_1_target.eval(feed_dict={self.fai_input_1_target:act1_next_state_feature_1_batch})
            # Get state feature batches
            act1_state_feature_1_batch = self.state_feature_1.eval(feed_dict={self.input_1: act1_state_1_batch})
            act1_next_state_fai_act1_1_batch = self.fai_act1_1.eval(feed_dict={self.fai_input_1: act1_next_state_feature_1_batch})
            act1_next_state_fai_act2_1_batch = self.fai_act2_1.eval(feed_dict={self.fai_input_1: act1_next_state_feature_1_batch})
            act1_qact1_batch = self.q_out_1.eval(feed_dict={self.fai_for_q_1: act1_next_state_fai_act1_1_batch})
            act1_qact2_batch = self.q_out_1.eval(feed_dict={self.fai_for_q_1: act1_next_state_fai_act2_1_batch})
            act1_next_act_batch = []
            for i in range(len(act1_state_1_batch)):
                if(act1_qact1_batch[i][0] > act1_qact2_batch[i][0]):
                    act1_next_act_batch.append(0)
                else:
                    act1_next_act_batch.append(1)
            # get target fai values
            for i in range(len(act1_state_1_batch)):
                if terminal_batch[i] == True:
                    act1_y1_batch.append(act1_state_feature_1_batch[i])
                else:
                    if act1_next_act_batch[i] == 0:
                        act1_y1_batch.append(act1_state_feature_1_batch[i] + self.gamma * act1_target_next_state_fai_act1_batch[i])
                    else:
                        act1_y1_batch.append(act1_state_feature_1_batch[i] + self.gamma * act1_target_next_state_fai_act2_batch[i])

            _ = self.sess.run(self.train_fai_act1_1,
                              feed_dict={self.fai_input_1: act1_state_feature_1_batch,
                                         self.fai_act1_1_goal: act1_y1_batch})
        # get y_prediction for fai_act2
        act2_y1_batch = []
        if len(act2_state_1_batch) != 0:
            act2_next_state_feature_1_batch = self.state_feature_1.eval(feed_dict={self.input_1: act2_next_state_1_batch})
            act2_target_next_state_fai_act1_batch = self.fai_act1_1_target.eval(feed_dict={self.fai_input_1_target:act2_next_state_feature_1_batch})
            act2_target_next_state_fai_act2_batch = self.fai_act2_1_target.eval(feed_dict={self.fai_input_1_target:act2_next_state_feature_1_batch})
            # Get state feature batches
            act2_state_feature_1_batch = self.state_feature_1.eval(feed_dict={self.input_1: act2_state_1_batch})
            act2_next_state_fai_act1_1_batch = self.fai_act1_1.eval(feed_dict={self.fai_input_1: act2_next_state_feature_1_batch})
            act2_next_state_fai_act2_1_batch = self.fai_act2_1.eval(feed_dict={self.fai_input_1: act2_next_state_feature_1_batch})
            act2_qact1_batch = self.q_out_1.eval(feed_dict={self.fai_for_q_1: act2_next_state_fai_act1_1_batch})
            act2_qact2_batch = self.q_out_1.eval(feed_dict={self.fai_for_q_1: act2_next_state_fai_act2_1_batch})
            act2_next_act_batch = []
            for i in range(len(act2_state_1_batch)):
                if(act2_qact1_batch[i][0] > act2_qact2_batch[i][0]):
                    act2_next_act_batch.append(0)
                else:
                    act2_next_act_batch.append(1)
            # get target fai values
            for i in range(len(act2_state_1_batch)):
                if terminal_batch[i] == True:
                    act2_y1_batch.append(act2_state_feature_1_batch[i])
                else:
                    if act2_next_act_batch[i] == 0:
                        act2_y1_batch.append(act2_state_feature_1_batch[i] + self.gamma * act2_target_next_state_fai_act1_batch[i])
                    else:
                        act2_y1_batch.append(act2_state_feature_1_batch[i] + self.gamma * act2_target_next_state_fai_act2_batch[i])

            _ = self.sess.run(self.train_fai_act2_1,
                              feed_dict={self.fai_input_1: act2_state_feature_1_batch,
                                         self.fai_act2_1_goal: act2_y1_batch})

    def select_action(self, stack_state_1):
        action_1 = np.zeros([self.Num_action])
        action_1_index = 0
        # choose action
        if self.progress == 'Exploring' or self.progress == 'RLearning':
            # choose random action
            action_1_index = random.randint(0, self.Num_action - 1)
            action_1[action_1_index] = 1
        elif self.progress == 'Training':
            if random.random() < self.epsilon:
                # choose random action1
                action_1_index = random.randint(0, self.Num_action - 1)
                action_1[action_1_index] = 1
            else:
                # choose greedy action1
                st1 = self.state_feature_1.eval(feed_dict={self.input_1: [stack_state_1]})
                fai_act1_1 = self.fai_act1_1.eval(feed_dict={self.fai_input_1:st1})
                fai_act2_1 = self.fai_act2_1.eval(feed_dict={self.fai_input_1:st1})
                q_act1_1 = self.q_out_1.eval(feed_dict={self.fai_for_q_1:fai_act1_1})
                q_act2_1 = self.q_out_1.eval(feed_dict={self.fai_for_q_1:fai_act2_1})
                if q_act1_1[0] > q_act2_1[0]:
                    action_1_index = 0
                else:
                    action_1_index = 1
                action_1[action_1_index] = 1

            # Decrease epsilon while training
            if self.epsilon > self.final_epsilon:
                self.epsilon -= self.first_epsilon / self.Num_Training

        return action_1

    def save_model(self):
        # Save the variables to disk.
        if self.step == self.Num_Exploration + self.Num_Training:
            save_path = self.saver.save(self.sess, base_path + self.game_name +
                                        '//' + self.date_time +  '_' + self.algorithm + "//model.ckpt")
            print("Model saved in file: %s" % save_path)

    def save_model_backup(self):
        # Save the variables to disk.
        if self.step == 101000 or self.step % 100000 == 0:
            save_path = self.saver.save(self.sess, base_path + self.game_name +
                                        '//' + self.date_time + '_' + self.algorithm + '_' + str(
                self.step) + "//model.ckpt")
            print("Model saved in file: %s" % save_path)

    def main(self):
        states = self.initialization()
        stacked_states = self.skip_and_stack_frame(states)
        while True:
            # Get progress
            self.progress = self.get_progress()

            # select action
            act1_one_shot = self.select_action(stacked_states[0])
            act1 = np.argmax(act1_one_shot)
            act2 = 2
            # Take actions and get info for update
            r1, r2 = self.env.move(act1, act2)
            r = r1 + r2
            next_states_pre = self.env.get_states()
            next_states = [to_gray(next_states_pre[0]), to_gray(next_states_pre[1])]
            if r1 == 5 or r2 == 5:
                terminal = False
            else:
                terminal = False
            next_states = [self.reshape_input(next_states[0]), self.reshape_input(next_states[1])]
            stacked_next_states = self.skip_and_stack_frame(next_states)

            # Experience Replay
            self.experience_replay(stacked_states, [act1, act2], r, stacked_next_states, terminal)

            # Training r
            if self.progress == 'RLearning':
                self.train_r()
                self.save_model()
                self.save_model_backup()
            # Training fai
            if self.progress == 'Training':
                # Update target network
                if self.step % self.Num_update_target == 0:
                    self.update_target()

                # train
                self.train_fai()

                self.save_model()
                self.save_model_backup()

            # update former info.
            stacked_states = stacked_next_states
            self.score += r
            if len(self.reward_database) >= self.Num_rdatebase:
                if self.r_saved_flag == False:
                    self.score = 0
                    self.r_saved_flag = True
                    # self.save_reward()
                    self.estep = 1
                self.step += 1
            else:
                self.estep += 1
            if self.estep % 10000 == 0:
                print(len(self.reward_database), self.score)


            if self.step % 50000 == 0:
                print('Step: ' + str(self.step) + ' / ' +
                      'Episode: ' + str(self.episode) + ' / ' +
                      'Progress: ' + self.progress + ' / ' +
                      'Epsilon: ' + str(self.epsilon) + ' / ' +
                      'Score: ' + str(self.score))

            # If game is over(terminal)
            # if terminal:
            #     stacked_states = self.if_terminal()

            # Finished!
            if self.progress == 'Finished':
                print('Finished!')
                break

if __name__ == '__main__':
    agent = VDN_DSR()
    if TEST:
        agent.test()
    else:
        agent.main()