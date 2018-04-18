import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from PIL import Image
import pylab
from Parameters import Parameters
from Fetch_3act import GameEnv
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def to_gray(state):
    im = Image.fromarray(np.uint8(state * 255))
    im = im.convert('L')
    return pylab.array(im)


game = 'Fetch_3act'
USE_GPU = False
base_path = 'F://bs//sf-vdn//VDN//saved_models//'

class VDN(object):
    def __init__(self):

        self.algorithm = 'VDN'
        self.game_name = game
        self.env = GameEnv()
        self.progress = ''
        self.Num_action = Parameters.Num_action ######################

        self.Num_replay_episode = 500
        self.step_size = 6
        self.lstm_size = 32
        # self.flatten_size = 10 * 10 * 64

        self.episode_memory = []

        # Initialize parameters
        self.Num_Exploration = Parameters.Num_start_training
        self.Num_Training = Parameters.Num_training
        self.Num_Testing = Parameters.Num_test

        self.learning_rate = Parameters.Learning_rate
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
        self.state_set = []
        self.Num_skipping = Parameters.Num_skipFrame
        self.Num_stacking = Parameters.Num_stackFrame

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
        self.decoder_dense = Parameters.decoder_dense
        self.reward_weight = Parameters.reward_weight
        self.fai_first_dense = Parameters.fai_first_dense
        self.fai_second_dense = Parameters.fai_second_dense
        self.fai_out_dense = Parameters.fai_out_dense

        self.GPU_fraction = Parameters.GPU_fraction

        # Variables for tensorboard
        self.loss = 0
        self.maxQ = 0
        self.score_board = 0
        self.maxQ_board = 0
        self.loss_board = 0
        self.step_old = 0

        # Initialize Network
        if USE_GPU:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.GPU_fraction
            config.log_device_placement = False
            self.sess = tf.InteractiveSession(config=config)
        else:
            self.sess = tf.InteractiveSession()

        self.input_1,self.input_2, self.autoencoder_out_1, self.autoencoder_out_2, self.reward_estimator_1, \
        self.reward_estimator_2, self.fai_1, self.fai_2, self.state_feature_1, self.state_feature_2 = self.network('network')

        self.input_target_1, self.input_target_2, self.autoencoder_out_target_1, self.autoencoder_out_target_2, \
        self.reward_estimator_target_1, self.reward_estimator_target_2, self.fai_target_1, self.fai_target_2, \
        self.state_feature_target_1, self.state_feature_target_2 = self.network('target')

        self.train_step, self.action_target, self.y_target, self.loss_train = self.loss_and_train()
        self.saver, self.summary_placeholders, \
        self.update_ops, self.summary_op, self.summary_writer = self.init()

        self.sess.run(tf.global_variables_initializer())
        # check save
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

    def test(self):
        load_path = input("Please input the model path:")
        self.saver.restore(self.sess, load_path)
        states = self.initialization()
        stacked_states = self.skip_and_stack_frame(states)
        stacked_states[0] = np.reshape(stacked_states[0], [1, 100])
        stacked_states[1] = np.reshape(stacked_states[1], [1, 100])
        while self.step < self.Num_Testing:
            if random.random() < 0.0:
                act1 = random.randint(0, self.Num_action - 1)
                act2 = random.randint(0, self.Num_action - 1)
            else:
                action_one_shot = self.output.eval(feed_dict={self.input_1:stacked_states[0],
                                                              self.input_2:stacked_states[1]})
                action_joint = np.argmax(action_one_shot)
                act1 = action_joint // self.Num_action
                act2 = action_joint % self.Num_action
            r1, r2 = self.env.move(act1, act2)
            self.step += 1
            agent1_state_pre, agent2_state_pre = self.env.get_states()
            agent1_state = to_gray(agent1_state_pre)
            agent2_state = to_gray(agent2_state_pre)
            agent1_state_reshape = self.reshape_input(agent1_state)
            agent2_state_reshape = self.reshape_input(agent2_state)
            states = [agent1_state_reshape, agent2_state_reshape]
            stacked_states = self.skip_and_stack_frame(states)
            stacked_states[0] = np.reshape(stacked_states[0], [1, 100])
            stacked_states[1] = np.reshape(stacked_states[1], [1, 100])
            print("act1: %d, act2: %d" % (act1, act2))
            print("r1: %d, r2: %d" % (r1, r2))
            im = self.env.render_env()
            plt.imshow(im)
            plt.show(block=False)
            plt.pause(0.02)
            plt.clf()
            terminal = False
            if r1 == 5 or r2 == 5:
                terminal = False
            if terminal:
                print('Step: ' + str(self.step) + ' / ' +
                      'Episode: ' + str(self.episode) + ' / ' +
                      'Progress: ' + self.progress + ' / ' +
                      'Epsilon: ' + str(self.epsilon) + ' / ' +
                      'Score: ' + str(self.score))
                # If game is finished, initialize the state
                states = self.initialization()
                stacked_states = self.skip_and_stack_frame(states)
                stacked_states[0] = np.reshape(stacked_states[0], [1, 100])
                stacked_states[1] = np.reshape(stacked_states[1], [1, 100])
                self.episode += 1

    def main(self):
        states = self.initialization()
        stacked_states = self.skip_and_stack_frame(states)
        while True:
            # Get progress
            self.progress = self.get_progress()

            # select action
            act1_one_shot, act2_one_shot = self.select_action(stacked_states[0], stacked_states[1])
            act1 = np.argmax(act1_one_shot)
            act2 = np.argmax(act2_one_shot)
            # Take actions and get info for update
            r1, r2 = self.env.move(act1, act2)
            r = r1 + r2
            next_states_pre = self.env.get_states()
            next_states = [to_gray(next_states_pre[0]), to_gray(next_states_pre[1])]
            if r1 ==5 or r2 == 5:
                terminal = False
            else:
                terminal = False
            next_states= [self.reshape_input(next_states[0]), self.reshape_input(next_states[1])]
            stacked_next_states = self.skip_and_stack_frame(next_states)

            # Experience Replay
            self.experience_replay(stacked_states, [act1, act2], r, stacked_next_states, terminal)

            # Training
            if self.progress == 'Training':
                # Update target network
                if self.step % self.Num_update_target == 0:
                    self.update_target()

                # train
                self.train(self.replay_memory)

                self.save_model()
                self.save_model_backup()

            # update former info.
            stacked_states = stacked_next_states
            self.score += r
            self.step += 1

            # Plotting
            self.plotting(terminal)

            if self.step % 50000 == 0:
                print('Step: ' + str(self.step) + ' / ' +
                      'Episode: ' + str(self.episode) + ' / ' +
                      'Progress: ' + self.progress + ' / ' +
                      'Epsilon: ' + str(self.epsilon) + ' / ' +
                      'Score: ' + str(self.score))

            # If game is over(terminal)
            if terminal:
                stacked_states = self.if_terminal()

            # Finished!
            if self.progress == 'Finished':
                print('Finished!')
                break

    def init(self):


        # make folder to save model
        save_path = '/saved_models/'+ self.game_name + '/' + self.date_time
        os.makedirs(save_path)

        # Summary for tensorboard
        summary_placeholdes, update_ops, summary_op = self.setup_summary()
        summary_writer = tf.summary.FileWriter(save_path, self.sess.graph)

        saver = tf.train.Saver()
        return saver, summary_placeholdes, update_ops, summary_op, summary_writer

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
        agent1_state_in = np.zeros([self.Num_stacking, np.int(self.input_size/self.Num_stacking)])
        agent2_state_in = np.zeros([self.Num_stacking, np.int(self.input_size/self.Num_stacking)])
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
        elif self.step <= self.Num_Exploration + self.Num_Training:
            progress = 'Training'
        elif self.step <= self.Num_Exploration + self.Num_Training + self.Num_Testing:
            progress = 'Testing'
        else:
            progress = 'Finished'
        return progress

    # Resize the input into a long vector
    def reshape_input(self, state):
        out = np.reshape(state, [-1])
        return out

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

    def weight_variable(self, name, shape):
        return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

    def bias_variable(self, name, shape):
        return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

    def _dsr_net(self, network_name):
        x = tf.placeholder(tf.float32, shape=[None, 25 * self.Num_stacking])
        x_norm = (x - (255.0 / 2)) / (255.0 / 2)
        with tf.variable_scope(network_name):
            w_fc1 = self.weight_variable(network_name + 'w_fc1', self.first_dense)
            b_fc1 = self.bias_variable(network_name + 'b_fc1', self.first_dense[1])

            state_feature = tf.nn.relu(tf.matmul(x_norm, w_fc1) + b_fc1)

            w_decoder = self.weight_variable(network_name + 'w_decoder', self.decoder_dense)
            b_decoder = self.weight_variable(network_name + 'b_decoder', self.decoder_dense[1])

            autoencoder_out = tf.matmul(state_feature, w_decoder) + b_decoder

            reward_weight = self.weight_variable(network_name + 'reward_weight', self.reward_weight)

            reward_estimator = tf.matmul(state_feature, reward_weight)

            fai = []
            for i in range(self.Num_action):
                fai.append(self._fai_net(network_name, state_feature))

            return x, autoencoder_out, reward_estimator, fai, state_feature

    def _fai_net(self, network_name, state_feature):
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

    def network(self, network_name):
        with tf.variable_scope(network_name):
            x_1, autoencoder_out_1, reward_estimator_1, fai_1, state_feature_1 = self._dsr_net('agt1_dsrnet')
            x_2, autoencoder_out_2, reward_estimator_2, fai_2, state_feature_2 = self._dsr_net('agt1_dsrnet')

            # compute joint Q
            # tmp1 = []
            # l = tf.reshape(Q_1, [-1])
            # for i in range(self.Num_action):
            #     tmp1.append(l)
            # tmp1 = tf.transpose(tmp1, [1, 0])
            # tmp1 = tf.reshape(tmp1, [-1, self.Num_action * self.Num_action])
            # tmp2 = Q_2
            # for i in range(self.Num_action - 1):
            #     tmp2 = tf.concat([tmp2, Q_2], 1)
            # Q_joint = tf.add(tmp1, tmp2)

            return x_1, x_2, autoencoder_out_1, autoencoder_out_2, reward_estimator_1, reward_estimator_2, \
                   fai_1, fai_2, state_feature_1, state_feature_2

    def loss_and_train(self):
        # action_target = tf.placeholder(tf.float32, shape=[None, self.Num_action * self.Num_action])
        fai_1_target = tf.placeholder(tf.float32, shape=[None, 64])
        fai_2_target = tf.placeholder(tf.float32, shape=[None, 64])
        Loss_fai_1 = []
        Loss_fai_2 = []
        train_fai_1 = []
        train_fai_2 = []
        for act in range(self.Num_action):
            fai_1_prediction = self.fai_1[act]
            fai_2_prediction = self.fai_2[act] # tf.reduce_sum(tf.multiply(self.output, action_target), reduction_indices=1)
            loss1 = tf.reduce_mean(tf.square(fai_1_prediction - fai_1_target))
            loss2 = tf.reduce_mean(tf.square(fai_2_prediction - fai_2_target))
            Loss_fai_1.append(loss1)
            Loss_fai_2.append(loss2)
            train_fai_1.append(tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(loss1))
            train_fai_2.append(tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-02).minimize(loss2))

        r_1_target = tf.placeholder(tf.float32, shape=[None])
        r_2_target = tf.placeholder(tf.float32, shape=[None])
        Loss_r_1 = tf.reduce_mean(tf.square(r_1_target - self.reward_estimator_1))
        Loss_r_2 = tf.reduce_mean(tf.square(r_2_target - self.reward_estimator_2))
        Loss_autoencoder_1 = tf.reduce_mean(tf.square(self.input_1 - self.autoencoder_out_1))
        Loss_autoencoder_2 = tf.reduce_mean(tf.square(self.input_2 - self.autoencoder_out_2))
        Loss_sum_1 = Loss_r_1 + Loss_autoencoder_1
        Loss_sum_2 = Loss_r_2 + Loss_autoencoder_2

        return fai_1_target, fai_2_target, train_fai_1, train_fai_2, Loss_fai_1, Loss_fai_2, \
               r_1_target, r_2_target,

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
                # choose random action1
                action_1_index = random.randint(0, self.Num_action - 1)
                action_1[action_1_index] = 1
            else:
                # choose greedy action1
                Q_1_value = self.Q_1.eval(feed_dict={self.input_1:[stack_state_1]})
                action_1_index = np.argmax(Q_1_value)
                action_1[action_1_index] = 1

            if random.random() < self.epsilon:
                # choose random action2
                action_2_index = random.randint(0, self.Num_action - 1)
                action_2[action_2_index] = 1
            else:
                # choose greedy action2
                Q_2_value = self.Q_2.eval(feed_dict={self.input_2: [stack_state_2]})
                action_2_index = np.argmax(Q_2_value)
                action_2[action_2_index] = 1

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
            index = action_1_batch[i] * self.Num_action + action_2_batch[i]
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
                y_batch.append(reward_batch[i] + self.gamma * np.max(Q_batch[i]))
        _, self.loss = self.sess.run([self.train_step, self.loss_train],
                                     feed_dict={self.action_target:action_batch,
                                                self.y_target:y_batch,
                                                self.input_1:state_1_batch,
                                                self.input_2:state_2_batch})

    def save_model(self):
        # Save the variables to disk.
        if self.step == self.Num_Exploration + self.Num_Training:
            save_path = self.saver.save(self.sess, base_path + self.game_name +
                                        '//' + self.date_time +  '_' + self.algorithm + "//model.ckpt")
            print("Model saved in file: %s" % save_path)

    def save_model_backup(self):
        # Save the variables to disk.
        if self.step == 51000 or self.step % 100000 == 0:
            save_path = self.saver.save(self.sess, base_path + self.game_name +
                                        '//' + self.date_time +  '_' + self.algorithm + '_' + str(self.step) + "//model.ckpt")
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

    def if_terminal(self):
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
        state = self.initialization()
        stacked_state = self.skip_and_stack_frame(state)

        return stacked_state


if __name__ == '__main__':
    agent = VDN()
    # agent.main()
    agent.test()






















