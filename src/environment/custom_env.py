import time
import sys

import numpy as np
from environment.base_environment import BaseEnvironment
from gym import spaces
from helper.moderator import Moderator
from network.netlink_communicator import NetlinkCommunicator
import math
import copy


class CCEnv(BaseEnvironment):
    '''Kernel Environment that follows gym interface'''
    metadata = {'render.modes': ['human']}

    def __init__(self, num_features: int,
                 window_len: int, num_fields_kernel: int, jiffies_per_state: int,
                 num_actions: int, steps_per_episode: int,
                 step_wait_seconds: float, comm: NetlinkCommunicator, moderator: Moderator, reward_name: str) -> None:
        super(CCEnv, self).__init__(comm, num_fields_kernel, moderator)

        # self.moderator = moderator
        self.width_state = num_features # state dimension
        self.height_state = window_len

        # Kernel state
        self.jiffies_per_state = jiffies_per_state
        self.last_delivered = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(num_actions)
        # self.observation_space = spaces.Box(
        #     low=0, high=np.inf, shape=(self.height_state, self.width_state), dtype=int)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.height_state, self.width_state), dtype=int)

        # Step counter
        self.steps_per_episode = steps_per_episode
        self.step_counter = 0

        # Keep current state
        self.curr_state = np.zeros((self.height_state, self.width_state))

        # Reward
        self.rws = dict() # list of rws for each step
        self.curr_reward = 0
        self.last_rtt = 0
        self.min_thr = 0
        self.min_rtt = sys.maxsize
        self.last_cwnd = 0
        self.epoch = -1
        self.allow_save = False
        self.step_wait = step_wait_seconds
        self.mss = None
        self.reward_name = reward_name
        self.max_bw = 0.0
        self.num_features_tmp = self.width_state + 2 # all features to be normalized (this is not the state dimension)
        self.normalizer = Normalizer(self.num_features_tmp)
        self.cwnd = []

    def update_rtt(self, rtt: float) -> None:

        if rtt > 0:
            self.last_rtt = rtt

    def _get_state(self):
        """
        Get the state from the mab environment.
        This function is called after the action has been taken.
        It gets network statistics from the kernel, normalizes them and compute the reward for that action.
        To get the state, we designed our algorithm such that the RL module is able to keep up with the speed of the network.
        In order to achieve this goal, we "slow down" our algorithm. It means that each observation is stored into an array which is filled 
        for step_wait seconds (the time we keep a single protocol executing within Mimic). 
        For each observation we compute the associated reward for the action taken. 
        After step_wait seconds, we compute the mean reward.

        The function returns:

        state_n: the normalized observation from the environment
        action: the action taken
        rws: the list of rewards (one for each observed state during the step_wait time)
        binary_rws: as above but the reward values are binary (1 if improved w.r.t the previous step, 0 o.w.)
        """
        s_tmp = np.array([])
        state_n = np.array([])
        received_jiffies = 0

        # Callbacks data
        self.features = []
        rws = []
        binary_rws = []

        
        if self.with_kernel_thread:
            print("[DEBUG] reading data from kernel...")
            timestamp, cwnd, rtt, rtt_dev, rtt_min, self.mss, delivered, lost, in_flight, retrans, action = self._read_data()
        else:
            timestamp, cwnd, rtt, rtt_dev, rtt_min, self.mss, delivered, lost, in_flight, retrans, action = self._recv_data()

        
        # Compute thr and loss rate from kernel statistics
        rtt = rtt if rtt > 0 else 1e-5
        thr = cwnd * self.mss / (rtt/self.nb)
        loss_rate = lost / (lost + delivered)

        # print(f"[DEBUG] {delivered}, {self.mss}, {cwnd}, {rtt}, {thr}, {cwnd*self.mss/(rtt/self.nb)}")

        print("[DEBUG] current CWND=", cwnd)

        delivered_diff = delivered - self.last_delivered
        self.last_delivered = delivered
        self.update_rtt(rtt)

        # to ms
        rtt = rtt/1000
        rtt_dev = rtt_dev/1000
        rtt_min = rtt_min/1000

        curr_kernel_features = np.array(
            [cwnd, rtt, rtt_dev, delivered, delivered_diff, loss_rate, in_flight, retrans, thr, rtt_min])
        
        self.features.append(curr_kernel_features)

        # Number of total features which will be extracted during the step_wait time (switching time) 
        num_features_tmp = self.num_features_tmp

        curr_timestamp = timestamp

        num_msg = 1

        start = time.time()

        # Read and record data for step_wait seconds
        while float(time.time() - start) <= float(self.step_wait):

            if self.with_kernel_thread:
                timestamp, cwnd, rtt, rtt_dev, rtt_min, self.mss, delivered, lost, in_flight, retrans, action = self._read_data()
            else:
                timestamp, cwnd, rtt, rtt_dev, rtt_min, self.mss, delivered, lost, in_flight, retrans, action = self._recv_data()

            # thruput bit/s
            rtt = rtt if rtt > 0 else 1e-5
            thr = (cwnd * self.mss) / (rtt/self.nb)
            loss_rate = lost / (lost + delivered)
          
            # print(f"[DEBUG] {delivered}, {self.mss}, {cwnd}, {rtt}, {thr}, {cwnd*self.mss/(rtt/self.nb)}")
            

            delivered_diff = delivered - self.last_delivered
            self.last_delivered = delivered
            self.update_rtt(rtt)

            # To ms
            rtt = rtt/1000
            rtt_dev = rtt/1000
            rtt_min = rtt_min/1000


            if timestamp != curr_timestamp:
                curr_kernel_features = np.divide(curr_kernel_features, num_msg)

                if s_tmp.shape[0] == 0:
                    s_tmp = np.array(curr_kernel_features).reshape(
                        1, num_features_tmp)
                else:
                    s_tmp = np.vstack(
                        (s_tmp, np.array(curr_kernel_features).reshape(1, num_features_tmp)))

                curr_kernel_features = np.array(
                    [cwnd, rtt, rtt_dev, delivered, delivered_diff, loss_rate, in_flight, retrans, thr, rtt_min])
                
                curr_timestamp = timestamp

                num_msg = 1

                received_jiffies += 1

            else:
                # sum new reading to existing readings
                curr_kernel_features = np.add(curr_kernel_features,
                            np.array([cwnd, rtt, rtt_dev, delivered, delivered_diff, loss_rate, in_flight, retrans, thr, rtt_min]))
                num_msg += 1

        # Kernel features for callbacks
        self.features = s_tmp

        # Normalize the state (from ORCA) -> normalization is relative to the entire session
        for s in s_tmp:
            self.normalizer.observe(s)
            s_n = self.normalizer.normalize(s)
            min_ = self.normalizer.stats()

            # print("[DEBUG] s_n", s_n)
            # print("[DEBUG] min", min_)

            cwnd_n_min = s_n[0] - min_[0]
            rtt_ms_n_min = s_n[1] - min_[1]
            rtt_ms_dev_n_min = s_n[2] - min_[2]
            delivered_n_min = s_n[3] - min_[3]
            delivered_diff_n_min = s_n[4] - min_[4]
            loss_rate_n_min = s_n[5] - min_[5]
            in_flight_n_min = s_n[6] - min_[6]
            retrans_n_min = s_n[7] - min_[7]
            thr_n_min = s_n[8] - min_[8]
            min_rtt_min = s_n[9] - min_[9]

            if self.max_bw<thr_n_min:
                self.max_bw=thr_n_min

            # cwnd, rtt, rtt_dev, delivered, delivered_diff, loss_rate, in_flight, retrans
            kernel_feat_n = np.array(
                [cwnd_n_min, rtt_ms_n_min, rtt_ms_dev_n_min, delivered_n_min, delivered_diff_n_min, loss_rate_n_min,
                    in_flight_n_min, retrans_n_min])

            # TODO: check min_rtt_min time unit (has to be ms)
            if min_rtt_min*1.25 < rtt_ms_n_min:
                delay_metric=(min_rtt_min*1.25)/rtt_ms_n_min
            else:
                delay_metric=1            

            # Compute the reward for each observation
            if self.max_bw!=0:
                rw = (thr_n_min-5*loss_rate_n_min)/self.max_bw*delay_metric
            else:
                rw = 0.0
            rws.append(rw)
            binary_rw = 0 if rw <= self.curr_reward else 1
            binary_rws.append(binary_rw)

            if state_n.shape[0] == 0:
                state_n = np.array(kernel_feat_n).reshape(
                        1, self.width_state)
            else:
                state_n =  np.vstack(
                        (state_n, np.array(kernel_feat_n).reshape(1, self.width_state)))

        # TODO: mean of rewards; could we do better?
        # The following aggregated value refers to the mean of the rewards computed during step_wait (switching time) within the step
        reward = np.mean(rws)
        self.curr_reward = reward
        self.curr_state = state_n

        # Padding
        print("[INFO] Data Size:", state_n.shape[0])
        if state_n.shape[0] > self.height_state:
            state_n = state_n[-self.height_state:, :]
        else:
            padding_len = self.height_state - state_n.shape[0]
            median_values = np.median(state_n, axis=0)
            padding = np.tile(median_values, (padding_len, 1))
            state_n = np.concatenate((padding, state_n), axis=0)
        
        return (state_n, action, rws, binary_rws)

    def record(self, state, reward, step, action):
        cwnd, rtt, rtt_dev, delivered, delivered_diff, lost, in_flight, retrans = state

        if cwnd == 0:
            return

        cwnd_diff = cwnd - self.last_cwnd

        self.last_cwnd = cwnd

        self.log_traces = f'{self.log_traces}\n {action}, {cwnd}, {rtt}, {rtt_dev}, {delivered}, {delivered_diff}, {lost}, {in_flight}, {retrans}, {cwnd_diff}, {step}, {round(self.curr_reward, 4)}, {round(reward, 4)}'

    def step(self, action):
        self._change_cca(int(action))
        observation, observed_action, rewards, binary_rewards = self._get_state()
        # binary_rewards, rewards = self._get_reward()
        done = False if self.step_counter != self.steps_per_episode-1 else True
        self.step_counter = (self.step_counter+1) % self.steps_per_episode

        avg_reward = round(np.mean(rewards), 4)
        avg_binary_reward = np.bincount(binary_rewards).argmax()

        info = {'reward': avg_reward, 'obs': self.features}
        data = {'rewards': binary_rewards, 'normalized_rewards': rewards, "features": self.features, 'obs': observation}

        print(
            f'\nStep: {self.step_counter} \t Sent Action: {action} \t Received Action: {observed_action} \t Epoch: {self.epoch} | Reward: {avg_reward} ({np.mean(avg_binary_reward)})')

        counter = self.step_counter if self.step_counter != 0 else self.steps_per_episode

        step = self.epoch * self.steps_per_episode + counter

        just_observed = np.mean(observation, axis=0)

        if self.allow_save:
            self.record(just_observed, avg_binary_reward,
                        step, observed_action)

        if not self.moderator.can_start() and step > 1:
            done = True

        return observation, avg_reward, done, info

    def reset(self):
        self._init_communication()
        self._change_cca(0)
        observation, _, _, _ = self._get_state()

        self.epoch += 1

        # data = {'obs': observation}

        return observation  # reward, done, info can't be included


class Normalizer():
    def __init__(self, input_dim):
        # self.params = params
        # self.config = config
        self.n = 1e-5
        num_inputs = input_dim
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)
        self.dim = num_inputs
        self.min = np.zeros(num_inputs)

    def observe(self, x):
        self.n += 1
        last_mean = np.copy(self.mean)
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = self.mean_diff/self.n
        # Check for zero standard deviation and set it to a small value
        for i in range(self.dim):
            if self.var[i] == 0:
                self.var[i] = 1e-5

    def normalize(self, inputs):
        obs_std = np.sqrt(self.var)
        a=np.zeros(self.dim)
        if self.n > 2:
            a=(inputs - self.mean)/obs_std
            for i in range(0,self.dim):
                if a[i] < self.min[i]:
                    self.min[i] = a[i]
            return a
        else:
            return np.zeros(self.dim)

    def normalize_delay(self,delay):
        obs_std = math.sqrt(self.var[0])
        if self.n > 2:
            return (delay - self.mean[0])/obs_std
        else:
            return 0

    def stats(self):
        return self.min
