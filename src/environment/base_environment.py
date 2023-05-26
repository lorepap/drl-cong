import configparser
import os
import sys
import time
import traceback
from datetime import datetime

import gym
import numpy as np
from gym import spaces
from helper import context, utils
from network.kernel_feedback import KernelRequest
from network.netlink_communicator import NetlinkCommunicator
from helper.moderator import Moderator


class BaseEnvironment(gym.Env):
    '''Kernel Environment that follows gym interface'''
    metadata = {'render.modes': ['human']}

    def __init__(self, comm: NetlinkCommunicator, num_fields_kernel: int, moderator: Moderator=None):
        super(BaseEnvironment, self).__init__()

        # Netlink communicator
        self.netlink_communicator = comm

        self.with_kernel_thread = True
        self.num_fields_kernel = num_fields_kernel

        self.now_str = utils.time_to_str()
        self.log_traces = ""
        self.allow_save = False
        self.floating_error = 1e-12
        self.nb = 1e6
        self.initiated = False
        self.curr_reward = 0
        self.moderator = moderator

    def _init_communication(self):

        if not self.initiated:
            print("Initiating communication...")

            # Thread for kernel info
            self.kernel_thread = KernelRequest(
                self.netlink_communicator, self.num_fields_kernel)

            self.kernel_thread.start()

            print("Communication initiated")
            self.initiated = True

    def _change_cca(self, action):

        msg = self.netlink_communicator.create_netlink_msg(
            'SENDING ACTION', msg_flags=self.netlink_communicator.ACTION_FLAG, msg_seq=action)

        self.netlink_communicator.send_msg(msg)

    def _end_communication(self):
        try:
            print("Closing communication...")

            # Close thread
            self.kernel_thread.exit()
            self.initiated = False

            print("Communication closed")

        except Exception as err:
            print(traceback.format_exc())

    def _recv_data(self):
        msg = self.netlink_communicator.recv_msg()
        data = self.netlink_communicator.read_netlink_msg(msg)
        split_data = data.decode(
            'utf-8').split(';')[:self.num_fields_kernel]
        return list(map(int, split_data))

    def _read_data(self):
        kernel_info = self.kernel_thread.queue.get()
        self.kernel_thread.queue.task_done()
        return kernel_info

    def _get_state(self):
        raise NotImplementedError()

    def _get_reward(self):
        raise NotImplementedError()

    def record(self, state, reward, step, action):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def enable_log_traces(self):
        self.allow_save = True

    def save_log(self, model_name: str, log_path: str) -> str:

        log_file_name = f'{model_name}.{self.now_str}.csv'
        log_fullpath = os.path.join(context.entry_dir, log_path, log_file_name)

        with open(log_fullpath, 'w') as writer:
            writer.write(self.log_traces)

        return log_file_name, log_fullpath

    def reset(self):
        raise NotImplementedError()

    def render(self, mode='human'):
        print(f'Reward = {self.curr_reward}')

    def close(self):
        self._end_communication()
