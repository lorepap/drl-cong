import os
import traceback
from typing import Any
from helper.subprocess_wrappers import call, Popen, check_output, print_output
from helper.moderator import Moderator
from helper.debug import set_debug, is_debug_on, change_name
from iperf.iperf_client import IperfClient
from iperf.iperf_server import IperfServer
from helper import context, utils
from helper.mahimahi_trace import MahimahiTrace
from network.netlink_communicator import NetlinkCommunicator
from time import time

# Base class for Trainer
class Starter():

    def __init__(self, trace, iperf_dir, ip, iperf_time) -> None:
        # self.args = args
        self.init_kernel()
        self.netlink_communicator = NetlinkCommunicator()
        # init communication
        self.init_communication()
        self.client: IperfClient = None
        self.server: IperfServer = None
        self.moderator: Moderator = Moderator(use_iperf=True)
        self.trace = trace
        self.started = False
        self.iperf_dir = iperf_dir
        self.trace = trace
        self.ip = ip
        self.iperf_time = iperf_time

    def is_kernel_initialized(self) -> bool:
        cmd = ['cat', '/proc/sys/net/ipv4/tcp_congestion_control']

        res = check_output(cmd)

        protocol = res.strip().decode('utf-8')

        return protocol == 'mimic'

    def init_kernel(self):

        if self.is_kernel_initialized():
            print('Kernel has already been initialized\n')
            return

        cmd = os.path.join(context.src_dir, 'init_kernel.sh')

        # make script runnable
        res = call(['chmod', '755', cmd])
        if res != 0:
            raise Exception('Unable to set run permission\n')

        res = call(cmd)
        if res != 0:
            raise Exception('Unable to init kernel\n')
        
    def start_server(self, tag):
        base_path = os.path.join(context.entry_dir, "log", "iperf", "server")
        filename = f'server.log'
        if is_debug_on():
            filename = change_name(filename)
        log_filename = f'{base_path}/{filename}'
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        self.server = IperfServer(log_filename)
        self.server.start()

    # Initialize an IperfClient object for the experiment with an input mahimahi trace (from args)
    def start_client(self, tag: str, pid_file: str = None) -> str:

        if self.moderator.use_iperf != 1:
            self.moderator.start()
            return

        base_path = os.path.join(context.entry_dir, self.iperf_dir)
        utils.check_dir(base_path)

        filename = f'{tag}.{utils.time_to_str()}.json'
        
        if is_debug_on():
            filename = change_name(filename)
        
        log_filename = f'{base_path}/{filename}'
        
        self.client = IperfClient(MahimahiTrace.fromString(
            self.trace), self.ip, self.iperf_time, log_filename, self.moderator, )

        self.client.start()
        return log_filename
    
    
    def start_communication(self, tag):
        self.start_server(tag)
        self.start_client(tag)
    
    def change_iperf_logfile_name(old_name: str, new_name: str) -> None:
        try:
            new_file = new_name.replace("csv", "json")
            os.rename(old_name, new_file)

        except Exception as _:
            print('\n')
            print(traceback.print_exc())

    def init_communication(self):
        print("Initiating communication...")

        msg = self.netlink_communicator.create_netlink_msg(
            'INIT_COMMUNICATION', msg_flags=self.netlink_communicator.INIT_COMM_FLAG)

        self.netlink_communicator.send_msg(msg)

        print("Communication initiated")
        self.started = True

        self.nchoices, self.nprotocols = utils.get_number_of_actions(
            self.netlink_communicator)

        print(
            f'\n\n----- Number of protocols available is {self.nchoices} ----- \n\n')

    def close_communication(self) -> None:

        msg = self.netlink_communicator.create_netlink_msg(
            'END_COMMUNICATION', msg_flags=self.netlink_communicator.END_COMM_FLAG)

        self.netlink_communicator.send_msg(msg)
        self.netlink_communicator.close_socket()

        print("Communication closed")

    def stop_communication(self):
        self.client.stop()
        self.server.stop()

    def get_nchoices(self):
        if self.started:
            return self.nchoices
        raise RuntimeError("Communication not started. Start communication first.")
