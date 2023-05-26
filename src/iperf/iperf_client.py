"""
Author: Lorenzo Pappone
Year: 2023

Iperf Client Thread

In order to execute iperf in the context of mahimahi we run a python script after calling mm-link.
The python script run the iperf3 client with parameters.
To kill the process, the iperf runner in the script store its pid in a file and this client kills it
when the training is over.
"""

import os
import sys
import threading
from concurrent.futures import thread
import traceback, signal

from helper import context, utils, moderator, debug
from helper.subprocess_wrappers import Popen, call, check_output, print_output, check_call
from helper.mahimahi_trace import MahimahiTrace
import subprocess


class IperfClient(threading.Thread):

    def __init__(self, trace: MahimahiTrace, ip: str, time: int, log_file: str, moderator: moderator.Moderator) -> None:
        threading.Thread.__init__(self)

        self.trace: MahimahiTrace = trace
        self.ip = ip
        self.time = time
        self.log_file = log_file
        self.moderator = moderator
        self.ps = None
        self._pid_file = os.path.join(context.src_dir, "pid.txt")

    def _ip_forwarding_set(self) -> bool:
        cmd = ['sysctl', 'net.ipv4.ip_forward']

        res = check_output(cmd)

        val = res.strip().decode('utf-8')

        return val == 'net.ipv4.ip_forward = 1'

    def _set_ip_forwarding(self):

        if self._ip_forwarding_set():
            print('IP forwarding is already set\n')
            return

        cmd = ['sudo', 'sysctl', '-w', 'net.ipv4.ip_forward=1']
        res = call(cmd)

        if res != 0:
            raise Exception("Unable to set ipv4 forwarding")

    def _get_mahimahi_cmd(self):

        cmd = ['mm-link']

        utils.check_dir(os.path.join(context.entry_dir, 'log/mahimahi'))

        filename_up = f'{self.trace}.{utils.time_to_str()}.up.log'
        filename_down = f'{self.trace}.{utils.time_to_str()}.down.log'
        
        if debug.is_debug_on():
            print("Debug mode on")
            filename_up = debug.change_name(filename_up)
            filename_down = debug.change_name(filename_down)
        else:
            print("Debug mode off")
        
        uplink = os.path.join(context.entry_dir, 'log/mahimahi', filename_up)
        downlink = os.path.join(context.entry_dir, 'log/mahimahi', filename_down)

        up_path, down_path = MahimahiTrace.path(self.trace)

        print("[DEBUG] up path, down path", up_path, down_path)

        cmd = ['mm-link',
               up_path,
               down_path,
               f'--uplink-log={uplink}', f'--downlink-log={downlink}'
               ]

        return cmd

    def _get_iperf_cmd(self):

        return [
            'python3',
            f'{context.src_dir}/iperf.py',
            self.ip,
            str(self.time),
            self.log_file,
            self._pid_file
        ]
    

    def run(self) -> None:
        try:

            self._set_ip_forwarding()

            cmd = self._get_mahimahi_cmd() + self._get_iperf_cmd()
            print("[DEBUG] Command executing:", cmd)

            if self.trace == MahimahiTrace.none:
                cmd = self._get_iperf_cmd()

            self.moderator.start()

            check_call(cmd)

            self.moderator.stop()

            print("mahimahi experiment ran successfully\n")

        except Exception as _:
            print('\n')

            print(traceback.format_exc())
            self.moderator.stop()

    # This stop function is set when total episodes time in training > iperf time, so I need to stop manually the client when training is finished
    # This temporarily solution let me run the training with a specific number of episodes (and steps per episodes), while iperf is always running
    # TODO: iperf process should check the moderator which could be stopped by the RL-module.
    def stop(self) -> None:
        # Read the PID from the file
        if os.path.exists(self._pid_file):
            with open(self._pid_file, 'r') as f:
                # print("Getting pid from", self._pid_file)
                pid = int(f.read().strip())
                # print("Client PID:", pid)
            # Kill the client process if moderator is on
                if not(self.moderator.is_stopped()):
                    os.kill(pid, signal.SIGTERM)
                    print("Iperf client killed")
                # Remove the pid.txt file
                os.remove(self._pid_file)
        
