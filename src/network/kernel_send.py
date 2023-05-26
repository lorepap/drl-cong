import threading
import time
import random


class ThreadKernelSend(threading.Thread):
    def __init__(self, comm):
        threading.Thread.__init__(self)
        self.value = 0
        self.comm = comm
        self.last = None

    def run(self):
        while True:
            msg = self.comm.create_netlink_msg(
                'SENDING ACTION', msg_flags=self.comm.ACTION_FLAG, msg_seq=self.value)
            
            if (self.value != self.last):
                self.comm.send_msg(msg)
                print(f"Sending new action. Action ID = {self.value}")
                self.last = self.value

            time.sleep(5)

    def setValue(self, val):
        self.value = val