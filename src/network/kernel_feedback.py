import queue
import threading
import traceback

from network.netlink_communicator import NetlinkCommunicator


class KernelRequest(threading.Thread):
    def __init__(self, comm: NetlinkCommunicator, num_fields_kernel: int):
        threading.Thread.__init__(self)
        self.comm = comm
        self.num_fields_kernel = num_fields_kernel
        self.queue = queue.Queue()
        self.can_read = True

    def run(self):
        while self.can_read:
            try:
                msg = self.comm.recv_msg()

                if msg:
                    data = self.comm.read_netlink_msg(msg)
                    data_decoded = data.decode('utf-8')

                    split_data = data_decoded.split(';')[:self.num_fields_kernel]
                    entry = list(map(int, split_data))

                    self.queue.put(entry)
            except Exception as _:
                print('\n')
                print(traceback.format_exc())

    def exit(self):
        self.can_read = False
