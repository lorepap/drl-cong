import os
from pickle import TRUE
import socket
import struct
import traceback

NETLINK_TEST = 25

class NetlinkCommunicator:
    TEST_FLAG = 4
    INIT_SINGLE_PROT_TEST = 3
    ACTION_FLAG = 2
    INIT_COMM_FLAG = 1
    END_COMM_FLAG = 0

    def __init__(self):
        self.socket = self.init_socket()

    
    def init_socket(self):
        s = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, NETLINK_TEST)
        s.bind((os.getpid(), 0))
        return s

    def close_socket(self):
        self.socket.close()

    def create_netlink_msg(self, data, msg_type=0, msg_flags=0, msg_seq=0, msg_pid=os.getpid()):
        payload = f'{data}\0'
        header_size = 16
        payload_size = len(payload)
        msg_len = header_size + payload_size
        header = struct.pack("=LHHLL", msg_len, msg_type, msg_flags, msg_seq, msg_pid)
        msg = header + payload.encode()
        return msg

    def send_msg(self, msg):
        self.socket.send(msg)

    def recv_msg(self):
        try:
            return self.socket.recv(1024)
            # return self.socket.recv()
        except Exception as err:
            print('\n')
            print(traceback.format_exc())

            # clear buffer
            # self.socket.
            return None

    def read_netlink_msg(self, msg):
        value_len, value_type, value_flags, value_seq, value_pid = struct.unpack("=LHHLL", msg[:16])
        data = msg[16:value_len]
        return data
