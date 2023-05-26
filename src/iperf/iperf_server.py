import subprocess
import psutil

class IperfServer:
    def __init__(self, log_file):
        self._server_proc = None
        self._server_log_file = log_file

    def start(self):
        if self._server_proc is not None:
            raise Exception("Server already running")
        print("[SERVER] Server log:", self._server_log_file)
        server_cmd = ["iperf3", "-s", "--logfile", self._server_log_file]
        
        try:
        
            self._server_proc = subprocess.Popen(server_cmd, shell=False)
        
        except Exception as e:
            # Kill all iperf3 processes if an error is raised
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == 'iperf3':
                    proc.kill()
            raise e

        print("Server started with PID:", self._server_proc.pid)

    def stop(self):
        if self._server_proc is not None:
            self._server_proc.terminate()
            self._server_proc.wait()
            self._server_proc = None
            print("Server stopped")
        else:
            print("Server not running")

    def restart(self):
        self.stop()
        self.start()
