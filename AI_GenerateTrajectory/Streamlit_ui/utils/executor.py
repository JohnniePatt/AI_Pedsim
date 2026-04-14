import subprocess
import os
import signal
import sys
import threading
import queue

class ProcessManager:
    def __init__(self):
        self.process = None
        self.output_queue = queue.Queue()
        self.is_running = False

    def start_process(self, command, cwd):
        """
        Starts a subprocess with the given command and working directory.
        """
        if self.is_running:
            return False
        
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        self.is_running = True
        
        # Start a thread to read output
        thread = threading.Thread(target=self._read_output)
        thread.daemon = True
        thread.start()
        return True

    def _read_output(self):
        """
        Reads output from the subprocess and puts it into the queue.
        """
        for line in iter(self.process.stdout.readline, ''):
            self.output_queue.put(line)
        self.process.stdout.close()
        self.process.wait()
        self.is_running = False
        self.output_queue.put(None) # Sentinel to signal end of process

    def get_output(self):
        """
        Yields all available output from the queue.
        """
        while not self.output_queue.empty():
            line = self.output_queue.get()
            if line is None:
                break
            yield line

    def stop_process(self):
        """
        Stops the running subprocess.
        """
        if self.process and self.is_running:
            if os.name != 'nt':
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
            self.is_running = False
            return True
        return False
