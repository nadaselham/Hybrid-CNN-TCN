import time

class Logger:
    def __init__(self):
        self.start_time = time.time()

    def log_epoch(self, epoch):
        print(f"Epoch {epoch} completed in {time.time() - self.start_time:.2f}s")
