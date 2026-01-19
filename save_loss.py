import os
from config import config

class LossLogger:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        os.makedirs(self.folder_path, exist_ok=True)
        self.loss_file = os.path.join(self.folder_path, "loss.txt")

        # Ensure the file exists
        if not os.path.exists(self.loss_file):
            open(self.loss_file, "w").close()

    def log(self, loss: float):
        if not isinstance(loss, (float, int)):
            raise ValueError("Loss must be a float or int")

        with open(self.loss_file, "a") as f:
            f.write(f"{float(loss)}\n")
            
logger = LossLogger(folder_path=config.SAVE_PATH)
