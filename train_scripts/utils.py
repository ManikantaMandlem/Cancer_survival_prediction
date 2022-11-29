import yaml
import pandas as pd
import random
from clearml import Task
from torch.utils.tensorboard import SummaryWriter


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        return config


def input_to_generators(csv_path, remove_blank):
    meta_df = pd.read_csv(csv_path)
    if remove_blank:
        meta_df = meta_df.dropna()
    train_meta = meta_df[meta_df["set"] == "train"]
    valid_meta = meta_df[meta_df["set"] == "validation"]
    return train_meta, valid_meta


class earlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class clearmlLogger:
    def __init__(self, params, project_name, task_name, tags):
        self.params = params
        self.project_name = project_name
        self.task_name = task_name
        self.tags = tags
        self.logger = self.get_logger()

    def get_logger(self):
        # with open(self.api_key_path, 'r') as f:
        #     _clearml_keys = yaml.safe_load(f)
        # for k, v in _clearml_keys.items():
        #     os.environ[k] = v
        task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            reuse_last_task_id=False,
            tags=self.tags,
        )
        task.connect(self.params)
        writer = SummaryWriter("ECE_495 Project")
        return writer
