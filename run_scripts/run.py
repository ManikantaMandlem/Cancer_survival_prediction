import sys
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(home_dir)

from train_scripts.train import modelTrainer


def main(config_path):
    trainer_object = modelTrainer(config_path)
    trainer_object.train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", "-p", type=str, help="path to the configuration file"
    )
    args = parser.parse_args()
    main(args.config_path)
