from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
from mil.metrics import F1, Sensitivity, Specificity
from train_scripts.model import Classifier
from train_scripts.utils import (
    earlyStopping,
    load_config,
    input_to_generators,
    clearmlLogger,
)
from train_scripts.data import Dataset


class modelTrainer:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        assert (
            config_path.split(".")[0][-4:] == self.config["model"]["exp_version"]
        ), "version number not matching!"
        self.model_params = {}
        self._initialize_params()
        self._exp_name = "{}-{}".format(
            self.config["model"]["name"], self.config["model"]["exp_version"]
        )
        self.model = Classifier(self.model_params)
        self.early_stopping = earlyStopping(
            self.model_params["es_tolerance"], self.model_params["es_min_delta"]
        )
        self.clearml = clearmlLogger(
            params=self.model_params,
            project_name=self.config["project_name"],
            task_name=self._exp_name,
            tags=self.config["model"]["exp_tags"],
        )
        self.device = torch.device("cuda:0")
        self.model = self.model.to(self.device)
        self.criterion = (torch.nn.BCEWithLogitsLoss(reduction='sum'))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.model_params["base_lr"]
        )
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=self.model_params["base_lr"],
            max_lr=self.model_params["max_lr"],
            scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
            cycle_momentum=False,
            step_size_up=208,
        )
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.5)
        self.ckpt_path = os.path.join(
            self.config["misc"]["checkpoint_path"], self._exp_name
        )
        # self.sensitivity = Sensitivity()
        # self.specificity = Specificity()
        # self.f1 = F1()
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

    def _initialize_params(self):
        self.model_params["base_model"] = self.config["model"]["base_model"]
        self.model_params["pretrained"] = self.config["model"]["pretrained"]
        self.model_params["finetune"] = self.config["model"]["finetune"]
        self.model_params["n_classes"] = self.config["data"]["n_classes"]
        self.model_params["batch_size"] = self.config["train"]["batch_size"]
        self.model_params["base_lr"] = self.config["train"]["base_lr"]
        self.model_params["max_lr"] = self.config["train"]["max_lr"]
        self.model_params["class_weights"] = self.config["train"]["class_weights"]
        self.model_params["epochs"] = self.config["train"]["epochs"]
        self.model_params["es_tolerance"] = self.config["early_stopping"]["tolerance"]
        self.model_params["es_min_delta"] = self.config["early_stopping"]["min_delta"]
        self.model_params["shuffle"] = self.config["train"]["shuffle"]
        self.model_params["n_workers"] = self.config["data"]["n_workers"]
        self.model_params["data_transform"] = self.config["data"]["transform"]
        self.model_params["genomic"] = self.config["model"]["genomic"]
        # self.model_params["crop"] = self.config["data"]["crop"]

    def _data_generators(self):
        train_meta, valid_meta = input_to_generators(
            self.config["data"]["csv_meta_path"], self.config["data"]["remove_blank"]
        )
        train_generator = DataLoader(
            Dataset(
                data_path=self.config["data"]["base_path"],
                df_meta=train_meta,
                gender_list=self.config["data"]["gender_list"],
                mutant_list=self.config["data"]["mutant_list"],
                transform=self.model_params["data_transform"],
                genomic=self.config["model"]["genomic"]
            ),
            batch_size=self.model_params["batch_size"],
            shuffle=self.model_params["shuffle"],
            num_workers=self.model_params["n_workers"],
        )
        validation_generator = DataLoader(
            Dataset(
                data_path=self.config["data"]["base_path"],
                df_meta=valid_meta,
                gender_list=self.config["data"]["gender_list"],
                mutant_list=self.config["data"]["mutant_list"],
                transform=False,
                genomic=self.config["model"]["genomic"]
            ),
            batch_size=self.model_params["batch_size"],
            shuffle=False,
            num_workers=self.model_params["n_workers"],
        )
        return train_generator, validation_generator

    def train_model(self):
        print("creating train and validation data generators")
        # factor = 5
        train_generator, validation_generator = self._data_generators()
        print("Initializing training")
        for i in range(self.model_params["epochs"]):
            train_loss = 0
            batch_count = 0
            self.model.train()
            correct = 0
            total = 0
            with tqdm(
                desc="Training epoch no. {} progress".format(i),
                total=len(train_generator),
            ) as pbar:
                for idx, data in enumerate(train_generator):
                    inputs, gender, age, mutant, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # gender = gender.to(self.device)
                    # age = age.to(self.device)
                    # mutant = mutant.to(self.device)
                    extras = torch.tensor([gender, age, mutant])
                    extras = extras.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs.float(), extras.float())
                    probs = torch.nn.Sigmoid()(outputs)
                    preds = (probs>=0.5)*1.0
                    correct += (preds == labels).sum().item()
                    total += labels.shape[0]
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    train_loss += loss.item()
                    batch_count += 1
                    if idx % 1 == 0:
                        niter = i * len(train_generator) + idx
                        self.clearml.logger.add_scalar("Train Loss", train_loss/batch_count, niter)
                        self.clearml.logger.add_scalar(
                            "Train Accuracy",
                            correct / total,
                            niter,
                        )
                        # self.clearml.logger.add_scalar(
                        #     "Train Sensitivity",
                        #     self.sensitivity(labels, preds),
                        #     niter,
                        # )
                        # self.clearml.logger.add_scalar(
                        #     "Train Specificity",
                        #     self.specificity(labels, preds),
                        #     niter,
                        # )
                        # self.clearml.logger.add_scalar(
                        #     "Train F1 score",
                        #     self.f1(labels, preds),
                        #     niter,
                        # )
                        self.clearml.logger.add_scalar(
                            "Learning Rate",
                            self.optimizer.param_groups[0]["lr"],
                            niter,
                        )
                    if idx == len(train_generator) - 1:
                        torch.save(
                            {
                                "epoch": i,
                                "iteration": niter,
                                "model_state_dict": self.model.state_dict(),
                            },
                            os.path.join(
                                self.ckpt_path,
                                "{}_iteration{}.ckpt".format(self._exp_name, niter),
                            ),
                        )
                    pbar.update(1)
            train_loss /= batch_count
            train_accuracy = correct / total
            self.model.eval()
            val_loss = 0
            batch_count = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in validation_generator:
                    inputs, gender, age, mutant, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # gender = gender.to(self.device)
                    # age = age.to(self.device)
                    # mutant = mutant.to(self.device)
                    extras = torch.tensor([gender, age, mutant])
                    extras = extras.to(self.device)
                    outputs = self.model(inputs.float(), extras.float())
                    probs = torch.nn.Sigmoid()(outputs)
                    preds = (probs>=0.5)*1.0
                    correct += (preds == labels).sum().item()
                    total += labels.shape[0]
                    loss = self.criterion(outputs, labels)
                    batch_count += 1
                    val_loss += loss.item()
            val_loss /= batch_count
            val_accuracy = correct / total
            print(
                "epoch no. {} - training loss {} - training accuracy {} - validation loss {} - validation accuracy {}".format(
                    i, train_loss, train_accuracy, val_loss, val_accuracy
                )
            )
            self.clearml.logger.add_scalar(
                "Validation Loss", val_loss, (i + 1) * len(train_generator)
            )
            self.clearml.logger.add_scalar(
                "Validation Accuracy", val_accuracy, (i + 1) * len(train_generator)
            )
            self.early_stopping(train_loss, val_loss)
            if self.early_stopping.early_stop:
                print("stopping early at epoch {}".format(i))
                break
        print("Training Finished!")
