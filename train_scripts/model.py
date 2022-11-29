import torch.nn as nn
import torchvision.models as models
import torch


class Classifier(nn.Module):
    """
    Defines the pytorch models for the classifier using pretrained efficientnet_v2_s module
    """

    def __init__(self, params):
        super(Classifier, self).__init__()
        self.dim_changer = nn.Sequential(nn.Conv2d(in_channels=1280,out_channels=3,kernel_size=6, padding='same'))
        self.feature_extractor = eval(
            "models.{}(weights='{}')".format(params["base_model"], params["pretrained"])
        )
        if not params["finetune"]:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        self.feature_extractor.classifier = nn.Sequential()
        if params["genomic"]:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                # nn.Linear(in_features=1283, out_features=1283, bias=True),
                # nn.ReLU6(),
                # nn.Dropout(p=0.2, inplace=True),
                # nn.Linear(in_features=1280, out_features=1280, bias=True),
                # nn.ReLU6(),
                # nn.Linear(in_features=512, out_features=512, bias=True),
                # nn.ReLU6(),
                # nn.Linear(in_features=512, out_features=256, bias=True),
                # nn.ReLU6(),
                nn.Linear(
                    in_features=1283, out_features=params["n_classes"], bias=True
                ),
                # explore more about this later
            )
            self.genomic = True
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                # nn.Linear(in_features=1280, out_features=1280, bias=True),
                # nn.ReLU6(),
                # nn.Linear(in_features=1280, out_features=1280, bias=True),
                # nn.ReLU6(),
                # nn.Linear(in_features=512, out_features=512, bias=True),
                # nn.ReLU6(),
                # nn.Linear(in_features=512, out_features=256, bias=True),
                # nn.ReLU6(),
                nn.Linear(
                    in_features=1280, out_features=params["n_classes"], bias=True
                ),
                # explore more about this later
            )
            self.genomic = False

    def forward(self, x, extras):
        features = self.dim_changer(x)
        features = self.feature_extractor(features)
        features = torch.squeeze(features)
        if self.genomic:
            features = torch.cat((features, extras), dim=0)
        return self.classifier(features)
