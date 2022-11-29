import torch
from torch.utils.data import DataLoader
import os
import sys
import pandas as pd
from torchmetrics.classification import BinaryAUROC
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryMatthewsCorrCoef

f1_metric = BinaryF1Score(threshold=0.5)
auc_metric = BinaryAUROC(thresholds=None)
mcc_metric = BinaryMatthewsCorrCoef(threshold=0.5)

import csv
file_=open('/home/mmandlem/ece_495/results/mobilenet_no_extra_data.csv','w')
writer = csv.writer(file_)
writer.writerow(['test_example','ground_truth', 'prediction','correct_prediction'])

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_path)
from train_scripts import Classifier, Dataset

csv_path = "/home/mmandlem/ece_495/data/dataset_ece495_2022.csv"
meta_df = pd.read_csv(csv_path)
meta_df = meta_df.dropna()
test_meta = meta_df[meta_df["set"] == "test"]

genomic = False
# ckpt_path = '/home/mmandlem/ece_495/checkpoints/cancer_survival_prediction-v002/cancer_survival_prediction-v002_iteration1809.ckpt' #0.74468 accuracy
ckpt_path = "/home/mmandlem/ece_495/checkpoints/cancer_survival_prediction-v007/cancer_survival_prediction-v007_iteration1809.ckpt" #0.53 accuracy
device = torch.device('cuda:0')

data_params={}
data_params['base_path']='/home/mmandlem/ece_495/data'
data_params['gender_list']=['male','female']
data_params['mutant_list']=['IDHmut-non-codel', 'IDHwt', 'IDHmut-codel']
data_params['data_transform']=False
data_params['genomic']=genomic

model_params = {}
model_params['base_model']='efficientnet_v2_s'
model_params['pretrained']='IMAGENET1K_V1'
model_params['finetune']=False
model_params['genomic']=genomic
model_params['n_classes']=1

test_generator = DataLoader(
    Dataset(
        data_path=data_params['base_path'],
        df_meta=test_meta,
        gender_list=data_params['gender_list'],
        mutant_list=data_params['mutant_list'],
        transform=data_params['data_transform'],
        genomic=data_params['genomic']
    ),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

model = Classifier(model_params)
weights = torch.load(ckpt_path)
model.load_state_dict(weights['model_state_dict'])
model = model.to(device)
model.eval()
correct = 0
total = 0
batch_count = 0
y_true = []
y_prob = []
with torch.no_grad():
    for data in test_generator:
        path, inputs, gender, age, mutant, labels = data
        path = path[0].split('/')[-1].replace('npz','tfr')
        inputs = inputs.to(device)
        labels = labels.to(device)
        extras = torch.tensor([gender, age, mutant])
        extras = extras.to(device)
        outputs = model(inputs.float(), extras.float())
        probs = torch.nn.Sigmoid()(outputs)
        preds = (probs>=0.5)*1.0
        y_true.append(labels.item())
        y_prob.append(probs.item())
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
        batch_count += 1
        writer.writerow([path,labels.item(),preds.item(),labels.item()==preds.item()])
file_.close()
        
test_accuracy = correct / total
auc_score = auc_metric(torch.tensor(y_prob), torch.tensor(y_true))
f1_score = f1_metric((torch.tensor(y_prob)>=0.5)*1, torch.tensor(y_true))
mcc_score = mcc_metric((torch.tensor(y_prob)>=0.5)*1, torch.tensor(y_true))
print("test_accuracy: {}, auc: {}, f1: {}, mcc: {}".format(test_accuracy,auc_score,f1_score,mcc_score))