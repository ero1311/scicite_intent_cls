from inference_utils import eval
from preprocess_dataset import get_data_loaders
from model import SciLSTM
from pathlib import Path
from datasets import load_dataset
import argparse
import torch

parser = argparse.ArgumentParser("SciCite training")
parser.add_argument('--exp_name', type=str, help='base name of the experiment', default='SciCite_scibert_2layer_wd2e-3_weighted_11:01:2023:13:32:44')
parser.add_argument('--emb_type', type=str, help='algo to use for word/sentence embeddings', default='scibert')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, _, test_loader = get_data_loaders(args.emb_type, 1)

if args.emb_type == 'scibert':
    model = SciLSTM(emb_dim=768)
else:
    model = SciLSTM(emb_dim=300)

log_base = Path('./logs')
model_weights = torch.load(log_base / args.exp_name / 'best.pth')
model.load_state_dict(model_weights)
model.to(DEVICE)

preds, f1, acc, conf_mat_fig = eval(model, test_loader, DEVICE)
print("RESULTS ON THE TEST SET: F1: {}, BALANCED ACCURACY: {}".format(f1, acc))
dataset = load_dataset('scicite', cache_dir='./data')
test_df = dataset['test'].to_pandas()
test_df['label_pred'] = preds
test_df.to_csv(log_base / args.exp_name / 'test_predictions.csv')
conf_mat_fig.savefig(log_base / args.exp_name / 'confusion_matrix_test.png')