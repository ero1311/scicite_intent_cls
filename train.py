from model import SciLSTM
from preprocess_dataset import process_scicite
from utils import create_conf_matrix_fig
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, balanced_accuracy_score
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import torch
import argparse

parser = argparse.ArgumentParser("SciCite training")
parser.add_argument('--batch_size', type=int, help='batch size', default=32)
parser.add_argument('--epoch', type=int, help='number of epochs', default=100)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('--dropout_rate', type=float, help='dropout rate', default=0.2)
parser.add_argument('--exp_name', type=str, help='base name of the experiment', default='SciCite')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = process_scicite('./data')
train_dataset = TensorDataset(data['train'][0], data['train'][1])
val_dataset = TensorDataset(data['validation'][0], data['validation'][1])
test_dataset = TensorDataset(data['test'][0], data['test'][1])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = SciLSTM(dropout_rate=args.dropout_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
model.to(DEVICE)

exp_base_path = Path('./logs')
if not exp_base_path.exists():
    exp_base_path.mkdir()

curr_time = datetime.now()
exp_full_name = args.exp_name + '_' + curr_time.strftime("%d:%m:%Y:%H:%M:%S")
logger = SummaryWriter(exp_base_path / exp_full_name)
best_f1 = 0

for epoch in tqdm(range(args.epoch)):
    #training
    model.train()
    train_loss_avg, val_loss_avg = 0., 0.
    preds, targets = [], []
    for x_batch, y_batch in tqdm(train_loader):
        targets.extend(y_batch.tolist())
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        pred, loss = model(x_batch, y_batch)
        pred = pred.max(1)[1].detach().cpu().tolist()
        preds.extend(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_avg += loss.item() / len(train_loader)
    
    train_f1, train_acc = f1_score(targets, preds, average='macro'), balanced_accuracy_score(targets, preds)
    
    #validation
    model.eval()
    preds, targets = [], []
    for x_batch, y_batch in tqdm(val_loader):
        targets.extend(y_batch.tolist())
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        pred, loss = model(x_batch, y_batch)
        pred, loss = model(x_batch, y_batch)
        pred = pred.max(1)[1].detach().cpu().tolist()
        preds.extend(pred)
        val_loss_avg += loss.item() / len(val_loader)

    val_f1, val_acc = f1_score(targets, preds, average='macro'), balanced_accuracy_score(targets, preds)
    # update best weights
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), exp_base_path / exp_full_name / 'best.pth')

    # update tensorboard
    logger.add_scalars(
        'loss',
        {
            'train': train_loss_avg,
            'val': val_loss_avg
        },
        epoch
    )
    logger.add_scalars(
        'f1',
        {
            'train': train_f1,
            'val': val_f1
        },
        epoch
    )
    logger.add_scalars(
        'acc',
        {
            'train': train_acc,
            'val': val_acc
        },
        epoch
    )
    logger.add_figure("Confusion Matrix Val", create_conf_matrix_fig(targets, preds), epoch)