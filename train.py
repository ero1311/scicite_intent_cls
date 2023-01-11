from model import SciLSTM, SciBertCls
from preprocess_dataset import get_data_loaders
from inference_utils import epoch_step
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import torch
import argparse

parser = argparse.ArgumentParser("SciCite training")
parser.add_argument('--batch_size', type=int, help='batch size', default=32)
parser.add_argument('--epoch', type=int, help='number of epochs', default=100)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('--weight_decay', type=float, help='weight decay', default=2e-4)
parser.add_argument('--dropout_rate', type=float, help='dropout rate', default=0.2)
parser.add_argument('--exp_name', type=str, help='base name of the experiment', default='SciCite')
parser.add_argument('--emb_type', type=str, help='algo to use for word/sentence embeddings', default='glove')
parser.add_argument("--is_weighting", action="store_true", help="use weighted cross entropy")
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, val_loader, test_loader = get_data_loaders(args.emb_type, args.batch_size)
model = None

if args.emb_type == 'glove':
    model = SciLSTM(dropout_rate=args.dropout_rate)
elif args.emb_type == 'scibert':
    model = SciLSTM(dropout_rate=args.dropout_rate, emb_dim=768)
else:
    raise NotImplementedError

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.is_weighting:
    weights = train_loader.dataset.tensors[1].unique(return_counts=True)[1]
    weights = weights.float()
    weights /= weights.sum()
    weights = 1 / torch.log(1.2 + weights)
    weights = weights.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
else:
    criterion = torch.nn.CrossEntropyLoss()

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
    train_loss_avg, train_f1, train_acc, _ = epoch_step(model, train_loader, criterion, DEVICE, mode='train', optimizer=optimizer)

    #validation
    val_loss_avg, val_f1, val_acc, conf_mat = epoch_step(model, val_loader, criterion, DEVICE, mode='val')

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
    logger.add_figure("Confusion Matrix Val", conf_mat, epoch)