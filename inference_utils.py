from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score
from utils import create_conf_matrix_fig

def epoch_step(model, data_loader, criterion, device, mode='train', optimizer=None):
    if mode == 'train':
        model.train()
    elif mode == 'val':
        model.eval()

    loss_avg = 0.
    preds, targets = [], []
    conf_mat_fig = None
    for x_batch, y_batch in tqdm(data_loader):
        targets.extend(y_batch.tolist())
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        pred = pred.max(1)[1].detach().cpu().tolist()
        preds.extend(pred)
        loss_avg += loss.item() / len(data_loader)
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    f1, acc = f1_score(targets, preds, average='macro'), balanced_accuracy_score(targets, preds)
    if mode == 'val':
        conf_mat_fig = create_conf_matrix_fig(targets, preds)
    
    return loss_avg, f1, acc, conf_mat_fig