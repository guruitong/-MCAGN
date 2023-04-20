from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, RandomSampler
from feeder import Feeder
from config import TRAIN_CFG
from model.model import Model
import torch.optim as optim
from torch import nn
import os
from sklearn.metrics import precision_recall_fscore_support

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def train(task='sentiment', checkpoint=None):
    assert (task == 'sentiment') or (task == 'emotion') or (task == 'desire')

    dataset = Feeder(train=True)
    sampler = BatchSampler(
        RandomSampler(dataset),
        batch_size=TRAIN_CFG['batch_size'],
        drop_last=False)
    loader = DataLoader(dataset, sampler=sampler, batch_size=None)

    model = Model().to(device)
    if checkpoint is not None:
        weights = torch.load(checkpoint)
        model.load_state_dict(weights)

    optimizer = optim.Adam([
        {'params': model.ViT.parameters(), 'lr': TRAIN_CFG['vit_lr']},
        {'params': model.bert.parameters(), 'lr': TRAIN_CFG['bert_lr']},
        {'params': model.fusion.parameters()},
        {'params': model.s_head.parameters()},
        {'params': model.e_head.parameters()},
        {'params': model.d_head.parameters()}],
        lr=TRAIN_CFG['base_lr'],
        weight_decay=TRAIN_CFG['wd']
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(TRAIN_CFG['epochs']):
        losses = []
        for txt_inputs, images, sentiments, emotions, desires in loader:
            sentiments = sentiments.type(torch.LongTensor).to(device)
            emotions = emotions.type(torch.LongTensor).to(device)
            desires = desires.type(torch.LongTensor).to(device)
            s, e, d = model(txt_inputs, images)
            if task == 'sentiment':
                loss = loss_fn(s, sentiments)
            elif task == 'emotion':
                loss = loss_fn(e, emotions)
            else:
                loss = loss_fn(d, desires)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        print(f'Epoch {epoch}, loss = {np.mean(losses)}')
        test(model, task)
    PATH = task + '.pt'
    torch.save(model.state_dict(), PATH)
    return model

def test(model=None, task='sentiment', checkpoint=None):
    assert (task == 'sentiment') or (task == 'emotion') or (task == 'desire')
    assert (model is None) or (checkpoint is None)
    assert not ((model is not None) and (checkpoint is not None))

    dataset = Feeder(train=False)
    sampler = BatchSampler(
        RandomSampler(dataset),
        batch_size=TRAIN_CFG['batch_size'],
        drop_last=False)
    loader = DataLoader(dataset, sampler=sampler, batch_size=None)

    if checkpoint is not None:
        weights = torch.load(checkpoint)
        model = Model()
        model.load_state_dict(weights)
        model = model.to(device)

    preds = []
    labels =[]
    model.eval()
    for txt_inputs, images, sentiments, emotions, desires in loader:
        with torch.no_grad():
            s, e, d = model(txt_inputs, images)
            if task == 'sentiment':
                pred = s.detach()
                label = sentiments
            elif task == 'emotion':
                pred = e.detach()
                label = emotions
            else:
                pred = d.detach()
                label = desires

        _, pred = torch.max(pred, 1)
        preds += pred.int().tolist()
        labels += label.reshape(-1).int().tolist()
    print(precision_recall_fscore_support(labels, preds, average='micro'))
    return

checkpoints_folder = 'results'
sentiments_chkpt = os.path.join(checkpoints_folder, 'sentiment.pt')
emotions_chkpt = os.path.join(checkpoints_folder, 'emotion.pt')
desires_chkpt = os.path.join(checkpoints_folder, 'desire.pt')


#train('sentiment', sentiments_chkpt)

test(task='sentiment', checkpoint=sentiments_chkpt)
test(task='emotion', checkpoint=emotions_chkpt)
test(task='desire', checkpoint=desires_chkpt)


