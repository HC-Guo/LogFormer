import argparse
import time
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, f1_score
from tqdm import tqdm

from dataloader import DataGenerator
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--log_name', type=str,
                    default='HDFS', help='log file name')
parser.add_argument('--window_size', type=int,
                    default='50', help='log sequence length')
parser.add_argument('--mode', type=str, default='classifier',
                    help='use adapter or not')
parser.add_argument('--num_layers', type=int, default=1,
                    help='num of encoder layer')
parser.add_argument('--adapter_size', type=int, default=64,
                    help='adapter size')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument("--resume", type=int, default=0,
                    help="resume training of model (0/no, 1/yes)")
parser.add_argument("--load_path", type=str,
                    default='checkpoints/model-latest.pt', help="latest model path")
parser.add_argument("--num_samples", type=int,
                    default='50000', help="number of training samples")

args = parser.parse_args()
suffix = f'{args.log_name}_{args.mode}_{args.num_layers}_{args.adapter_size}_{args.lr}'
with open(f'result_{args.num_samples}/train_{suffix}.txt', 'w', encoding='utf-8') as f:
    f.write(str(args)+'\n')

# hyper-parameters
EMBEDDING_DIM = 768
batch_size = 64
epochs = 10
lr = args.lr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]
print('Using device = ', device)
print(f'Model mode is {args.mode}')

# fix all random seeds
warnings.filterwarnings('ignore')
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

# load data Hdfs
training_data = np.load(
    f'./preprocessed_data/{args.log_name}_training.npz', allow_pickle=True)
# load test data Hdfs
testing_data = np.load(
    f'./preprocessed_data/{args.log_name}_testing.npz', allow_pickle=True)
x_train, y_train = training_data['x'], training_data['y']
x_test, y_test = testing_data['x'], testing_data['y']
del testing_data
del training_data

train_generator = DataGenerator(
    x_train[:args.num_samples], y_train[:args.num_samples], args.window_size)
test_generator = DataGenerator(x_test, y_test, args.window_size)
train_loader = torch.utils.data.DataLoader(
    train_generator, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_generator, batch_size=batch_size, shuffle=False)

model = Model(mode=args.mode, num_layers=args.num_layers, adapter_size=args.adapter_size, dim=EMBEDDING_DIM, window_size=args.window_size, nhead=8, dim_feedforward=4 *
              EMBEDDING_DIM, dropout=0.1)
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=device_ids)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.7, patience=4, threshold=1e-4, verbose=True)
criterion = nn.BCEWithLogitsLoss()

start_epoch = -1
if args.resume == 1:
    path_checkpoint = args.load_path
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print("resume training from epoch ", start_epoch)


best_acc = np.inf
best_f1 = 0
log_interval = 100
for epoch in range(start_epoch+1, epochs):
    loss_all, f1_all = [], []
    train_loss = 0
    train_pred, train_true = [], []

    model.train()
    start_time = time.time()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        x, y = data[0].to(device), data[1].to(device)
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        train_loss += loss.item()
        train_pred.extend(out.argmax(1).tolist())
        train_true.extend(y.argmax(1).tolist())

        if batch_idx % log_interval == 0 and batch_idx > 0:
            cur_loss = train_loss / log_interval
            # scheduler.step(cur_loss)
            cur_f1 = f1_score(train_true, train_pred)
            time_cost = time.time()-start_time

            with open(f'result_{args.num_samples}/train_{suffix}.txt', 'a', encoding='utf-8') as f:
                f.write(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | '
                        f'loss {cur_loss:2.5f} |'
                        f'f1 {cur_f1:.5f} |'
                        f'time {time_cost:4.2f}\n')
            print(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | '
                  f'loss {cur_loss} |'
                  f'f1 {cur_f1}')

            loss_all.append(train_loss)
            f1_all.append(cur_f1)

            start_time = time.time()
            train_loss = 0
            train_acc = 0

    train_loss = sum(loss_all) / len(train_loader)
    print("epoch : {}/{}, loss = {:.6f}".format(epoch, epochs, train_loss))

    model.eval()
    n = 0.0
    acc = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            x, y = data[0].to(device), data[1].to(device)
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            out = model(x).cpu()
            if batch_idx == 0:
                y_pred = out
                y_true = y.cpu()
            else:
                y_pred = np.concatenate((y_pred, out), axis=0)
                y_true = np.concatenate((y_true, y.cpu()), axis=0)

    # calculate metrics
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    report = precision_recall_fscore_support(y_true, y_pred, average='binary')
    with open(f'result_{args.num_samples}/train_{suffix}.txt', 'a', encoding='utf-8') as f:
        f.write('number of epochs:'+str(epoch)+'\n')
        f.write('Number of testing data:'+str(x_test.shape[0])+'\n')
        f.write('Precision:'+str(report[0])+'\n')
        f.write('Recall:'+str(report[1])+'\n')
        f.write('F1 score:'+str(report[2])+'\n')
        f.write('all_loss:'+str(loss_all)+'\n')
        f.write('\n')
        f.close()

    print(f'Number of testing data: {x_test.shape[0]}')
    print(f'Precision: {report[0]:.4f}')
    print(f'Recall: {report[1]:.4f}')
    print(f'F1 score: {report[2]:.4f}')