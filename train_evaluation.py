import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_data, R2_score_calculate, IC_ICIR_score_calculate
from sklearn import metrics
from models import VGNN
import logging

logging.basicConfig(filename="output.log", level=logging.DEBUG)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=137, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--eta', type=float, default=1e-5, help='Weight decay (L1 loss on feature interactions).')
parser.add_argument('--lambda_im', type=float, default=2e-8, help='Weight decay (L1 loss on implicit link).')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=20, help='Patience.')
parser.add_argument('--gpu_ids', type=list, default=[1], help='Disables CUDA training.')
parser.add_argument('--accumulation_steps', type=int, default=16, help='Gradient Accumulation.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--hidden_RNN', type=int, default=32, help='Hidden size of RNN.')
parser.add_argument('--hidden_spillover', type=int, default=32, help='Hidden size of spillover embedding.')
parser.add_argument('--nclass', type=int, default=1, help='Number of class.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# Set seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# load data
# features shape: [Months, Firms, Dimension of features]
features, labels, adj_Ind, adj_Loc = load_data()

# Model and optimizer
model = VGNN(nfeat=features.shape[-1],
             nhid=args.hidden,
             hidden_RNN=args.hidden_RNN,
             hidden_spillover=args.hidden_spillover,
             nclass=args.nclass,
             dropout=args.dropout,
             alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
loss_fun = nn.MSELoss()

if (len(args.gpu_ids) == 0) or (not args.cuda):
    device = torch.device("cpu")
    print("Train Mode : CPU")
elif args.cuda and len(args.gpu_ids) > 1:
    # len(gpu_ids) > 1
    device = torch.device("cuda:0")
    model = nn.DataParallel(model, device_ids=args.gpu_ids)
    print("Train Mode : Multi GPU;", args.gpu_ids)
else:
    # len(gpu_ids) = 1
    device = torch.device("cuda:" + str(args.gpu_ids[0]) if args.cuda else "cpu")
    print("Train Mode : One GPU;", device)
print("\n", "##" * 10, "  NetWork  ", "##" * 10, "\n", model, "\n", "##" * 26, "\n")

model.to(torch.float)
model = model.to(device)
features = features.to(device)
labels = labels.to(device)
adj_Ind = adj_Ind.to(device)
adj_Loc = adj_Loc.to(device)

# Split data
rnn_length = 12
train_end_time = 12 * 7
val_end_time = 12 * 9
X_train, X_eval, X_test = features[:train_end_time], features[train_end_time - rnn_length + 1:val_end_time], features[val_end_time - rnn_length + 1:]
y_train, y_eval, y_test = labels[:train_end_time], labels[train_end_time - rnn_length + 1:val_end_time], labels[val_end_time - rnn_length + 1:]


def train(epoch):
    t = time.time()
    model.train()
    train_seq = list(range(len(X_train) + 1))[rnn_length:]
    random.shuffle(train_seq)
    total_loss = 0
    count_train = 0
    for i in train_seq:
        output, implicit_relation, regularization_R = model(adj_Ind, adj_Loc, X_train[i - rnn_length: i])
        # regression loss
        reg_loss = loss_fun(output, y_train[i - 1].reshape(-1, 1))
        # L1 loss on feature interactions and implicit link
        l1_loss_both = args.lambda_im * torch.sum(torch.abs(implicit_relation)) + args.eta * regularization_R / (X_train.size(1) * rnn_length)
        # total loss
        loss = reg_loss + l1_loss_both
        total_loss += loss.item()
        count_train += 1
        loss = loss / args.accumulation_steps
        loss.backward()
        if (count_train % args.accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
    if (count_train % args.accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    phase_pred_val = []
    phase_label_val = []
    eval_seq = list(range(len(X_eval) + 1))[rnn_length:]
    for i in eval_seq:
        with torch.no_grad():
            output, _, _ = model(adj_Ind, adj_Loc, X_eval[i - rnn_length: i])
        phase_pred_val.extend(output.detach().cpu().numpy().reshape(-1))
        phase_label_val.extend(y_eval[i - 1].detach().cpu().numpy())

    mse_val = metrics.mean_squared_error(np.array(phase_label_val), np.array(phase_pred_val))
    r2_val = R2_score_calculate(np.array(phase_label_val), np.array(phase_pred_val))
    rank_ic_val, rank_ic_ir_val = IC_ICIR_score_calculate(phase_label_val, phase_pred_val, len(eval_seq))

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(total_loss / count_train),
          'loss_val: {:.4f}'.format(mse_val),
          'R2_val: {:.4f}'.format(r2_val),
          'Rank_IC_val: {:.4f}'.format(rank_ic_val),
          'Rank_ICIR_val: {:.4f}'.format(rank_ic_ir_val),
          'time: {:.4f}s'.format(time.time() - t))

    return r2_val


def compute_test():
    model.eval()
    phase_pred_test = []
    phase_label_test = []
    test_seq = list(range(len(X_test) + 1))[rnn_length:]
    for i in test_seq:
        with torch.no_grad():
            output, _, _ = model(adj_Ind, adj_Loc, X_test[i - rnn_length: i])
        phase_pred_test.extend(output.detach().cpu().numpy().reshape(-1))
        phase_label_test.extend(y_test[i - 1].detach().cpu().numpy())

    mse_test = metrics.mean_squared_error(np.array(phase_label_test), np.array(phase_pred_test))
    r2_test = R2_score_calculate(np.array(phase_label_test), np.array(phase_pred_test))
    rank_ic_test, rank_ic_ir_test = IC_ICIR_score_calculate(phase_label_test, phase_pred_test, len(test_seq))
    print('Test results:',
          'loss_test: {:.4f}'.format(mse_test),
          'R2_test: {:.4f}'.format(r2_test),
          'Rank_IC_test: {:.4f}'.format(rank_ic_test),
          'Rank_ICIR_test: {:.4f}'.format(rank_ic_ir_test))

# Training model
t_total = time.time()
r2_values = []
bad_counter = 0
best = -100
best_epoch = 0
for epoch in range(args.epochs):
    r2_values.append(train(epoch))
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if r2_values[-1] > best:
        best = r2_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore the best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()