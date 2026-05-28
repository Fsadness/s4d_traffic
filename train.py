import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import importlib
from tqdm.auto import tqdm
from torchvision.datasets import ImageFolder

from models.layers.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.layers.s4d import S4D
from datasets.dataloader import build_dataloader
from optim.optimizer import setup_optimizer


###############################################################################
# Argument Parser
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch Traffic Training')
# Dataset
parser.add_argument('--data_dir', default='/mnt/e/datasets/NetMamba', type=str, help='数据集根目录')
parser.add_argument('--dataset', default='ISCXVPN2016', type=str, help='Dataset')
parser.add_argument('--img_size', default=40, type=int, help='img_size (40x40)')
# Dataloader
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
# Model
parser.add_argument('--model', default='s4d_classification', type=str,help='Model name (should match a file in models/networks)')
parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# Training
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
# Optimizer
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
# Mode
parser.add_argument('--mode', default='finetune', choices=['pretrain', 'finetune'], help='Model training mode')
# Checkpoint
# parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--resume', '-r', type=str, default=None, help='Resume from checkpoint path')
# parser.add_argument('--checkpoint_path', default='./checkpoint/ckpt.pth', type=str, help='Checkpoint path')
parser.add_argument('--pretrained_path', default='', type=str, help='Pretrained model path')
parser.add_argument('--save_path', default='./checkpoint/ckpt.pth', type=str, help='Path to save checkpoints')

args = parser.parse_args()


###############################################################################
# Device
###############################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0 # best test accuracy
start_epoch = 0 # start from epoch 0 or last checkpoint epoch


###############################################################################
# Dataloader
###############################################################################

print(f'==> Preparing {args.dataset} data..')

trainloader, valloader, testloader, num_classes = (
    build_dataloader(args)
)

d_input = 1


###############################################################################
# Dynamic Model Import
###############################################################################

print(f'==> Building model: {args.model}')

model_module = importlib.import_module(
    f'models.networks.{args.model}'
)

ModelClass = model_module.Model


###############################################################################
# Build Model
###############################################################################

if args.mode == 'pretrain':
    model = ModelClass(
        d_input=d_input,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prenorm=args.prenorm,
        lr=args.lr,
    )
    # For pretraining, use MSE loss to reconstruct the input
    criterion = nn.MSELoss()
else:
    model = ModelClass(
        d_input=d_input,
        d_output=num_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prenorm=args.prenorm,
        lr=args.lr,
    )
    # For classification, use cross-entropy loss
    criterion = nn.CrossEntropyLoss()


model = model.to(device)

if device == 'cuda':
    cudnn.benchmark = True


###############################################################################
# Resume / Pretrained Loading
###############################################################################

if args.mode == 'finetune':
    if args.resume:
        # resume from specified checkpoint path
        print('==> Resuming from checkpoint..')
        assert os.path.exists(args.resume), 'Checkpoint file not found'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    else:
        if os.path.exists(args.pretrained_path):
            # load pretrained weights
            print(f'==> Loading pretrained weights: {args.pretrained_path}')
            checkpoint = torch.load(args.pretrained_path)
            pretrained_dict = checkpoint['model']
            model_dict = model.state_dict()

            # remove head/decoder
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if (
                    k in model_dict
                    and 'head' not in k
                    and 'decoder' not in k
                    and 'reconstruction_head' not in k
                )
            }

            # Update model weights
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print('==> Pretrained weights loaded')

        else:
            print('==> Training from scratch')
else: # pretraining mode, always start from scratch
    print('==> Pretraining from scratch')

###############################################################################
# Optimizer
###############################################################################

optimizer, scheduler = setup_optimizer(
    model=model,
    lr=args.lr,
    weight_decay=args.weight_decay,
    epochs=args.epochs,
)


###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))

    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        #######################################################################
        # Pretrain
        #######################################################################

        if args.mode == 'pretrain':
            # Create random mask
            mask_ratio = 0.5
            B, L, D = inputs.shape
            mask = torch.rand(B, L, device=device) < mask_ratio
            mask = mask.unsqueeze(-1) # (B, L, 1)

            # Mask the input
            masked_inputs = inputs * (~mask)

            # Forward pass
            outputs = model(masked_inputs) # (B, L, 1)

            # Compute loss only on masked positions
            loss = criterion(outputs[mask], inputs[mask])
            # Alternative: compute loss on all positions
            loss = criterion(outputs, inputs)

            train_loss += loss.item()
            desc = (
                f'Loss: {train_loss/(batch_idx+1):.4f}'
                # f'Avg Loss: {train_loss/(batch_idx+1):.4f} | '
                # f'Curr Loss: {loss.item():.4f}'
            )

        #######################################################################
        # Finetune
        #######################################################################

        else:

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            train_loss += loss.item()
            acc = 100. * correct / total
            desc = (
                f'Loss: {train_loss/(batch_idx+1):.4f} | '
                f'Acc: {acc:.3f}% '
                f'({correct}/{total})'
            )

        loss.backward()
        optimizer.step()

        pbar.set_description(
            f'Batch ({batch_idx}/{len(trainloader)}) | '
            f'{desc}'
        )

    return train_loss / len(trainloader)


###############################################################################
# Evaluation
###############################################################################

def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


            pbar.set_description(
                f'Batch ({batch_idx}/{len(dataloader)}) | '
                f'Loss: {eval_loss/(batch_idx+1):.4f} | '
                f'Acc: {100.*correct/total:.3f}% ({correct}/{total})'
            )


    ###########################################################################
    # Save Checkpoint
    ###########################################################################
    
    acc = 100. * correct / total
    if checkpoint:
        if acc > best_acc:
            print(
                f'==> Saving checkpoint... '
                f'Epoch: {epoch} | Acc: {acc:.3f}%'
            )

            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }

            dir = os.path.dirname(args.save_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(state, args.save_path)

            best_acc = acc

    return acc


###############################################################################
# Main Loop
###############################################################################

pbar = tqdm(range(start_epoch, args.epochs))

for epoch in pbar:

    ###########################################################################
    # Pretrain
    ###########################################################################

    if args.mode == 'pretrain':

        pbar.set_description('Epoch: %d' % (epoch))
        train_loss = train()
        scheduler.step()

        print(
            f'Epoch {epoch} learning rate: {scheduler.get_last_lr()[0]:.6f} | '
            f'Train Loss: {train_loss:.4f}'
        )

        # save latest pretrain model
        state = {
            'model': model.state_dict(),
            'epoch': epoch,
        }

        dir = os.path.dirname(args.save_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(state, args.save_path)

    ###########################################################################
    # Finetune
    ###########################################################################

    else:
        if epoch == start_epoch:
            pbar.set_description(f'Epoch: {epoch}')
        else:
            pbar.set_description(f'Epoch: {epoch} | Val acc: {val_acc:.3f}')

        train_loss = train()
        val_acc = eval(epoch, valloader, checkpoint=True)
        test_acc = eval(epoch, testloader, checkpoint=False)
        scheduler.step()

        print(
            f'Epoch {epoch} learning rate: {scheduler.get_last_lr()[0]:.6f} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Acc: {val_acc:.3f}% | '
            f'Test Acc: {test_acc:.3f}%'
        )