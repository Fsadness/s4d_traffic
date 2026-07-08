import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import yaml
import logging
import argparse
import importlib
from datetime import datetime
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
parser.add_argument('--model', default='s4d_classification', type=str, help='Model name (should match a file in models/networks)')
parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# Patch
parser.add_argument('--tokenizer', type=str, default="pixel", choices=["pixel", "patch", "compressed_patch", "packet", "log_compressed_packet"], help='Tokenizer')
parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
parser.add_argument('--compressed_size', type=int, default=2, help='Compressed patch size')
# parser.add_argument('--packet_count', type=int, default=5, help='Packet count')
# parser.add_argument('--packet_size', type=int, default=320, help='Packet size')
parser.add_argument('--log_compressed_segment', type=int, default=64, help='Log-compressed segment size')
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
# parser.add_argument('--save_path', default='./checkpoint/ckpt.pth', type=str, help='Path to save checkpoints')
parser.add_argument('--save_dir', default='./checkpoint', type=str, help='Directory to save checkpoints')
parser.add_argument('--save_freq', default=10, type=int, help='Save checkpoint every N epochs')
# Early stopp
parser.add_argument('--use_early_stop', default=False, action='store_true', help='Enable early stopping')
parser.add_argument('--early_stop_patience', default=10, type=int, help='Early stopping patience')
parser.add_argument('--early_stop_delta', default=0.001, type=float, help='Early stopping delta')
# Config
parser.add_argument('--config', '-c', default=None, type=str, help='Config file path')
args = parser.parse_args()

###############################################################################
# Save directory setup
###############################################################################

# load from resume path
if args.resume:
    resume_pth = os.path.abspath(args.resume)
    resume_dir = os.path.dirname(resume_pth)
    # save_dir/model/timestamp/checkpoints/xxx.pth
    if os.path.basename(resume_dir) == "checkpoints":
        exp_dir = os.path.dirname(resume_dir)
    else:
        exp_dir = resume_dir
    # load config from exp_dir/config.yaml
    args.config = os.path.join(exp_dir, 'config.yaml')
else:
    exp_dir = None

################################################################################
# Config Loading
################################################################################

if args.config is not None:
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override args with config values
    for key, value in config.items():
        if hasattr(args, key):
            # override if args is default, otherwise keep command line args
            if getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
            # else do not override command line args
        else:
            print(f'Warning: Unknown config key "{key}"')


###############################################################################
# Save config and logs in exp_dir
###############################################################################

# create save_dir/model/timestamp/
if exp_dir is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.save_dir, args.model, timestamp)
    os.makedirs(exp_dir, exist_ok=True)

config_file = os.path.join(exp_dir, 'config.yaml')
log_file = os.path.join(exp_dir, 'train.log')

latest_ckpt = os.path.join(exp_dir, 'latest.pth')
best_ckpt = os.path.join(exp_dir, 'best.pth')

checkpoint_dir = os.path.join(exp_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Save config
if not args.resume:
    with open(config_file, 'w') as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

# Init logger
logger = logging.getLogger("experiment")
logger.setLevel(logging.INFO)
logger.propagate = False

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(message)s"
    )
)

logger.addHandler(file_handler)

def log(msg):
    logger.info(msg)

if args.resume:
    log(f"Resuming from checkpoint: {args.resume}")
else:
    log("=" * 80)
    log(f"Experiment: {exp_dir}")

    for k, v in vars(args).items():
        log(f"{k}: {v}")

    log("=" * 80)


###############################################################################
# Device
###############################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0 # best test accuracy
best_loss = float('inf') # best pretrain loss
early_stop_counter = 0 # counter for early stopping
start_epoch = 0 # start from epoch 0 or last checkpoint epoch


###############################################################################
# Dataloader
###############################################################################

print(f'==> Preparing {args.dataset} data..')

trainloader, valloader, testloader, num_classes = (
    build_dataloader(args)
)
print(f"Data shape: {trainloader.dataset[0][0].shape}")

d_input = 1
if args.tokenizer == "pixel":
    d_input = 1
elif args.tokenizer == "patch":
    d_input = args.patch_size * args.patch_size
elif args.tokenizer == "compressed_patch":
    d_input = args.compressed_size * args.compressed_size
elif args.tokenizer == "packet" or args.tokenizer == "log_compressed_packet":
    d_input = 5 # packet_count is fixed to 5 for now
else:
    raise ValueError(f"Unsupported tokenizer type: {args.tokenizer}")



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
# Optimizer
###############################################################################

optimizer, scheduler = setup_optimizer(
    model=model,
    lr=args.lr,
    weight_decay=args.weight_decay,
    epochs=args.epochs,
)

###############################################################################
# Resume / Pretrained Loading
###############################################################################

if args.mode == 'finetune':
    if args.resume:
        # resume from specified checkpoint path
        print('==> Resuming from checkpoint..')
        assert os.path.exists(args.resume), 'Checkpoint file not found'
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc = checkpoint['acc']
        early_stop_counter = checkpoint['early_stop_counter']
        start_epoch = checkpoint['epoch'] + 1
        print(f'==> Resumed from checkpoint: {args.resume} | Epoch: {checkpoint["epoch"]}')

    else:
        if os.path.exists(args.pretrained_path):
            # load pretrained weights
            print(f'==> Loading pretrained weights: {args.pretrained_path}')
            checkpoint = torch.load(args.pretrained_path, map_location=device)
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
# Everything after this point is standard PyTorch training!
###############################################################################

def train():
    model.train()
    train_loss = 0
    acc = 0
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
            # loss = criterion(outputs, inputs)


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

    if args.mode == 'pretrain':
        log(f'| Epoch: {epoch} | lr: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss/len(trainloader):.4f} |')
    else:
        log(
            f'| Epoch: {epoch} | lr: {scheduler.get_last_lr()[0]:.6f} | '
            f'Train Loss: {train_loss/len(trainloader):.4f} | Train Acc: {acc:.3f}% ({correct}/{total}) |'
        )

    return train_loss / len(trainloader)


###############################################################################
# Evaluation
###############################################################################

def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    global early_stop_counter
    model.eval()
    eval_loss = 0
    acc = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            eval_loss += loss.item()
            acc = 100. * correct / total

            pbar.set_description(
                f'Batch ({batch_idx}/{len(dataloader)}) | '
                f'Loss: {eval_loss/(batch_idx+1):.4f} | '
                f'Acc: {acc:.3f}% ({correct}/{total})'
            )

    if checkpoint:
        log(
            f'| Epoch: {epoch} | lr: {scheduler.get_last_lr()[0]:.6f} | '
            f'Val Loss: {eval_loss/len(dataloader):.4f} | Val Acc: {acc:.3f}% ({correct}/{total}) |'
        )
    else:
        log(
            f'| Epoch: {epoch} | lr: {scheduler.get_last_lr()[0]:.6f} | '
            f'Test Loss: {eval_loss/len(dataloader):.4f} | Test Acc: {acc:.3f}% ({correct}/{total}) |'
        )

    ###########################################################################
    # Save Checkpoint
    ###########################################################################
    
    acc = 100. * correct / total
    if args.use_early_stop:
        improved = acc > best_acc + args.early_stop_delta
    else:
        improved = acc > best_acc
    if checkpoint:
        if improved:
            print(
                f'==> Saving checkpoint... '
                f'Epoch: {epoch} | Acc: {acc:.3f}%'
            )
            log(f'Current best: Epoch: {epoch} | Acc: {acc:.3f}%')

            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc': acc,
                'early_stop_counter': early_stop_counter,
                'epoch': epoch,
            }

            torch.save(state, best_ckpt)

            best_acc = acc
            early_stop_counter = 0
        else:
            early_stop_counter += 1

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
        torch.save(state, latest_ckpt)

        # save best pretrain model
        if train_loss < best_loss:
            print(
                f'==> Saving checkpoint... '
                f'Epoch: {epoch} | Loss: {train_loss:.4f}'
            )
            log(f'Current best: Epoch: {epoch} | Loss: {train_loss:.4f}')

            torch.save(state, best_ckpt)
            best_loss = train_loss

        # epoch backup
        if epoch > 0 and epoch % args.save_freq == 0:
            backup_ckpt = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
            torch.save(state, backup_ckpt)

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

        # Save latest checkpoint every epoch
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'acc': val_acc,
            'early_stop_counter': early_stop_counter,
            'epoch': epoch,
        }
        torch.save(state, latest_ckpt)

        # epoch backup
        if epoch > 0 and epoch % args.save_freq == 0:
            backup_ckpt = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
            torch.save(state, backup_ckpt)

        print(
            f'Epoch {epoch} learning rate: {scheduler.get_last_lr()[0]:.6f} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Acc: {val_acc:.3f}% | '
            f'Test Acc: {test_acc:.3f}%'
        )
        log(
            f'| Epoch: {epoch} | lr: {scheduler.get_last_lr()[0]:.6f} | '
            f'Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f}% | Test Acc: {test_acc:.3f}% |'
        )
    
    # early stop check
    if args.mode == 'finetune' and args.use_early_stop and early_stop_counter >= args.early_stop_patience:
        print(f'Early stop at epoch {epoch} | Val Acc: {val_acc:.3f}% | Test Acc: {test_acc:.3f}%')
        log(f'Early stop at epoch {epoch} | Val Acc: {val_acc:.3f}% | Test Acc: {test_acc:.3f}%')
        break