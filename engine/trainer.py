import os
import torch
from tqdm.auto import tqdm

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        args,
    ):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.args = args
        self.best_acc = 0
    
    ###########################################################################
    # Train One Epoch
    ###########################################################################
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            ###################################################################
            # Pretrain
            ###################################################################
            
            # todo
            if self.args.mode == 'pretrain':
                loss = self.pretrain_step(
                    inputs
                )
                desc = (
                    f'Loss: '
                    f'{loss.item():.4f}'
                )
            ###################################################################
            # Finetune
            ###################################################################
            else:
                outputs = self.model(
                    inputs
                )
                loss = self.criterion(
                    outputs,
                    targets,
                )
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(
                    targets
                ).sum().item()
                acc = (
                    100. * correct / total
                )
                desc = (
                    f'Loss: {loss.item():.4f} | '
                    f'Acc: {acc:.2f}% '
                    f'({correct}/{total})'
                )
            ###################################################################
            # Backward
            ###################################################################
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_description(desc)
        avg_loss = (
            total_loss / len(dataloader)
        )
        return avg_loss
    ###########################################################################
    # Pretrain Step
    ###########################################################################
    def pretrain_step(self, inputs):
        mask_ratio = 0.5
        B, L, D = inputs.shape
        mask = (
            torch.rand(
                B,
                L,
                device=self.device,
            ) < mask_ratio
        )
        mask = mask.unsqueeze(-1)
        masked_inputs = inputs * (~mask)
        outputs = self.model(
            masked_inputs
        )
        loss = self.criterion(
            outputs,
            inputs,
        )
        return loss
    ###########################################################################
    # Evaluate
    ###########################################################################
    def evaluate(
        self,
        dataloader,
        save_checkpoint=False,
        epoch=None,
    ):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
            )
            for batch_idx, (inputs, targets) in pbar:
                inputs = inputs.to(
                    self.device
                )
                targets = targets.to(
                    self.device
                )
                outputs = self.model(
                    inputs
                )
                loss = self.criterion(
                    outputs,
                    targets,
                )
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(
                    targets
                ).sum().item()
                acc = (
                    100. * correct / total
                )
                pbar.set_description(
                    f'Loss: '
                    f'{total_loss/(batch_idx+1):.4f} | '
                    f'Acc: '
                    f'{acc:.2f}% '
                    f'({correct}/{total})'
                )
        avg_loss = (
            total_loss / len(dataloader)
        )
        acc = (
            100. * correct / total
        )
        #######################################################################
        # Save Checkpoint
        #######################################################################
        if save_checkpoint:
            if acc > self.best_acc:
                self.best_acc = acc
                self.save_checkpoint(
                    epoch,
                    acc,
                )
        return avg_loss, acc
    ###########################################################################
    # Save Checkpoint
    ###########################################################################
    def save_checkpoint(
        self,
        epoch,
        acc,
    ):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'acc': acc,
        }
        os.makedirs(
            './checkpoint',
            exist_ok=True,
        )
        torch.save(
            state,
            self.args.checkpoint_path,
        )
        print(
            f'==> Checkpoint saved '
            f'({acc:.2f}%)'
        )
    ###########################################################################
    # Save Pretrain Checkpoint
    ###########################################################################
    def save_pretrain_checkpoint(
        self,
        epoch,
    ):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        os.makedirs(
            './checkpoint',
            exist_ok=True,
        )
        torch.save(
            state,
            self.args.pretrained_path,
        )
    ###########################################################################
    # Fit
    ###########################################################################
    def fit(
        self,
        trainloader,
        valloader=None,
        testloader=None,
    ):
        for epoch in range(
            self.args.epochs
        ):
            print(
                f'\nEpoch: {epoch}'
            )
            ###################################################################
            # Train
            ###################################################################
            train_loss = self.train_epoch(
                trainloader
            )
            ###################################################################
            # Pretrain
            ###################################################################
            if self.args.mode == 'pretrain':
                print(
                    f'Train Loss: '
                    f'{train_loss:.4f}'
                )
                self.save_pretrain_checkpoint(
                    epoch
                )
            ###################################################################
            # Finetune
            ###################################################################
            else:
                val_loss, val_acc = self.evaluate(
                    valloader,
                    save_checkpoint=True,
                    epoch=epoch,
                )
                test_loss, test_acc = self.evaluate(
                    testloader
                )
                print(
                    f'Train Loss: '
                    f'{train_loss:.4f} | '
                    f'Val Acc: '
                    f'{val_acc:.2f}% | '
                    f'Test Acc: '
                    f'{test_acc:.2f}%'
                )
            ###################################################################
            # Scheduler
            ###################################################################
            self.scheduler.step()