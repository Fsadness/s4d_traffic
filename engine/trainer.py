"""训练循环：所有训练入口的唯一实现。

历史版本 `train.py` / `train_log.py` / `train_log_patch.py` 各自复制了一份训练循环，
这里把三者差异收敛为参数：

- 日志与早停：始终写 `train.log`；是否提前停止由 `args.use_early_stop` 决定
- tokenizer：由 `datasets.specs.InputSpec` 决定，训练循环不关心具体方案
- 预训练：`args.mode == "pretrain"` 时使用掩码重建 loss

训练循环保持与 `train_log_patch.py` 完全一致的行为与日志格式。
"""

import os

import torch
from tqdm.auto import tqdm

from engine.checkpoint import build_state, save_checkpoint


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, args, exp_dir, logger):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.args = args
        self.exp_dir = exp_dir
        self.logger = logger

        self.best_acc = 0.0
        self.best_loss = float("inf")
        self.early_stop_counter = 0
        self.start_epoch = 0

        self.latest_ckpt = os.path.join(exp_dir, "latest.pth")
        self.best_ckpt = os.path.join(exp_dir, "best.pth")
        self.checkpoint_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def log(self, message):
        self.logger.info(message)

    # ------------------------------------------------------------------
    # 单个 epoch 的训练
    # ------------------------------------------------------------------
    def train_epoch(self, epoch, trainloader):
        self.model.train()
        train_loss = 0
        acc = 0
        correct = 0
        total = 0

        pbar = tqdm(enumerate(trainloader), total=len(trainloader))

        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # ------------------------------ 预训练 ------------------------------
            if self.args.mode == "pretrain":
                mask_ratio = 0.5
                batch, length, _ = inputs.shape
                mask = torch.rand(batch, length, device=self.device) < mask_ratio
                mask = mask.unsqueeze(-1)  # (B, L, 1)

                masked_inputs = inputs * (~mask)
                outputs = self.model(masked_inputs)  # (B, L, 1)
                loss = self.criterion(outputs[mask], inputs[mask])

                train_loss += loss.item()
                desc = f"Loss: {train_loss / (batch_idx + 1):.4f}"

            # ------------------------------ 微调 --------------------------------
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                train_loss += loss.item()
                acc = 100.0 * correct / total
                desc = (
                    f"Loss: {train_loss / (batch_idx + 1):.4f} | "
                    f"Acc: {acc:.3f}% "
                    f"({correct}/{total})"
                )

            loss.backward()
            self.optimizer.step()

            pbar.set_description(f"Batch ({batch_idx}/{len(trainloader)}) | {desc}")

        lr = self.scheduler.get_last_lr()[0]
        if self.args.mode == "pretrain":
            self.log(
                f"| Epoch: {epoch} | lr: {lr:.6f} | "
                f"Train Loss: {train_loss / len(trainloader):.4f} |"
            )
        else:
            self.log(
                f"| Epoch: {epoch} | lr: {lr:.6f} | "
                f"Train Loss: {train_loss / len(trainloader):.4f} | "
                f"Train Acc: {acc:.3f}% ({correct}/{total}) |"
            )

        return train_loss / len(trainloader)

    # ------------------------------------------------------------------
    # 评估（验证集 checkpoint=True，测试集 checkpoint=False）
    # ------------------------------------------------------------------
    def evaluate(self, dataloader, epoch, checkpoint=False):
        self.model.eval()
        eval_loss = 0
        acc = 0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))

            for batch_idx, (inputs, targets) in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                eval_loss += loss.item()
                acc = 100.0 * correct / total

                pbar.set_description(
                    f"Batch ({batch_idx}/{len(dataloader)}) | "
                    f"Loss: {eval_loss / (batch_idx + 1):.4f} | "
                    f"Acc: {acc:.3f}% ({correct}/{total})"
                )

        lr = self.scheduler.get_last_lr()[0]
        if checkpoint:
            self.log(
                f"| Epoch: {epoch} | lr: {lr:.6f} | "
                f"Val Loss: {eval_loss / len(dataloader):.4f} | "
                f"Val Acc: {acc:.3f}% ({correct}/{total}) |"
            )
        else:
            self.log(
                f"| Epoch: {epoch} | lr: {lr:.6f} | "
                f"Test Loss: {eval_loss / len(dataloader):.4f} | "
                f"Test Acc: {acc:.3f}% ({correct}/{total}) |"
            )

        acc = 100.0 * correct / total

        # ---------------------------- 保存最优 ----------------------------
        if checkpoint:
            if self.args.use_early_stop:
                improved = acc > self.best_acc + self.args.early_stop_delta
            else:
                improved = acc > self.best_acc

            if improved:
                print(f"==> Saving checkpoint... Epoch: {epoch} | Acc: {acc:.3f}%")
                self.log(f"Current best: Epoch: {epoch} | Acc: {acc:.3f}%")

                state = build_state(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    acc=acc,
                    early_stop_counter=self.early_stop_counter,
                    epoch=epoch,
                )
                save_checkpoint(state, self.best_ckpt)

                self.best_acc = acc
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

        return acc

    # ------------------------------------------------------------------
    # 完整训练
    # ------------------------------------------------------------------
    def fit(self, trainloader, valloader=None, testloader=None):
        pbar = tqdm(range(self.start_epoch, self.args.epochs))

        for epoch in pbar:
            # ------------------------------ 预训练 ------------------------------
            if self.args.mode == "pretrain":
                pbar.set_description("Epoch: %d" % (epoch))
                train_loss = self.train_epoch(epoch, trainloader)
                self.scheduler.step()

                print(
                    f"Epoch {epoch} learning rate: {self.scheduler.get_last_lr()[0]:.6f} | "
                    f"Train Loss: {train_loss:.4f}"
                )

                state = {"model": self.model.state_dict(), "epoch": epoch}
                save_checkpoint(state, self.latest_ckpt)

                if train_loss < self.best_loss:
                    print(f"==> Saving checkpoint... Epoch: {epoch} | Loss: {train_loss:.4f}")
                    self.log(f"Current best: Epoch: {epoch} | Loss: {train_loss:.4f}")
                    save_checkpoint(state, self.best_ckpt)
                    self.best_loss = train_loss

                if epoch > 0 and epoch % self.args.save_freq == 0:
                    backup = os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pth")
                    save_checkpoint(state, backup)

            # ------------------------------ 微调 --------------------------------
            else:
                if epoch == self.start_epoch:
                    pbar.set_description(f"Epoch: {epoch}")
                else:
                    pbar.set_description(f"Epoch: {epoch} | Val acc: {val_acc:.3f}")

                train_loss = self.train_epoch(epoch, trainloader)
                val_acc = self.evaluate(valloader, epoch, checkpoint=True)
                test_acc = self.evaluate(testloader, epoch, checkpoint=False)
                self.scheduler.step()

                state = build_state(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    acc=val_acc,
                    early_stop_counter=self.early_stop_counter,
                    epoch=epoch,
                )
                save_checkpoint(state, self.latest_ckpt)

                if epoch > 0 and epoch % self.args.save_freq == 0:
                    backup = os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pth")
                    save_checkpoint(state, backup)

                print(
                    f"Epoch {epoch} learning rate: {self.scheduler.get_last_lr()[0]:.6f} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Acc: {val_acc:.3f}% | "
                    f"Test Acc: {test_acc:.3f}%"
                )
                self.log(
                    f"| Epoch: {epoch} | lr: {self.scheduler.get_last_lr()[0]:.6f} | "
                    f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.3f}% | "
                    f"Test Acc: {test_acc:.3f}% |"
                )

            # ------------------------------ 早停 --------------------------------
            if (
                self.args.mode == "finetune"
                and self.args.use_early_stop
                and self.early_stop_counter >= self.args.early_stop_patience
            ):
                print(
                    f"Early stop at epoch {epoch} | "
                    f"Val Acc: {val_acc:.3f}% | Test Acc: {test_acc:.3f}%"
                )
                self.log(
                    f"Early stop at epoch {epoch} | "
                    f"Val Acc: {val_acc:.3f}% | Test Acc: {test_acc:.3f}%"
                )
                break

        return {"best_acc": self.best_acc, "best_loss": self.best_loss}
