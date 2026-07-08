import os
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from datasets.transforms import build_transform


def _is_valid_image(path):
    """Check if file exists and is a valid image."""
    if not os.path.isfile(path):
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


class SafeImageFolder(ImageFolder):
    """ImageFolder that filters out missing/corrupt files to prevent DataLoader crashes."""
    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)
        # Filter out samples whose files are missing or corrupt
        original_count = len(self.samples)
        valid = [(p, c) for p, c in self.samples if _is_valid_image(p)]
        if len(valid) < original_count:
            print(f"[SafeImageFolder] {root}: filtered {original_count - len(valid)} invalid files, {len(valid)} remaining")
        self.samples = valid
        self.targets = [c for _, c in valid]
        self.imgs = self.samples


def build_dataloader(args):
    """
    基于args构建数据加载器,
    返回trainloader, valloader, testloader和num_classes
    """
    
    transform = build_transform(args)

    if args.mode == 'pretrain':
        # 预训练阶段只需要训练集, 不需要验证集和测试集
        data_dir = os.path.join(args.data_dir, args.dataset)
        train_dir = os.path.join(data_dir, 'train')
        trainset = ImageFolder(root=train_dir,transform=transform)

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        # num_classes设置为1
        return trainloader, None, None, 1
    
    else:
        data_dir = os.path.join(args.data_dir, args.dataset, 'dataset_sampled')

        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'valid')
        test_dir = os.path.join(data_dir, 'test')

        trainset = SafeImageFolder(root=train_dir, transform=transform)
        valset = SafeImageFolder(root=val_dir, transform=transform)
        testset = SafeImageFolder(root=test_dir, transform=transform)

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_classes = len(trainset.classes)
        return (trainloader, valloader, testloader, num_classes)