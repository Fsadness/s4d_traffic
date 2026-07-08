import torchvision.transforms as transforms

from datasets.tokenizer import *

def build_transform(args):
    """
    构建图像预处理的 transform
    将输入图像展平为长度为L的序列, 特征维度为d_input=1 (单通道灰度图)
    """
    
    transformList = [
        # 强制转为单通道灰度图
        transforms.Grayscale(num_output_channels=1),
        # 转为 Tensor, 范围 [0, 1]
        transforms.ToTensor(),
        # 归一化到 [-1, 1]
        transforms.Normalize((0.5,),(0.5,)),
    ]

    if args.tokenizer == 'pixel':
        transformList.append(PixelTokenizer())
    elif args.tokenizer == 'patch':
        transformList.append(PatchTokenizer(args.patch_size))
    elif args.tokenizer == 'compressed_patch':
        transformList.append(CompressedPatchTokenizer(args.patch_size, args.compressed_size))
    elif args.tokenizer == 'packet':
        transformList.append(PacketTokenizer())
    elif args.tokenizer == 'log_compressed_packet':
        transformList.append(LogCompressedPacketTokenizer(segment=args.log_compressed_segment))
    else:
        raise ValueError(f"Unsupported tokenizer type: {args.tokenizer}")
    
    return transforms.Compose(transformList)