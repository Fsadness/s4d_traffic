import torchvision.transforms as transforms

from datasets.specs import make_input_spec


def build_transform(args):
    """
    构建图像预处理的 transform
    将输入图像转为单通道灰度图并归一化, 再按 `InputSpec` 展平成序列 (L, d_input)
    """
    transform_list = [
        # 强制转为单通道灰度图
        transforms.Grayscale(num_output_channels=1),
        # 转为 Tensor, 范围 [0, 1]
        transforms.ToTensor(),
        # 归一化到 [-1, 1]
        transforms.Normalize((0.5,), (0.5,)),
        # tokenizer 由 datasets/specs.py 统一注册与构造
        make_input_spec(args).build(),
    ]

    return transforms.Compose(transform_list)
