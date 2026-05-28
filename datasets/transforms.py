import torchvision.transforms as transforms

def build_transform(img_size):
    """
    构建图像预处理的 transform
    将输入图像展平为长度为L的序列, 特征维度为d_input=1 (单通道灰度图)
    """
    
    transform = transforms.Compose([
        # 强制转为单通道灰度图
        transforms.Grayscale(num_output_channels=1),
        # 转为 Tensor, 范围 [0, 1]
        transforms.ToTensor(),
        # 归一化到 [-1, 1]
        transforms.Normalize((0.5,),(0.5,)),
        # 展平为 (L, d_input) 其中 L = img_size * img_size, d_input = 1
        transforms.Lambda(lambda x: x.view(1,img_size * img_size).t())
    ])
    return transform