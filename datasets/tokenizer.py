import math
import torch
import torch.nn.functional as F

class PixelTokenizer:
    """
    将输入按像素展平
    """
    def __call__(self, x):
        H = x.shape[1]
        return x.view(1, H * H).t()

class PatchTokenizer:
    """
    将输入按patch_size展平
    """
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, x):
        P = self.patch_size
        C, H, W = x.shape
        patches = x.unfold(1, P, P).unfold(2, P, P)
        patches = patches.contiguous()
        patches = patches.view(C, -1, P * P)
        patches = patches.squeeze(0)

        return patches

class CompressedPatchTokenizer:
    """
    将输入按patch_size展平, 并压缩到compressed_size
    """
    def __init__(self, patch_size, compressed_size):
        self.patch_size = patch_size
        self.compressed_size = compressed_size

    def __call__(self, x):
        P = self.patch_size
        S = self.compressed_size

        C, H, W = x.shape
        patches = x.unfold(1, P, P).unfold(2, P, P)
        patches = patches.contiguous()
        patches = patches.view(C, -1, P * P)  # (1, num_patches, P*P)

        # 压缩特征维度: (batch, num_patches, P*P) -> (batch, num_patches, S*S)
        patches = F.adaptive_avg_pool1d(patches, S * S)
        patches = patches.squeeze(0)  # (num_patches, S*S)

        return patches

class PacketTokenizer:
    """
    将输入按packet的形式展平
    """
    def __init__(self, packet_count=5, packet_size=320):
        self.packet_count = packet_count
        self.packet_size = packet_size

    def __call__(self, x):
        C, H, W = x.shape
        packets = x.reshape(self.packet_count, self.packet_size)
        tokens = packets.t()

        return tokens

class LogCompressedPacketTokenizer:
    """
    将输入按patch_size展平, 将packet_size划分为多个segment, 并进行对数压缩
    """
    def __init__(self, packet_count=5, packet_size=320, segment=64):
        self.packet_count = packet_count
        self.packet_size = packet_size
        self.segment = segment

    def compress_segment(self, seg, target_len):
        # seg: (segment_len, packet_count)
        seg = seg.t().unsqueeze(0) # (1, packet_count, segment_len)
        seg = F.adaptive_avg_pool1d(seg, target_len) # (1, packet_count, target_len)
        return seg.squeeze(0).t()  # return (target_len, packet_count)

    def __call__(self, x):
        C, H, W = x.shape
        packets = x.reshape(self.packet_count, self.packet_size)
        tokens = packets.t()

        total_len = tokens.shape[0]
        outputs = []
        num_segments = math.ceil(total_len / self.segment)
        for i in range(num_segments):
            start = i * self.segment
            end = min(start + self.segment, total_len)
            seg = tokens[start:end]  # (segment_len, packet_count)
            target_len = max(1, int(round(seg.shape[0] / (i + 1))))  # 根据segment长度动态计算目标长度, 保证至少为1
            compressed_seg = self.compress_segment(seg, target_len)  # (target_len, packet_count)
            outputs.append(compressed_seg)
        # 将token划分为多个segment
        compressed_tokens = torch.cat(outputs, dim=0)  # (total_compressed_len, packet_count)

        return compressed_tokens
if __name__ == "__main__":

    import torch

    test_input = torch.arange(
        1600,
        dtype=torch.float32
    ).reshape(1, 40, 40)

    print("=" * 80)
    print("Input")
    print("=" * 80)

    print("输入形状:", test_input.shape)

    print("\n前8行:")
    print(test_input[0, :8, :8])

    print()

    # ==================================================
    # PixelTokenizer
    # ==================================================

    print("=" * 80)
    print("PixelTokenizer")
    print("=" * 80)

    pixel_tokenizer = PixelTokenizer()

    tokens = pixel_tokenizer(test_input)

    print("输出形状:", tokens.shape)

    print("\n前5行:")
    print(tokens[:5])

    print("\n后5行:")
    print(tokens[-5:])

    print("\n验证:")
    print("token[0] =", tokens[0])
    print("token[1] =", tokens[1])
    print("token[39] =", tokens[39])
    print("token[40] =", tokens[40])

    print()

    # ==================================================
    # PatchTokenizer
    # ==================================================

    print("=" * 80)
    print("PatchTokenizer(p=4)")
    print("=" * 80)

    patch_tokenizer = PatchTokenizer(
        patch_size=4
    )

    tokens = patch_tokenizer(test_input)

    print("输出形状:", tokens.shape)

    print("\n前3个Token:")
    print(tokens[:3])

    print("\n最后3个Token:")
    print(tokens[-3:])

    print()

    # ==================================================
    # CompressedPatchTokenizer
    # ==================================================

    print("=" * 80)
    print("CompressedPatchTokenizer")
    print("=" * 80)

    compressed_tokenizer = CompressedPatchTokenizer(
        patch_size=4,
        compressed_size=2,
    )

    tokens = compressed_tokenizer(test_input)

    print("输出形状:", tokens.shape)

    print("\n前3个Token:")
    print(tokens[:3])

    print("\n最后3个Token:")
    print(tokens[-3:])

    print()

    # ==================================================
    # PacketTokenizer
    # ==================================================

    print("=" * 80)
    print("PacketTokenizer")
    print("=" * 80)

    packet_tokenizer = PacketTokenizer(
        packet_count=5,
        packet_size=320,
    )

    tokens = packet_tokenizer(test_input)

    print("输出形状:", tokens.shape)

    print("\n前5行:")
    print(tokens[:5])

    print("\n后5行:")
    print(tokens[-5:])

    print("\n验证:")
    print("offset 0 =", tokens[0])
    print("offset 1 =", tokens[1])
    print("offset 319 =", tokens[319])

    print()

    # ==================================================
    # LogCompressedPacketTokenizer
    # ==================================================

    print("=" * 80)
    print("LogCompressedPacketTokenizer")
    print("=" * 80)

    log_tokenizer = LogCompressedPacketTokenizer(
        packet_count=5,
        packet_size=320,
        segment=64,
    )

    tokens = log_tokenizer(test_input)

    print("输出形状:", tokens.shape)

    print("\n前5行:")
    print(tokens[:5])

    print("\n后5行:")
    print(tokens[-5:])

    print()