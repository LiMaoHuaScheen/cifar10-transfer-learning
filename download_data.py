"""自动下载 CIFAR-10 数据集到 data/ 目录。

默认走 torchvision 的官方镜像；若网络较慢，可改用下方 MIRROR 指向
国内镜像（腾讯云 COS 上的公开数据集镜像）。
"""
import os
import urllib.request

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# 若 torchvision 默认源访问缓慢，可改为国内镜像：
MIRROR = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    target = os.path.join(DATA_ROOT, "cifar-10-python.tar.gz")

    if not os.path.exists(target):
        print("Downloading CIFAR-10 from", MIRROR)
        urllib.request.urlretrieve(MIRROR, target)
        print("Saved to", target)
    else:
        print("Archive already exists:", target)

    # 触发 torchvision 的自动解压（解压到 data/cifar-10-batches-py/）
    from torchvision import datasets
    datasets.CIFAR10(root=DATA_ROOT, train=True, download=False)
    print("Ready. Training set:", os.path.join(DATA_ROOT, "cifar-10-batches-py"))


if __name__ == "__main__":
    main()
