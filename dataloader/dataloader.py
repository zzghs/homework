import os
import struct
import zlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SVHN_MEAN = [0.4377, 0.4438, 0.4728]
SVHN_STD = [0.1980, 0.2010, 0.1970]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13
MI_MATRIX = 14
MI_COMPRESSED = 15

_DTYPE_MAP = {
    MI_INT8: np.int8,
    MI_UINT8: np.uint8,
    MI_INT16: np.int16,
    MI_UINT16: np.uint16,
    MI_INT32: np.int32,
    MI_UINT32: np.uint32,
    MI_SINGLE: np.float32,
    MI_DOUBLE: np.float64,
    MI_INT64: np.int64,
    MI_UINT64: np.uint64,
}


def _pad_to_8(size):
    return (size + 7) & ~7


def _read_tag(buffer, offset):
    first, second = struct.unpack_from("<II", buffer, offset)
    data_type = first & 0xFFFF
    num_bytes = (first >> 16) & 0xFFFF

    if num_bytes != 0:
        payload_offset = offset + 4
        next_offset = offset + 8
        return data_type, num_bytes, payload_offset, next_offset

    payload_offset = offset + 8
    next_offset = payload_offset + _pad_to_8(second)
    return first, second, payload_offset, next_offset


def _parse_numeric_array(payload, data_type, dims):
    if data_type not in _DTYPE_MAP:
        raise ValueError(f"Unsupported MATLAB data type: {data_type}")
    array = np.frombuffer(payload, dtype=_DTYPE_MAP[data_type])
    return array.reshape(tuple(dims), order="F")


def _parse_matrix(buffer):
    offset = 0

    _, _, _, offset = _read_tag(buffer, offset)

    _, dims_size, dims_payload_offset, next_offset = _read_tag(buffer, offset)
    dims = np.frombuffer(
        buffer[dims_payload_offset : dims_payload_offset + dims_size],
        dtype=np.int32,
    )
    offset = next_offset

    name_type, name_size, name_payload_offset, next_offset = _read_tag(buffer, offset)
    if name_type not in (MI_INT8, MI_UINT8):
        raise ValueError("Unsupported MATLAB variable name encoding")
    name = buffer[name_payload_offset : name_payload_offset + name_size].decode("ascii")
    offset = next_offset

    real_type, real_size, real_payload_offset, next_offset = _read_tag(buffer, offset)
    real_payload = buffer[real_payload_offset : real_payload_offset + real_size]
    data = _parse_numeric_array(real_payload, real_type, dims)
    return name, data


def load_svhn_mat(mat_path):
    variables = {}
    with open(mat_path, "rb") as file:
        buffer = file.read()

    offset = 128
    while offset < len(buffer):
        data_type, num_bytes, payload_offset, next_offset = _read_tag(buffer, offset)
        payload = buffer[payload_offset : payload_offset + num_bytes]

        if data_type == MI_COMPRESSED:
            inner = zlib.decompress(payload)
            inner_type, inner_size, inner_payload_offset, _ = _read_tag(inner, 0)
            if inner_type != MI_MATRIX:
                raise ValueError("Compressed element is not an miMATRIX block")
            name, data = _parse_matrix(inner[inner_payload_offset : inner_payload_offset + inner_size])
            variables[name] = data
            offset = payload_offset + num_bytes
        elif data_type == MI_MATRIX:
            name, data = _parse_matrix(payload)
            variables[name] = data
            offset = next_offset
        else:
            offset = next_offset

    if "X" not in variables or "y" not in variables:
        raise ValueError(f"Missing X or y in MATLAB file: {mat_path}")

    images = variables["X"]
    labels = variables["y"].reshape(-1)
    labels = np.where(labels == 10, 0, labels).astype(np.int64)
    return images, labels


def _build_transforms(image_size=224, train=True, pretrained=False):
    mean = IMAGENET_MEAN if pretrained else SVHN_MEAN
    std = IMAGENET_STD if pretrained else SVHN_STD

    transform_list = [transforms.Resize((image_size, image_size))]
    if train:
        transform_list.extend(
            [
                transforms.RandomCrop(image_size, padding=12),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(transform_list)


class SVHNMatDataset(Dataset):
    def __init__(self, mat_path, image_size=224, train=True, pretrained=False):
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"MAT file not found: {mat_path}")

        self.images, self.labels = load_svhn_mat(mat_path)
        self.transform = _build_transforms(image_size=image_size, train=train, pretrained=pretrained)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[:, :, :, index]
        label = int(self.labels[index])
        image = Image.fromarray(np.transpose(image, (1, 0, 2)))
        image = self.transform(image)
        return image, label


def create_dataloader(config, mode="train"):
    batch_size = config["batch_size"] if mode == "train" else config.get("test_batch_size", config["batch_size"])
    shuffle = config.get("shuffle_train", True) if mode == "train" else False
    num_workers = config.get("num_workers", 0)
    pin_memory = config.get("pin_memory", torch.cuda.is_available())
    drop_last = config.get("drop_last", False) if mode == "train" else False
    data_root = config.get("data_root", "datasets")
    image_size = config.get("image_size", 224)
    pretrained = config.get("pretrained", False)

    filename = "train_32x32.mat" if mode == "train" else "test_32x32.mat"
    dataset = SVHNMatDataset(
        mat_path=os.path.join(data_root, filename),
        image_size=image_size,
        train=(mode == "train"),
        pretrained=pretrained,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    print(f"{mode} DataLoader: {len(dataset)} samples, batch_size={batch_size}")
    return loader
