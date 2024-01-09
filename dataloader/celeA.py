import csv
import os
from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union

import PIL
import torch

from torchvision.datasets.utils import check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


CSV = namedtuple("CSV", ["header", "index", "data"])


class CelebA(VisionDataset):
    """
    We construct a training set that r percent of the samples with [label] are also [shortcut]
    Args:
        r: ratio of positive samples have shortcut
    """

    base_folder = "celeba"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        short_cut_prob = 0.9,
        label_index = 31, # Smiling
        shortcut_index = 9 # Blond_Hair
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        identity = self._load_csv("identity_CelebA.txt")
        bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        # if mask == slice(None):  # if split == "all"
        #     self.filename = splits.index
        # else:
        #     self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        if split == 'train':
            """
            In training set, we ensure [short_cut_prob] percent of samples with [label] also have [shortcut],
            also, we ensure all negative samples have no [shortcut]
            """
            locas = torch.squeeze(torch.nonzero(mask))
            label_set = set((attr.data[mask][:, label_index]==1).nonzero().squeeze_().tolist())
            shortcut_set = set((attr.data[mask][:, shortcut_index]==1).nonzero().squeeze_().tolist())
            len_both = len(label_set.intersection(shortcut_set))
            len_pos = int(len_both / short_cut_prob)
            len_more = len_pos - len_both
            complement = label_set - shortcut_set # positive samples with no shortcut
            len_throw = len(torch.tensor(list(complement))) - len_more
            set_throw = set(torch.tensor(list(complement))[torch.randperm(len(complement))][:len_throw].tolist())

            negative_set = set((attr.data[mask][:, label_index]!=1).nonzero().squeeze_().tolist())
            neg_short_set = negative_set.intersection(shortcut_set)

            filtered = torch.tensor(list(set(locas.tolist()) - set_throw - neg_short_set))



            self.filename = [splits.index[i] for i in filtered]
            self.attr = attr.data[filtered]
        elif split == 'test':
            """
            In test set, we ensure all samples with [label] do not have [shortcut],
            also, we ensure all negative samples have no [shortcut]
            """
            locas = torch.squeeze(torch.nonzero(mask))
            offset = (splits.data == 2).squeeze().nonzero().squeeze()[0].item()
            label_set = set(((attr.data[mask][:, label_index]==1).nonzero().squeeze_() + offset).tolist())
            shortcut_set = set(((attr.data[mask][:, shortcut_index]==1).nonzero().squeeze_() + offset).tolist())

            negative_set = set(((attr.data[mask][:, label_index]!=1).nonzero().squeeze_()+ offset).tolist())
            neg_short_set = negative_set.intersection(shortcut_set)

            filtered = torch.tensor(list(set(locas.tolist()) - (label_set.intersection(shortcut_set)) - neg_short_set))

            self.filename = [splits.index[i] for i in filtered]
            self.attr = attr.data[filtered]
        elif split == 'all':
            self.filename = splits.index
            self.attr = attr.data[mask]
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
            self.attr = attr.data[mask]

        
        self.identity = identity.data[mask]
        self.bbox = bbox.data[mask]
        self.landmarks_align = landmarks_align.data[mask]
        
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header
        self.short_cut_prob = short_cut_prob

    def _load_csv(
        self,
         filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        extract_archive(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None
        
        # return X, (target[0].to(torch.float32), target[1].to(torch.float32))
        return X, target

    def __len__(self) -> int:
        return len(self.filename)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)
