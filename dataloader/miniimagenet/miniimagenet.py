import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MiniImageNet(Dataset):

    def __init__(self, root='./data', train=True,
                 transform=None,
                 index_path=None, index=None, base_sess=None, is_clip=False):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.IMAGE_PATH = osp.join(self.root, "miniimagenet", "images")
        self.SPLIT_PATH = osp.join(self.root, "miniimagenet", "split")

        def find_csv(split_name):
            candidates = [
                osp.join(self.root, "miniimagenet", split_name + ".csv"),
                osp.join(self.SPLIT_PATH, split_name + ".csv"),
            ]
            path = next((candidate for candidate in candidates if osp.isfile(candidate)), None)
            if path is None:
                raise FileNotFoundError(
                    "MiniImageNet {}.csv not found. Checked: {}".format(
                        split_name, ", ".join(candidates)
                    )
                )
            return path

        train_csv = find_csv("train")
        test_csv = find_csv("test")
        if not osp.isdir(self.IMAGE_PATH):
            raise FileNotFoundError(
                "MiniImageNet image directory not found: {}".format(self.IMAGE_PATH)
            )

        def read_rows(csv_path):
            with open(csv_path, "r", encoding="utf-8") as handle:
                rows = [line.strip() for line in handle.readlines()][1:]
            parsed = []
            for line_number, line in enumerate(rows, start=2):
                if not line:
                    continue
                fields = line.split(",")
                if len(fields) != 2:
                    raise ValueError(
                        "Malformed MiniImageNet CSV row {} in {}: {!r}".format(
                            line_number, csv_path, line
                        )
                    )
                parsed.append((fields[0], fields[1]))
            return parsed

        train_rows = read_rows(train_csv)
        test_rows = read_rows(test_csv)

        # Use one global class-ID mapping for both splits. Training CSV order
        # defines the FSCIL class order; test-only classes are appended in a
        # deterministic order rather than being re-numbered by test CSV order.
        self.class_wnids = []
        for _, wnid in train_rows + test_rows:
            if wnid not in self.class_wnids:
                self.class_wnids.append(wnid)
        self.class_to_idx = {
            wnid: class_id for class_id, wnid in enumerate(self.class_wnids)
        }

        selected_rows = train_rows if train else test_rows
        self.data = []
        self.targets = []
        self.data2label = {}
        for name, wnid in selected_rows:
            if wnid not in self.class_to_idx:
                raise RuntimeError(
                    "MiniImageNet class {!r} has no global class ID.".format(wnid)
                )
            path = osp.join(self.IMAGE_PATH, name)
            if not osp.isfile(path):
                raise FileNotFoundError(
                    "MiniImageNet CSV references a missing image: {}".format(path)
                )
            label = self.class_to_idx[wnid]
            self.data.append(path)
            self.targets.append(label)
            self.data2label[path] = label

        # Class names are optional metadata; data loading does not depend on it.
        self.classes = self.mapping_clsidx_to_txt()
        self.wnids = [
            self.classes.get(name, name) for name in self.class_wnids
        ]

        if train:
            image_size = 224
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            image_size = 224
            self.transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                # transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)


    def SelectfromTxt(self, data2label, index_path):
        if not index_path or not osp.isfile(index_path):
            raise FileNotFoundError("MiniImageNet session index file not found: {}".format(index_path))

        basename_to_paths = {}
        for path in data2label:
            basename_to_paths.setdefault(osp.basename(path), []).append(path)

        data_tmp = []
        targets_tmp = []
        missing = []
        for raw_line in open(index_path, "r", encoding="utf-8"):
            line = raw_line.strip().replace("\\\\", "/")
            if not line:
                continue
            normalized = line.lstrip("./")
            candidates = []
            if osp.isabs(line):
                candidates.append(line)
            candidates.extend([
                osp.join(self.root, normalized),
                osp.join(self.IMAGE_PATH, osp.basename(normalized)),
            ])
            if "/images/" in normalized:
                candidates.append(osp.join(self.IMAGE_PATH, normalized.split("/images/", 1)[1]))

            resolved = next((p for p in candidates if p in data2label and osp.isfile(p)), None)
            if resolved is None:
                matches = basename_to_paths.get(osp.basename(normalized), [])
                if len(matches) == 1:
                    resolved = matches[0]
            if resolved is None:
                missing.append(line)
                continue
            data_tmp.append(resolved)
            targets_tmp.append(int(data2label[resolved]))

        if missing:
            raise RuntimeError(
                "MiniImageNet session index {} has {} unresolved paths. Examples: {}"
                .format(index_path, len(missing), ", ".join(missing[:5]))
            )
        if not data_tmp:
            raise RuntimeError("MiniImageNet session index is empty: {}".format(index_path))
        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        target_array = np.asarray(targets, dtype=np.int64)
        selected_indices = np.flatnonzero(np.isin(target_array, np.asarray(index, dtype=np.int64)))
        if selected_indices.size == 0:
            raise RuntimeError(
                "MiniImageNet class selection is empty for class IDs: {}"
                .format(list(index))
            )
        selected_data = [data[position] for position in selected_indices]
        selected_targets = target_array[selected_indices].tolist()
        return selected_data, selected_targets

    def mapping_clsidx_to_txt(self):
        classes = {}
        # 这里的路径需要修改一下
        map_path = osp.join(osp.dirname(__file__), "map_clsloc.txt")
        if not osp.isfile(map_path):
            return classes
        lines = [x.strip() for x in open(map_path, 'r').readlines()][1:]
        for l in lines:
            name, class_num, class_txt = l.split(' ')
            if name not in classes.keys():
                classes[name]=class_txt
        return classes
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets


if __name__ == '__main__':
    txt_path = "../../data/index_list/mini_imagenet/session_2.txt"
    base_class = 100
    class_index = np.arange(base_class)
    dataroot = '../../data'

    trainset = MiniImageNet(root=dataroot, train=True, transform=None, index_path=txt_path)
    # print(trainset.targets)
    cls = np.unique(trainset.targets)
    batch_size_base = len(trainset)
    print(batch_size_base)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)
