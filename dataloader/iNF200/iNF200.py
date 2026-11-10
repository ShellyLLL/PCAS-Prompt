import os
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class iNF200(Dataset):
    def __init__(self, root='./', train=True,
                 index_path=None, index=None, 
                 base_sess=False, is_clip=False):
 
        self.root = os.path.expanduser(root)
        self.train = train
        self.base_sess = base_sess
        self.data = []
        self.targets = []
        
      
        self._build_label_mapping()
        
        if train:
            if base_sess:
                self._select_by_txt(index_path)
            else:
                self._select_by_txt(index_path)
        else:
            self._select_test_set(index)

       
        self.transform = self._build_transform(is_clip)

    def _build_label_mapping(self):
        """Use one stable class order for both train_mini and validation data."""
        train_dir = osp.join(self.root, "iNF200", "train_mini")
        val_dir = osp.join(self.root, "iNF200", "val")
        if not osp.isdir(train_dir):
            raise FileNotFoundError("iNF200 training directory not found: {}".format(train_dir))
        if not osp.isdir(val_dir):
            raise FileNotFoundError("iNF200 validation directory not found: {}".format(val_dir))

        class_folders = sorted(
            name for name in os.listdir(train_dir)
            if osp.isdir(osp.join(train_dir, name))
        )
        if not class_folders:
            raise RuntimeError("No iNF200 class directories found in: {}".format(train_dir))

        self.label_dict = {name: idx for idx, name in enumerate(class_folders)}
        self.reverse_label_dict = {
            idx: name.split("_")[-1] for name, idx in self.label_dict.items()
        }

        val_folders = {
            name for name in os.listdir(val_dir)
            if osp.isdir(osp.join(val_dir, name))
        }
        unknown = sorted(val_folders.difference(self.label_dict))
        if unknown:
            raise RuntimeError(
                "iNF200 validation contains class folders absent from train_mini: {}"
                .format(unknown[:10])
            )

    def _select_by_classes(self, class_indices):

        selected_data = []
        
       
        train_dir = osp.join(self.root, "iNF200", "train_mini")
        all_classes = sorted(
            name for name in os.listdir(train_dir)
            if osp.isdir(osp.join(train_dir, name))
        )
        
        for class_idx in class_indices:
            if class_idx >= len(all_classes):
                raise ValueError(f"Class index {class_idx} out of range")
                
            class_folder = all_classes[class_idx]
            class_path = osp.join(self.root, "iNF200", "train_mini", class_folder)
            
          
            samples = sorted(os.listdir(class_path))[:50]
            for img_name in samples:
                img_path = osp.join(class_path, img_name)
                if osp.exists(img_path):
                    selected_data.append((
                        img_path,
                        self.label_dict[class_folder] 
                    ))

        if not selected_data:
            raise RuntimeError("No data found for selected classes")
            
        self.data, self.targets = zip(*selected_data)

    def _select_by_txt(self, index_path):
        if not index_path or not osp.isfile(index_path):
            raise FileNotFoundError("iNF200 session index file not found: {}".format(index_path))

        selected_data = []
        malformed = []
        missing = []
        for raw_line in open(index_path, "r", encoding="utf-8"):
            rel_path = raw_line.strip().replace("\\", "/").lstrip("./")
            if not rel_path:
                continue
            parts = rel_path.split("/")
            if len(parts) != 3 or parts[0] != "train_mini":
                malformed.append(rel_path)
                continue

            class_folder = parts[1]
            if class_folder not in self.label_dict:
                missing.append(rel_path)
                continue

            full_path = osp.join(self.root, "iNF200", *parts)
            if not osp.isfile(full_path):
                missing.append(rel_path)
                continue
            selected_data.append((full_path, self.label_dict[class_folder]))

        if malformed or missing:
            examples = (malformed + missing)[:5]
            raise RuntimeError(
                "iNF200 session index {} contains {} malformed or unresolved entries. "
                "Expected paths like train_mini/<class_folder>/<image>. Examples: {}"
                .format(index_path, len(malformed) + len(missing), examples)
            )
        if not selected_data:
            raise RuntimeError("iNF200 session index is empty: {}".format(index_path))

        self.data, self.targets = zip(*selected_data)
        self.data, self.targets = list(self.data), list(self.targets)

    def _select_test_set(self, class_indices=None):
        val_dir = osp.join(self.root, "iNF200", "val")
        if not osp.isdir(val_dir):
            raise FileNotFoundError("iNF200 validation directory not found: {}".format(val_dir))

        if class_indices is None:
            allowed_classes = set(self.label_dict.values())
        else:
            allowed_classes = {int(value) for value in class_indices}
        invalid = sorted(
            value for value in allowed_classes
            if value < 0 or value >= len(self.label_dict)
        )
        if invalid:
            raise ValueError("iNF200 test class indices are out of range: {}".format(invalid))

        selected_data = []
        observed_classes = set()
        for class_folder in sorted(os.listdir(val_dir)):
            class_path = osp.join(val_dir, class_folder)
            if not osp.isdir(class_path):
                continue
            if class_folder not in self.label_dict:
                raise RuntimeError(
                    "Unknown iNF200 validation class folder: {}".format(class_folder)
                )
            class_id = self.label_dict[class_folder]
            if class_id not in allowed_classes:
                continue
            for img_name in sorted(os.listdir(class_path)):
                img_path = osp.join(class_path, img_name)
                if osp.isfile(img_path):
                    selected_data.append((img_path, class_id))
                    observed_classes.add(class_id)

        missing_classes = sorted(allowed_classes.difference(observed_classes))
        if missing_classes:
            raise RuntimeError(
                "iNF200 validation set has no images for selected class IDs: {}"
                .format(missing_classes)
            )
        self.data, self.targets = zip(*selected_data)
        self.data, self.targets = list(self.data), list(self.targets)

    def _build_transform(self, is_clip):
       
        if is_clip:
            return transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        if self.train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, target = self.data[idx], self.targets[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), target

    @property
    def num_classes(self):
        return len(self.label_dict)

if __name__ == '__main__':
   
    base_dataset = iNF200Dataset(
        root="/path/to/iNF200",
        train=True,
        index=list(range(100)), 
        base_sess=True
    )
    
    inc_dataset = iNF200Dataset(
        root="/path/to/iNF200",
        train=True,
        index_path="data/index_list/iNF200/session_2.txt",
        base_sess=False
    )
  
    val_dataset = iNF200Dataset(
        root="/path/to/iNF200",
        train=False
    )
  