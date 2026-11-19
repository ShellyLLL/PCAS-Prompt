import os
import re

from .dataset_base import DatasetBase

class CUB200(DatasetBase):

    REQUIRED_METADATA = (
        "classes.txt",
        "images.txt",
        "image_class_labels.txt",
        "train_test_split.txt",
    )

    def __init__(self, root):
        super(CUB200, self).__init__(root=root, name="cub200")
        self.root = os.path.abspath(os.path.expanduser(root))
        self.classes = CLASSES
        self.dataset_root = self._resolve_dataset_root()

        (
            self.train_data,
            self.train_targets,
            self.test_data,
            self.test_targets,
        ) = self.load_data()

        self.gpt_prompt_path = None

    def _resolve_dataset_root(self):
        """Accept either the CUB_200_2011 directory or its parent directory."""
        candidates = (
            self.root,
            os.path.join(self.root, "CUB_200_2011"),
        )
        for candidate in candidates:
            metadata_exists = all(
                os.path.isfile(os.path.join(candidate, filename))
                for filename in self.REQUIRED_METADATA
            )
            images_exist = os.path.isdir(os.path.join(candidate, "images"))
            if metadata_exists and images_exist:
                return candidate

        expected = ", ".join(self.REQUIRED_METADATA)
        raise FileNotFoundError(
            "CUB-200 dataset root is invalid. ROOT must point to the directory "
            "containing the metadata files "
            f"({expected}) and images/. Checked: {', '.join(candidates)}"
        )

    @staticmethod
    def _normalize_class_name(name):
        name = re.sub(r"^\s*\d+\.", "", str(name))
        name = name.replace("'", "").replace("_", " ").replace("-", " ")
        name = re.sub(r"[^A-Za-z0-9\s]", " ", name)
        return " ".join(name.lower().split())

    def _validate_class_metadata(self, class_file):
        class_records = self.list2dict(self.text_read(class_file))
        expected_ids = set(range(1, len(self.classes) + 1))
        observed_ids = set(class_records)
        if observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)
            extra = sorted(observed_ids - expected_ids)
            raise ValueError(
                "CUB classes.txt must contain exactly class IDs 1.."
                f"{len(self.classes)}; missing={missing[:10]}, extra={extra[:10]}."
            )

        mismatches = []
        for class_id, metadata_name in class_records.items():
            code_name = self.classes[class_id - 1]
            if self._normalize_class_name(metadata_name) != self._normalize_class_name(code_name):
                mismatches.append((class_id, metadata_name, code_name))
        if mismatches:
            class_id, metadata_name, code_name = mismatches[0]
            raise ValueError(
                "CUB class-name mapping does not match datasets/cub200.py: "
                f"class ID {class_id}, classes.txt={metadata_name!r}, "
                f"code={code_name!r}. Check class ordering and spelling."
            )

    def get_class_name(self):
        return self.classes

    def get_train_data(self):
        return self.train_data, self.train_targets

    def get_test_data(self):
        return self.test_data, self.test_targets

    def load_data(self):
        image_file = os.path.join(self.dataset_root, "images.txt")
        split_file = os.path.join(self.dataset_root, "train_test_split.txt")
        class_file = os.path.join(self.dataset_root, "image_class_labels.txt")
        classes_file = os.path.join(self.dataset_root, "classes.txt")
        image_root = os.path.join(self.dataset_root, "images")

        self._validate_class_metadata(classes_file)
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))
        id2class = self.list2dict(self.text_read(class_file))

        image_ids = set(id2image)
        if image_ids != set(id2train) or image_ids != set(id2class):
            raise ValueError(
                "CUB metadata ID sets differ between images.txt, "
                "train_test_split.txt, and image_class_labels.txt."
            )

        invalid_splits = [
            (image_id, split)
            for image_id, split in id2train.items()
            if split not in {"0", "1"}
        ]
        if invalid_splits:
            raise ValueError(
                f"CUB train_test_split.txt contains invalid flags: {invalid_splits[:10]}."
            )

        invalid_labels = []
        for image_id, raw_label in id2class.items():
            try:
                class_id = int(raw_label)
            except ValueError:
                invalid_labels.append((image_id, raw_label))
                continue
            if not 1 <= class_id <= len(self.classes):
                invalid_labels.append((image_id, raw_label))
        if invalid_labels:
            raise ValueError(
                f"CUB image_class_labels.txt contains invalid class IDs: {invalid_labels[:10]}."
            )

        train_data, train_targets = [], []
        test_data, test_targets = [], []
        for image_id in sorted(image_ids):
            class_id = int(id2class[image_id]) - 1
            image_path = os.path.join(image_root, id2image[image_id])
            if id2train[image_id] == "1":
                train_data.append(image_path)
                train_targets.append(class_id)
            else:
                test_data.append(image_path)
                test_targets.append(class_id)

        if not train_data or not test_data:
            raise ValueError("CUB dataset must contain non-empty train and test splits.")

        return train_data, train_targets, test_data, test_targets

    def text_read(self, file):
        with open(file, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    @staticmethod
    def list2dict(lines):
        records = {}
        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed CUB metadata row: {line!r}")
            try:
                record_id = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"Invalid CUB metadata ID: {parts[0]!r}") from exc
            if record_id in records:
                raise ValueError(f"Duplicate CUB metadata ID: {record_id}")
            records[record_id] = parts[1].strip()
        return records


CLASSES = ['Black footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove billed Ani', 'Crested Auklet', 
           'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', 'Brewer Blackbird', 'Red winged Blackbird', 
           'Rusty Blackbird', 'Yellow headed Blackbird', 'Bobolink', 'Indigo Bunting', 'Lazuli Bunting', 'Painted Bunting', 
           'Cardinal', 'Spotted Catbird', 'Gray Catbird', 'Yellow breasted Chat', 'Eastern Towhee', 'Chuck will Widow', 'Brandt Cormorant', 
           'Red faced Cormorant', 'Pelagic Cormorant', 'Bronzed Cowbird', 'Shiny Cowbird', 'Brown Creeper', 'American Crow', 'Fish Crow', 
           'Black billed Cuckoo', 'Mangrove Cuckoo', 'Yellow billed Cuckoo', 'Gray crowned Rosy Finch', 'Purple Finch', 'Northern Flicker', 
           'Acadian Flycatcher', 'Great Crested Flycatcher', 'Least Flycatcher', 'Olive sided Flycatcher', 'Scissor tailed Flycatcher', 
           'Vermilion Flycatcher', 'Yellow bellied Flycatcher', 'Frigatebird', 'Northern Fulmar', 'Gadwall', 'American Goldfinch', 
           'European Goldfinch', 'Boat tailed Grackle', 'Eared Grebe', 'Horned Grebe', 'Pied billed Grebe', 'Western Grebe', 'Blue Grosbeak', 
           'Evening Grosbeak', 'Pine Grosbeak', 'Rose breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous winged Gull', 
           'Heermann Gull', 'Herring Gull', 'Ivory Gull', 'Ring billed Gull', 'Slaty backed Gull', 'Western Gull', 'Anna Hummingbird', 
           'Ruby throated Hummingbird', 'Rufous Hummingbird', 'Green Violetear', 'Long tailed Jaeger', 'Pomarine Jaeger', 'Blue Jay', 
           'Florida Jay', 'Green Jay', 'Dark eyed Junco', 'Tropical Kingbird', 'Gray Kingbird', 'Belted Kingfisher', 'Green Kingfisher', 
           'Pied Kingfisher', 'Ringed Kingfisher', 'White breasted Kingfisher', 'Red legged Kittiwake', 'Horned Lark', 'Pacific Loon', 
           'Mallard', 'Western Meadowlark', 'Hooded Merganser', 'Red breasted Merganser', 'Mockingbird', 'Nighthawk', 'Clark Nutcracker', 
           'White breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole', 'Orchard Oriole', 'Scott Oriole', 'Ovenbird', 'Brown Pelican', 
           'White Pelican', 'Western Wood Pewee', 'Sayornis', 'American Pipit', 'Whip poor Will', 'Horned Puffin', 'Common Raven', 
           'White necked Raven', 'American Redstart', 'Geococcyx', 'Loggerhead Shrike', 'Great Grey Shrike', 'Baird Sparrow', 
           'Black throated Sparrow', 'Brewer Sparrow', 'Chipping Sparrow', 'Clay colored Sparrow', 'House Sparrow', 'Field Sparrow', 
           'Fox Sparrow', 'Grasshopper Sparrow', 'Harris Sparrow', 'Henslow Sparrow', 'Le Conte Sparrow', 'Lincoln Sparrow', 
           'Nelson Sharp tailed Sparrow', 'Savannah Sparrow', 'Seaside Sparrow', 'Song Sparrow', 'Tree Sparrow', 'Vesper Sparrow', 
           'White crowned Sparrow', 'White throated Sparrow', 'Cape Glossy Starling', 'Bank Swallow', 'Barn Swallow', 'Cliff Swallow', 
           'Tree Swallow', 'Scarlet Tanager', 'Summer Tanager', 'Arctic Tern', 'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern', 
           'Forsters Tern', 'Least Tern', 'Green tailed Towhee', 'Brown Thrasher', 'Sage Thrasher', 'Black capped Vireo', 'Blue headed Vireo', 
           'Philadelphia Vireo', 'Red eyed Vireo', 'Warbling Vireo', 'White eyed Vireo', 'Yellow throated Vireo', 'Bay breasted Warbler', 
           'Black and white Warbler', 'Black throated Blue Warbler', 'Blue winged Warbler', 'Canada Warbler', 'Cape May Warbler', 'Cerulean Warbler', 
           'Chestnut sided Warbler', 'Golden winged Warbler', 'Hooded Warbler', 'Kentucky Warbler', 'Magnolia Warbler', 'Mourning Warbler', 
           'Myrtle Warbler', 'Nashville Warbler', 'Orange crowned Warbler', 'Palm Warbler', 'Pine Warbler', 'Prairie Warbler', 'Prothonotary Warbler', 
           'Swainson Warbler', 'Tennessee Warbler', 'Wilson Warbler', 'Worm eating Warbler', 'Yellow Warbler', 'Northern Waterthrush', 
           'Louisiana Waterthrush', 'Bohemian Waxwing', 'Cedar Waxwing', 'American Three toed Woodpecker', 'Pileated Woodpecker', 
           'Red bellied Woodpecker', 'Red cockaded Woodpecker', 'Red headed Woodpecker', 'Downy Woodpecker', 'Bewick Wren', 'Cactus Wren', 
           'Carolina Wren', 'House Wren', 'Marsh Wren', 'Rock Wren', 'Winter Wren', 'Common Yellowthroat']
