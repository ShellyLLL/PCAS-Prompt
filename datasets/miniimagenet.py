from .dataset_base import DatasetBase
import os


class MiniImagenet(DatasetBase):

    def __init__(self, root):
        super(MiniImagenet, self).__init__(root=root, name="miniimagenet")
        self.root = root
        self.classes = CLASSES
        self.image_folder = os.path.join(self.root, "miniimagenet/images")
        self.split_folder = os.path.join(self.root, "miniimagenet/split")

        train_csv = os.path.join(self.split_folder, "train.csv")
        test_csv = os.path.join(self.split_folder, "test.csv")
        self.train_data, self.train_targets, self.label_to_index = (
            self.load_data_targets_from_csv(train_csv)
        )
        if len(self.label_to_index) != len(self.classes):
            raise ValueError(
                "MiniImageNet train split contains "
                f"{len(self.label_to_index)} classes, but CLASSES defines {len(self.classes)}."
            )
        self.test_data, self.test_targets, _ = self.load_data_targets_from_csv(
            test_csv, class2index=self.label_to_index, allow_new_classes=False
        )
        self.gpt_prompt_path = None

    def load_data_targets_from_csv(
        self, csv_path, class2index=None, allow_new_classes=True
    ):
        data = []
        targets = []
        if class2index is None:
            class2index = {}

        with open(csv_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()[1:]
            for line_number, line in enumerate(lines, start=2):
                parts = line.strip().split(",")
                if len(parts) != 2:
                    raise ValueError(
                        f"Malformed MiniImageNet CSV row at {csv_path}:{line_number}: {line!r}"
                    )
                relative_path, label = parts
                if label not in class2index:
                    if not allow_new_classes:
                        raise ValueError(
                            f"Test split contains class {label!r} absent from train split."
                        )
                    class2index[label] = len(class2index)

                data.append(os.path.join(self.image_folder, relative_path))
                targets.append(class2index[label])

        return data, targets, class2index

    def get_class_name(self):
        return self.classes

    def get_train_data(self):
        return self.train_data, self.train_targets

    def get_test_data(self):
        return self.test_data, self.test_targets


CLASSES = ['house finch', 'robin', 'triceratops', 'green mamba', 'harvestman',
           'toucan', 'goose', 'jellyfish', 'nematode', 'king crab', 'dugong',
           'Walker hound', 'Ibizan hound', 'Saluki', 'golden retriever', 'Gordon setter',
           'komondor', 'boxer', 'Tibetan mastiff', 'French bulldog', 'malamute', 'dalmatian',
           'Newfoundland', 'miniature poodle', 'white wolf', 'African hunting dog', 'Arctic fox',
           'lion', 'meerkat', 'ladybug', 'rhinoceros beetle', 'ant', 'black-footed ferret',
           'three-toed sloth', 'rock beauty', 'aircraft carrier', 'ashcan', 'barrel', 'beer bottle',
           'bookshop', 'cannon', 'carousel', 'carton', 'catamaran', 'chime', 'clog', 'cocktail shaker',
           'combination lock', 'crate', 'cuirass', 'dishrag', 'dome', 'electric guitar', 'file', 'fire screen',
           'frying pan', 'garbage truck', 'hair slide', 'holster', 'horizontal bar', 'hourglass', 'iPod', 'lipstick',
           'miniskirt', 'missile', 'mixing bowl', 'oboe', 'organ', 'parallel bars', 'pencil box', 'photocopier',
           'poncho', 'prayer rug', 'reel', 'school bus', 'scoreboard', 'slot', 'snorkel', 'solar dish',
           'spider web', 'stage', 'tank', 'theater curtain', 'tile roof', 'tobacco shop', 'unicycle', 'upright',
           'vase', 'wok', 'worm fence', 'yawl', 'street sign', 'consomme', 'trifle', 'hotdog', 'orange', 'cliff',
           'coral reef', 'bolete', 'ear']
