import os
import numpy as np
import torch
from dataloader.sampler import CategoriesSampler


def _session_index_path(args, session):
    """Resolve the original FSCIL split numbering (session 1 is base)."""
    session = int(session)
    if session < 0:
        raise ValueError("FSCIL session index cannot be negative.")

    root = getattr(args, "index_list_root", None)
    if root:
        root = os.path.abspath(os.path.expanduser(root))
    else:
        data_root = getattr(args, "_configured_data_root", args.dataroot)
        root = os.path.join(
            os.path.abspath(os.path.expanduser(data_root)), "index_list"
        )

    dataset_dirs = {
        "cub200": ["cub200", "CUB200", "CUB_200_2011"],
        "cifar100": ["cifar100", "CIFAR100", "CIFAR_100"],
        "mini_imagenet": ["mini_imagenet", "miniimagenet", "MiniImageNet"],
        "FGVCAircraft": ["FGVCAircraft", "fgvc-aircraft", "fgvc_aircraft"],
        "iNF200": ["iNF200", "inf200", "iNF-200"],
    }.get(args.dataset)
    if dataset_dirs is None:
        raise ValueError("Unsupported FSCIL dataset: {!r}".format(args.dataset))

    # The original split convention numbers the base session as session_1.
    # Thus the first incremental task (trainer session=1) uses session_2.txt.
    split_number = session + 1
    filename = "session_{}.txt".format(split_number)
    candidates = [os.path.join(root, name, filename) for name in dataset_dirs]
    for path in candidates:
        if os.path.isfile(path):
            if os.path.getsize(path) == 0:
                raise ValueError("FSCIL session split is empty: {}".format(path))
            return path
    raise FileNotFoundError(
        "Missing FSCIL session split for dataset={!r}, trainer_session={}. "
        "The expected split is session_{}.txt. Checked: {}".format(
            args.dataset, session, split_number, ", ".join(candidates)
        )
    )


def _has_files(directory, filenames):
    return os.path.isdir(directory) and all(
        os.path.isfile(os.path.join(directory, name)) for name in filenames
    )


def _resolve_data_roots(args):
    """Resolve the configured data root against the exact paths used by loaders.

    args.dataroot may be the shared data directory or the dataset directory
    itself. args._configured_data_root is kept as the shared root so the
    default index_list directory remains alongside the dataset directories.
    """
    configured_root = os.path.abspath(os.path.expanduser(args.dataroot))
    if not os.path.isdir(configured_root):
        raise FileNotFoundError(
            "Configured FSCIL data root does not exist: {}".format(configured_root)
        )

    root_name = os.path.basename(os.path.normpath(configured_root))

    def cub_valid(loader_root):
        base = os.path.join(loader_root, "CUB_200_2011")
        return (
            _has_files(base, [
                "images.txt", "train_test_split.txt",
                "image_class_labels.txt", "classes.txt",
            ])
            and os.path.isdir(os.path.join(base, "images"))
        )

    def cifar_valid(loader_root):
        return _has_files(
            os.path.join(loader_root, "cifar-100-python"),
            ["train", "test", "meta"],
        )

    def mini_valid(loader_root):
        base = os.path.join(loader_root, "miniimagenet")
        image_dir = os.path.join(base, "images")
        train_csv = (
            os.path.isfile(os.path.join(base, "train.csv"))
            or os.path.isfile(os.path.join(base, "split", "train.csv"))
        )
        test_csv = (
            os.path.isfile(os.path.join(base, "test.csv"))
            or os.path.isfile(os.path.join(base, "split", "test.csv"))
        )
        return os.path.isdir(image_dir) and train_csv and test_csv

    def aircraft_valid(loader_root):
        base = os.path.join(loader_root, "fgvc-aircraft-2013b", "data")
        return (
            _has_files(base, [
                "variants.txt", "images_variant_train.txt",
                "images_variant_test.txt",
            ])
            and os.path.isdir(os.path.join(base, "images"))
        )

    def inf200_valid(loader_root):
        base = os.path.join(loader_root, "iNF200")
        return (
            os.path.isdir(os.path.join(base, "train_mini"))
            and os.path.isdir(os.path.join(base, "val"))
        )

    if args.dataset == "cub200":
        if root_name == "CUB_200_2011":
            candidates = [
                (os.path.dirname(configured_root), os.path.dirname(configured_root)),
                (configured_root, os.path.dirname(configured_root)),
            ]
        else:
            candidates = [
                (configured_root, configured_root),
                (os.path.join(configured_root, "CUB_200_2011"), configured_root),
            ]
        validator = cub_valid
        expected = (
            "<data_root>/CUB_200_2011/{images.txt, train_test_split.txt, "
            "image_class_labels.txt, classes.txt, images/}"
        )

    elif args.dataset == "cifar100":
        if root_name in ("CIFAR_100", "CIFAR100"):
            candidates = [
                (configured_root, os.path.dirname(configured_root)),
                (os.path.dirname(configured_root), os.path.dirname(configured_root)),
            ]
        elif root_name == "cifar-100-python":
            parent = os.path.dirname(configured_root)
            index_root = (
                os.path.dirname(parent)
                if os.path.basename(parent) in ("CIFAR_100", "CIFAR100")
                else parent
            )
            candidates = [(parent, index_root)]
        else:
            candidates = [
                (os.path.join(configured_root, "CIFAR_100"), configured_root),
                (configured_root, configured_root),
                (os.path.join(configured_root, "CIFAR100"), configured_root),
            ]
        validator = cifar_valid
        expected = "<data_root>/cifar-100-python/{train, test, meta}"

    elif args.dataset == "mini_imagenet":
        if root_name == "miniimagenet":
            candidates = [
                (os.path.dirname(configured_root), os.path.dirname(configured_root)),
                (configured_root, os.path.dirname(configured_root)),
            ]
        elif root_name == "images":
            shared_root = os.path.dirname(os.path.dirname(configured_root))
            candidates = [(shared_root, shared_root)]
        else:
            candidates = [
                (configured_root, configured_root),
                (os.path.join(configured_root, "miniimagenet"), configured_root),
                (os.path.dirname(configured_root), os.path.dirname(configured_root)),
            ]
        validator = mini_valid
        expected = (
            "<data_root>/miniimagenet/images and train.csv/test.csv "
            "(or miniimagenet/split/train.csv and test.csv)"
        )

    elif args.dataset == "FGVCAircraft":
        if root_name == "fgvc-aircraft-2013b":
            candidates = [
                (os.path.dirname(configured_root), os.path.dirname(configured_root)),
                (configured_root, os.path.dirname(configured_root)),
            ]
        elif root_name == "data" and os.path.basename(
            os.path.dirname(configured_root)
        ) == "fgvc-aircraft-2013b":
            shared_root = os.path.dirname(os.path.dirname(configured_root))
            candidates = [(shared_root, shared_root)]
        else:
            candidates = [
                (configured_root, configured_root),
                (os.path.join(configured_root, "fgvc-aircraft-2013b"), configured_root),
            ]
        validator = aircraft_valid
        expected = (
            "<data_root>/fgvc-aircraft-2013b/data/{variants.txt, "
            "images_variant_train.txt, images_variant_test.txt, images/}"
        )

    elif args.dataset == "iNF200":
        if root_name == "iNF200":
            candidates = [
                (os.path.dirname(configured_root), os.path.dirname(configured_root)),
                (configured_root, os.path.dirname(configured_root)),
            ]
        else:
            candidates = [
                (configured_root, configured_root),
                (os.path.join(configured_root, "iNF200"), configured_root),
            ]
        validator = inf200_valid
        expected = "<data_root>/iNF200/{train_mini/, val/}"

    else:
        raise ValueError("Unsupported FSCIL dataset: {!r}".format(args.dataset))

    checked = []
    for loader_root, index_root in candidates:
        loader_root = os.path.abspath(loader_root)
        checked.append(loader_root)
        if validator(loader_root):
            args.dataroot = loader_root
            args._configured_data_root = os.path.abspath(index_root)
            return

    raise FileNotFoundError(
        "Could not resolve dataset={!r} from dataroot={!r}. The loader requires "
        "{}. Checked loader roots: {}".format(
            args.dataset, configured_root, expected, ", ".join(checked)
        )
    )


def set_up_datasets(args):
    _resolve_data_roots(args)
    if args.dataset == "cifar100":
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    elif args.dataset == "cub200":
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    elif args.dataset == "mini_imagenet":
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    elif args.dataset == "FGVCAircraft":
        import dataloader.FGVCAircraft.FGVCAircraft as Dataset
        args.base_class = 50
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 11
    elif args.dataset == "iNF200":
        import dataloader.iNF200.iNF200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    else:
        raise ValueError("Unsupported FSCIL dataset: {!r}".format(args.dataset))

    args.Dataset = Dataset

    # CUB/CIFAR/MiniImageNet base training is selected by class ID, so their
    # first consumed support file is session_2.txt. Aircraft/iNF200 base
    # training is explicitly defined by session_1.txt.
    first_indexed_session = 0 if args.dataset in ("FGVCAircraft", "iNF200") else 1
    for session in range(first_indexed_session, args.sessions):
        _session_index_path(args, session)

    return args


def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_base_dataloader(args, clip_trsf=None):
    txt_path = None
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':

        if args.clip:
            trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True, transform=clip_trsf,
                                            index=class_index, base_sess=True,is_vit=args.vit, is_clip=args.clip)
            testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                            index=class_index, base_sess=True,is_vit=args.vit, is_clip=args.clip)
        else:
            trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True, transform=clip_trsf,
                                         index=class_index, base_sess=True,is_vit=args.vit, is_clip=args.clip)
            testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                            index=class_index, base_sess=True,is_vit=args.vit, is_clip=args.clip)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True, is_clip=args.clip)
        
      
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index, is_clip=args.clip)
    if args.dataset == 'FGVCAircraft':
        txt_path = _session_index_path(args, 0)
        trainset = args.Dataset.FGVCAircraft(root=args.dataroot, train=True,
            index_path=txt_path, base_sess=True, is_clip=args.clip)
        testset = args.Dataset.FGVCAircraft(root=args.dataroot, train=False, 
                                            index=class_index, is_clip=args.clip)
        
       
        
    if args.dataset == 'iNF200':
        txt_path = _session_index_path(args, 0)
        trainset = args.Dataset.iNF200(root=args.dataroot, train=True,
            index_path=txt_path, base_sess=True, is_clip=args.clip)
        testset = args.Dataset.iNF200(root=args.dataroot, train=False, 
                                            index=class_index, is_clip=args.clip)
        
     

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True, is_clip=args.clip)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index, is_clip=args.clip)
    

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )

    return trainset, trainloader, testloader

def get_dataloader_RAPF(args, session):
    if session == 0:
        class_index = np.arange(args.base_class)
        if args.dataset == 'cub200':
            trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                        index=class_index, base_sess=True, is_clip=args.clip)
      
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=False,
                                                num_workers=4)
        return trainloader
    else:
        txt_path = _session_index_path(args, session)
        if args.dataset == 'cub200':
            trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                        index_path=txt_path, is_clip=args.clip)
        # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=False,
        #                                           num_workers=args.num_workers)
        if args.batch_size_new == 0:
            batch_size_new = trainset.__len__()
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                    num_workers=args.num_workers)
        # else:
        #     trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
        #                                             num_workers=args.num_workers)
        return trainloader


def get_standard_data_loader(args):
    #todo for the standard CIL Data
    pass

def get_base_dataloader_meta(args):
   
    txt_path = _session_index_path(args, 0)
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True,is_vit=args.vit, is_clip=args.clip)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True,is_vit=args.vit, is_clip=args.clip)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path, is_clip=args.clip)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_index, is_clip=args.clip)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path, is_clip=args.clip)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_index, is_clip=args.clip)


    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    return trainset, trainloader, testloader

def get_new_dataloader(args, session, clip_trsf=None):
    txt_path = _session_index_path(args, session)
    if args.dataset == 'cifar100':
        if args.clip:
            class_index = open(txt_path).read().splitlines()
            trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False, transform=clip_trsf,
                                            index=class_index, base_sess=False,is_vit=args.vit, is_clip=args.clip)
        else:
            class_index = open(txt_path).read().splitlines()
            trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False,is_vit=args.vit, is_clip=args.clip)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path, is_clip=args.clip)
    if args.dataset == 'FGVCAircraft':
        trainset = args.Dataset.FGVCAircraft(root=args.dataroot, train=True,
            index_path=txt_path, is_clip=args.clip)
        
    if args.dataset == 'iNF200':
        trainset = args.Dataset.iNF200(root=args.dataroot, train=True,
            index_path=txt_path, is_clip=args.clip)
    
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path, is_clip=args.clip)
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        if args.clip:
            testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                            index=class_new, base_sess=False,is_vit=args.vit, is_clip=args.clip)
        else:
            testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False, transform=clip_trsf,
                                            index=class_new, base_sess=False,is_vit=args.vit, is_clip=args.clip)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new, is_clip=args.clip)
    if args.dataset == 'FGVCAircraft':
        testset = args.Dataset.FGVCAircraft(root=args.dataroot, train=False,
            index=class_new, is_clip=args.clip)
    
    if args.dataset == 'iNF200':
        testset = args.Dataset.iNF200(root=args.dataroot, train=False,
            index=class_new, is_clip=args.clip)
        
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new, is_clip=args.clip)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list