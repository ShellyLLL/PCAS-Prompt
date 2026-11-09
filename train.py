import argparse
import importlib
from utils import *
import torch

MODEL_DIR=None
# DATA_DIR = './local_datasets/'
DATA_DIR = '/data/lx/fscil/data'
PROJECT='base'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-out', type=str, default='PCAS')
    parser.add_argument('--index_list_root', type=str, default=None,
                        help='Session split root. Defaults to <dataroot>/index_list.')
    parser.add_argument('--validation_mode', action='store_true',
                        help='Run pseudo-FSCIL validation using official training data only.')
    parser.add_argument('--validation_seed', type=int, default=2026,
                        help='Fixed seed for pseudo-FSCIL class/sample partitioning.')
    parser.add_argument('--validation_base_query_fraction', type=float, default=0.2,
                        help='Fraction of pseudo-base training images reserved for validation queries.')
    parser.add_argument('--pcas_layer', type=int, default=6)
    parser.add_argument('--bottleneck_dim', type=int, default=96)
    parser.add_argument('--num_components', type=int, default=4)
    parser.add_argument('--prompt_dropout', type=float, default=0.1)
    parser.add_argument('--route_temperature', type=float, default=0.2)
    parser.add_argument('--residual_init', type=float, default=0.1)
    parser.add_argument('--residual_max', type=float, default=0.5)
    parser.add_argument('--affinity_warmup_epochs', type=int, default=10,
                        help='Linear warm-up epochs for the PCAC affinity loss weight.')
    parser.add_argument('--classes_per_batch', type=int, default=16)
    parser.add_argument('--samples_per_class', type=int, default=4)
    parser.add_argument('--pcas_lr', type=float, default=1e-3)
    parser.add_argument('--pcas_weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_proto', type=float, default=0.5)
    parser.add_argument('--lambda_affinity', type=float, default=0.1)
    parser.add_argument('--beta_affinity', type=float, default=0.5)
    parser.add_argument('--affinity_scale', type=float, default=16.0)
    parser.add_argument('--pcas_logit_scale', type=float, default=16.0)
    parser.add_argument('--teen_alpha', type=float, default=0.5)
    parser.add_argument('--teen_tau', type=float, default=16.0)

    parser.add_argument('-epochs_base', type=int, default=50)
    parser.add_argument('-temperature', type=float, default=16)

    parser.add_argument('-batch_size_base', type=int, default=64)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=128)
# ft_dot means using linear classifier, ft_cos means using cosine classifier
# ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    # for episode learning

    # for cec
#lr for graph attention network


    # about training
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=-1)
    # parser.add_argument('-seeds', nargs='+', help='<Required> Set flag', required=True, default=1)
    
    parser.add_argument('-vit', action='store_true')
    parser.add_argument('-baseline', action='store_true')
    parser.add_argument('-clip', action='store_true')
    
    
    
    parser.add_argument('-l2p', action='store_true')
    parser.add_argument('-dp', action='store_true')
    
    # parser.add_argument('-scratch', action='store_true')
    parser.add_argument('-scratch', action='store_true')

    # for LGSP



    





    # VFPT
# False # True


# False # True
# FVG


    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100', 'FGVCAircraft', 'iNF200'])

    return parser


if __name__ == '__main__':
    
    parser = get_command_line_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    if args.vit:
        if args.baseline:
            trainer = importlib.import_module('models.%s.ViT_fscil_Baseline_trainer' % (args.project)).ViT_Baseline_FSCILTrainer(args)
        #* L2P / DP
        elif args.l2p:
            pass
        elif args.dp:
            pass
        else:
            trainer = importlib.import_module('models.%s.ViT_fscil_trainer' % (args.project)).ViT_FSCILTrainer(args)
        
    elif args.clip:
        trainer = importlib.import_module('models.%s.CLIP_fscil_trainer' % (args.project)).CLIP_FSCILTrainer(args)
    else:
        trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()