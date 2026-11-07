"""Select the fixed PCAS spatial-component count using base-only validation."""
from train import get_command_line_parser
from utils import set_gpu
from dataloader.data_utils import get_base_dataloader
from models.base.pcas_trainer import PCASFSCILTrainer
from models.base.pcas_selection import run_component_capacity_selection


def main():
    parser = get_command_line_parser()
    parser.add_argument(
        "--component_candidates", nargs="+", type=int, default=[1, 2, 4, 8, 16],
        help="Fixed component counts to compare; M=1 is required as a control.",
    )
    parser.add_argument(
        "--base_val_fraction", type=float, default=0.1,
        help="Per-class holdout fraction taken only from base training data.",
    )
    parser.add_argument(
        "--base_val_seed", type=int, default=1,
        help="Fixed seed for the stratified base-training validation split.",
    )
    args = parser.parse_args()
    if not args.vit:
        parser.error("PCAS component selection requires the -vit flag.")
    if 1 not in args.component_candidates:
        parser.error("--component_candidates must include M=1.")
    args.num_components = int(args.component_candidates[0])
    args.num_gpu = set_gpu(args)
    trainer = PCASFSCILTrainer(args)
    base_trainset, _, _ = get_base_dataloader(trainer.args)
    run_component_capacity_selection(
        trainer, base_trainset,
        candidates=args.component_candidates,
        validation_fraction=args.base_val_fraction,
        split_seed=args.base_val_seed,
    )


if __name__ == "__main__":
    main()
