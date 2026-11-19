from yacs.config import CfgNode as CN
from utils.util import set_gpu, set_seed
import argparse


def print_args(cfg):
    print("************")
    print("** Config **")
    print("************")
    print(cfg)
    print("************")


def extend_cfg(cfg):
    """Register all configuration fields before loading YAML files."""
    cfg.DEVICE = CN()
    cfg.DEVICE.DEVICE_NAME = ""
    cfg.DEVICE.GPU_ID = ""

    cfg.METHOD = "BIMC"
    cfg.SEED = -1

    cfg.DATASET = CN()
    cfg.DATASET.NAME = ""
    cfg.DATASET.ROOT = ""
    cfg.DATASET.GPT_PATH = ""
    cfg.DATASET.NUM_CLASSES = -1
    cfg.DATASET.NUM_INIT_CLS = -1
    cfg.DATASET.NUM_INC_CLS = -1
    cfg.DATASET.NUM_BASE_SHOT = -1
    cfg.DATASET.NUM_INC_SHOT = -1
    cfg.DATASET.BETA = -1.0
    cfg.DATASET.ENSEMBLE_ALPHA = -1.0

    cfg.DATALOADER = CN()
    cfg.DATALOADER.TRAIN = CN()
    cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE = -1
    cfg.DATALOADER.TRAIN.BATCH_SIZE_INC = -1
    cfg.DATALOADER.TEST = CN()
    cfg.DATALOADER.TEST.BATCH_SIZE = -1
    cfg.DATALOADER.NUM_WORKERS = -1

    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ""

    cfg.TRAINER = CN()
    cfg.TRAINER.BiMC = CN()
    cfg.TRAINER.BiMC.PREC = ""
    cfg.TRAINER.BiMC.VISION_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_I = -1.0
    cfg.TRAINER.BiMC.TAU = -1
    cfg.TRAINER.BiMC.TEXT_CALIBRATION = False
    cfg.TRAINER.BiMC.LAMBDA_T = -1.0
    cfg.TRAINER.BiMC.GAMMA_BASE = -1.0
    cfg.TRAINER.BiMC.GAMMA_INC = -1.0
    cfg.TRAINER.BiMC.USING_ENSEMBLE = False

    cfg.TRAINER.CSPC = CN()
    cfg.TRAINER.CSPC.PREC = "fp32"
    cfg.TRAINER.CSPC.N_CTX = 4
    cfg.TRAINER.CSPC.PROMPT_INIT_TEXT = "a photo of a"
    cfg.TRAINER.CSPC.BASE_EPOCHS = 20
    cfg.TRAINER.CSPC.INC_EPOCHS = 10
    cfg.TRAINER.CSPC.BASE_LR = 0.001
    cfg.TRAINER.CSPC.INC_PROMPT_LR = 0.0002
    cfg.TRAINER.CSPC.INC_SCORER_LR = 0.005
    cfg.TRAINER.CSPC.INC_GATE_LR = 0.005
    cfg.TRAINER.CSPC.WEIGHT_DECAY = 0.0
    cfg.TRAINER.CSPC.SCORER_TAU = 0.1
    cfg.TRAINER.CSPC.ALPHA_IMG_TXT = 1.0
    cfg.TRAINER.CSPC.BETA_SEM = 0.1
    cfg.TRAINER.CSPC.GAMMA_PROX = 0.01
    cfg.TRAINER.CSPC.GRAD_CLIP = 1.0
    cfg.TRAINER.CSPC.TEXT_CHUNK_SIZE = 64
    cfg.TRAINER.CSPC.OUTPUT_DIR = "./output/cspc"


def setup_cfg(dataset_cfg_file, method_cfg_file):
    cfg = CN()
    extend_cfg(cfg)
    cfg.merge_from_file(dataset_cfg_file)
    cfg.merge_from_file(method_cfg_file)
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run the FSCIL pipeline")
    parser.add_argument("--data_cfg", type=str, required=True, help="Dataset configuration file")
    parser.add_argument("--train_cfg", type=str, required=True, help="Method configuration file")
    args = parser.parse_args()

    cfg = setup_cfg(args.data_cfg, args.train_cfg)
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    if str(cfg.METHOD).upper() == "CSPC":
        from engine.cspc_engine import CSPCRunner
        engine = CSPCRunner(cfg)
    else:
        from engine.engine import Runner
        engine = Runner(cfg)
    engine.run()


if __name__ == "__main__":
    main()
