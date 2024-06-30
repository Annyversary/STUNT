import sys
import torch

from torchmeta.utils.data import BatchMetaDataLoader
from common.args import parse_args
from common.utils import get_optimizer, load_model
from data.dataset import get_meta_dataset
from models.model import get_model
from train.trainer import meta_trainer
from utils import Logger, set_random_seed, cycle

def main(rank, P):
    P.rank = rank

    """ Setze das Torch-Gerät """
    if torch.cuda.is_available():
        torch.cuda.set_device(P.rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    """ Fixiere Zufälligkeit """
    set_random_seed(P.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    """ Definiere Dataset und Dataloader """
    kwargs = {'batch_size': P.batch_size, 'shuffle': True,
              'pin_memory': True, 'num_workers': 2}
    train_set, val_set = get_meta_dataset(P, dataset='income')

    train_loader = train_set
    test_loader = val_set

    """ Initialisiere Modell, Optimierer und Scheduler """
    model = get_model(P, P.model).to(device)
    optimizer = get_optimizer(P, model)

    """ Definiere Trainings- und Testtyp """
    from train import setup as train_setup
    from evals import setup as test_setup
    train_func, fname, today = train_setup(P.mode, P)
    test_func = test_setup(P.mode, P)

    """ Definiere Logger """
    logger = Logger(fname, ask=P.resume_path is None, today=today, rank=P.rank)
    logger.log(P)
    logger.log(model)

    """ Lade Modell, falls notwendig """
    load_model(P, model, logger)

    """ Trainiere Modell """
    meta_trainer(P, train_func, test_func, model, optimizer, train_loader, test_loader, logger)

    """ Schließe TensorBoard """
    logger.close_writer()

if __name__ == "__main__":
    """ Definiere Argumente """
    P = parse_args()

    P.world_size = torch.cuda.device_count()
    P.distributed = P.world_size > 1
    if P.distributed:
        print("Currently, DDP is not supported, should consider transductive BN before using DDP",
              file=sys.stderr)
    else:
        main(0, P)
