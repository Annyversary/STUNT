import torch
from torchvision import transforms

from torchmeta.transforms import ClassSplitter, Categorical

from data.income import Income


def get_meta_dataset(P, dataset, only_test=False):
    """
    Erstellt und gibt die Meta-Datasets für Training und Validierung basierend auf den Eingabeparametern zurück.

    Args:
        P: Argumente, die durch den Argument-Parser übergeben wurden.
        dataset (str): Der Name des zu verwendenden Datasets.
        only_test (bool): Flag, das angibt, ob nur das Test-Dataset erstellt werden soll.

    Returns:
        tuple: Ein Tuple, das das Trainings- und Validierungs-Meta-Dataset enthält.
    """

    if dataset == 'income':
        # Erstelle das Trainings-Meta-Dataset
        meta_train_dataset = Income(
            tabular_size=105,
            seed=P.seed,
            source='train',
            shot=P.num_shots,
            tasks_per_batch=P.batch_size,
            test_num_way=P.num_ways,
            query=P.num_shots_test
        )

        # Erstelle das Validierungs-Meta-Dataset
        meta_val_dataset = Income(
            tabular_size=105,
            seed=P.seed,
            source='val',
            shot=1,
            tasks_per_batch=P.test_batch_size,
            test_num_way=2,
            query=30
        )

    else:
        # Falls ein nicht implementiertes Dataset angefordert wird, werfe einen Fehler
        raise NotImplementedError()

    return meta_train_dataset, meta_val_dataset
