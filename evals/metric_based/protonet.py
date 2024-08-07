import torch
from torchmeta.utils.prototype import get_prototypes
from train.metric_based import get_accuracy
from utils import MetricLogger
import torch.nn.functional as F

# Bestimme das Gerät für die Berechnungen (GPU oder CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check(P):
    # Dummy-Funktion, die immer True zurückgibt
    filename_with_today_date = True
    return filename_with_today_date

def test_classifier(P, model, loader, criterion, steps, logger=None):
    """
    Testet den Klassifikator und berechnet die Genauigkeit und den Verlust.

    Args:
        P: Trainingsparameter.
        model: Das zu testende Modell.
        loader: DataLoader für die Testdaten.
        criterion: Verlustfunktion.
        steps: Anzahl der Schritte.
        logger: Logger zum Protokollieren (optional).

    Returns:
        float: Durchschnittliche Genauigkeit.
    """
    # Initialisiere einen MetricLogger für das Protokollieren der Metriken
    metric_logger = MetricLogger(delimiter="  ")

    # Wenn kein Logger angegeben ist, verwende die print-Funktion
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # Wechsle in den Evaluierungsmodus
    mode = model.training
    model.eval()

    # Schleife über die Batches im DataLoader
    for n, batch in enumerate(loader):
        if n * P.test_batch_size > P.max_test_task:
            break

        train_inputs, train_targets = batch['train']

        # Bestimme die Anzahl der Klassen (Wege)
        num_ways = len(set(list(train_targets[0].numpy())))
        train_inputs = train_inputs.to(device)
        train_targets = train_targets.to(device)
        with torch.no_grad():
            train_embeddings = model(train_inputs)

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.to(device)
        test_targets = test_targets.to(device)
        with torch.no_grad():
            test_embeddings = model(test_inputs)

        # Berechne die Prototypen der Trainings-Einbettungen
        prototypes = get_prototypes(train_embeddings, train_targets, num_ways)

        # Berechne die quadratischen Distanzen zwischen den Prototypen und den Test-Einbettungen
        squared_distances = torch.sum((prototypes.unsqueeze(2)
                                       - test_embeddings.unsqueeze(1)) ** 2, dim=-1)
        # Berechne den Verlust
        loss = criterion(-squared_distances, test_targets)

        # Berechne die Genauigkeit
        acc = get_accuracy(prototypes, test_embeddings, test_targets).item()

        # Aktualisiere die Metriken im Logger
        metric_logger.meters['loss'].update(loss.item())
        metric_logger.meters['acc'].update(acc)

    # Synchronisiere die Metriken zwischen allen Prozessen
    metric_logger.synchronize_between_processes()

    # Logge die Durchschnittswerte der Genauigkeit und des Verlusts
    log_(' * [Acc@1 %.3f] [LossOut %.3f]' %
         (metric_logger.acc.global_avg, metric_logger.loss.global_avg))

    if logger is not None:
        logger.scalar_summary('eval/acc', metric_logger.acc.global_avg, steps)
        logger.scalar_summary('eval/loss_test', metric_logger.loss.global_avg, steps)

    # Setze das Modell zurück in den ursprünglichen Modus (Training/Evaluierung)
    model.train(mode)

    return metric_logger.acc.global_avg
