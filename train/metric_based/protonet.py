import time

import torch
import torch.nn.functional as F
import numpy as np

from torchmeta.utils.prototype import get_prototypes
from train.metric_based import get_accuracy

# Bestimme das Gerät für die Berechnungen (GPU oder CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check(P):
    """
    Überprüft bestimmte Parameter und gibt zurück, ob der Dateiname das heutige Datum enthalten soll.

    Args:
        P: Parameter für das Setup.

    Returns:
        bool: True, wenn der Dateiname das heutige Datum enthalten soll.
    """
    filename_with_today_date = True
    assert P.num_shots_global == 0  # Überprüft, dass num_shots_global gleich 0 ist
    return filename_with_today_date

def protonet_step(P, step, model, criterion, optimizer, batch, metric_logger, logger):
    """
    Führt einen Trainingsschritt für das Prototypen-Netzwerk (ProtoNet) durch.

    Args:
        P: Parameter für das Setup.
        step: Aktueller Trainingsschritt.
        model: Das zu trainierende Modell.
        criterion: Verlustfunktion.
        optimizer: Optimizer für die Modellparameter.
        batch: Batch-Daten für das Training.
        metric_logger: Logger für die Metriken.
        logger: Logger für die Ausgabe.
    """
    stime = time.time()  # Startzeit des Trainingsschritts
    model.train()  # Setze das Modell in den Trainingsmodus

    assert not P.regression  # Stelle sicher, dass es sich nicht um eine Regression handelt

    # Extrahiere Trainings- und Testdaten aus dem Batch
    train_inputs, train_targets = batch['train']
    num_ways = len(set(list(train_targets[0].numpy())))  # Bestimme die Anzahl der Klassen
    test_inputs, test_targets = batch['test']

    # Verschiebe die Trainingsdaten auf das entsprechende Gerät (CPU oder GPU)
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    train_embeddings = model(train_inputs)  # Berechne die Embeddings der Trainingsdaten

    # Verschiebe die Testdaten auf das entsprechende Gerät (CPU oder GPU)
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)
    test_embeddings = model(test_inputs)  # Berechne die Embeddings der Testdaten

    # Berechne die Prototypen der Trainingsembeddings
    prototypes = get_prototypes(train_embeddings, train_targets, num_ways)

    # Berechne die quadratischen Distanzen zwischen den Prototypen und den Testeinbettungen
    squared_distances = torch.sum((prototypes.unsqueeze(2)
                                   - test_embeddings.unsqueeze(1)) ** 2, dim=-1)
    loss = criterion(-squared_distances, test_targets)  # Berechne den Verlust

    """ Outer Gradient Step """
    optimizer.zero_grad()  # Setze die Gradienten auf null
    loss.backward()  # Berechne die Gradienten
    optimizer.step()  # Aktualisiere die Modellparameter

    # Berechne die Genauigkeit
    acc = get_accuracy(prototypes, test_embeddings, test_targets).item()

    """ Track Statistiken """
    metric_logger.meters['batch_time'].update(time.time() - stime)  # Aktualisiere die Batch-Zeit
    metric_logger.meters['meta_test_cls'].update(loss.item())  # Aktualisiere den Meta-Test-Verlust
    metric_logger.meters['train_acc'].update(acc)  # Aktualisiere die Trainingsgenauigkeit

    # Logge die Statistiken alle P.print_step Schritte
    if step % P.print_step == 0:
        logger.log_dirname(f"Step {step}")
        logger.scalar_summary('train/meta_test_cls',
                              metric_logger.meta_test_cls.global_avg, step)
        logger.scalar_summary('train/train_acc',
                              metric_logger.train_acc.global_avg, step)
        logger.scalar_summary('train/batch_time',
                              metric_logger.batch_time.global_avg, step)

        logger.log('[TRAIN] [Step %3d] [Time %.3f] [Data %.3f] '
                   '[MetaTestLoss %f]' %
                   (step, metric_logger.batch_time.global_avg, metric_logger.data_time.global_avg,
                    metric_logger.meta_test_cls.global_avg))
