import time
from collections import OrderedDict

import torch
import torch.nn as nn

from common.utils import is_resume
from utils import MetricLogger, save_checkpoint, save_checkpoint_step

# Bestimme das Gerät für die Berechnungen (GPU oder CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def meta_trainer(P, train_func, test_func, model, optimizer, train_loader, test_loader, logger):
    """
    Führt das Meta-Training durch, einschließlich Training, Evaluierung und Modell-Checkpointing.

    Args:
        P: Parameter für das Setup.
        train_func: Funktion, die den Trainingsschritt ausführt.
        test_func: Funktion, die den Testschritt ausführt.
        model: Das zu trainierende Modell.
        optimizer: Optimizer für die Modellparameter.
        train_loader: DataLoader für die Trainingsdaten.
        test_loader: DataLoader für die Testdaten.
        logger: Logger zum Protokollieren.
    """
    kwargs = {}  # Zusätzliche Argumente für die Trainingsfunktion
    kwargs_test = {}  # Zusätzliche Argumente für die Testfunktion

    # Initialisiere einen MetricLogger für das Protokollieren der Metriken
    metric_logger = MetricLogger(delimiter="  ")

    """ resume option """
    # Überprüfe, ob das Training von einem Checkpoint fortgesetzt werden soll
    is_best, start_step, best, acc = is_resume(P, model, optimizer)

    """ define loss function """
    # Definiere die Verlustfunktion (Cross-Entropy-Loss)
    criterion = nn.CrossEntropyLoss()

    """ training start """
    logger.log_dirname("Start training")
    for step in range(start_step, P.outer_steps + 1):
        stime = time.time()
        train_batch = next(train_loader)  # Hole den nächsten Trainingsbatch
        metric_logger.meters['data_time'].update(time.time() - stime)  # Aktualisiere die Datenladezeit

        # Führe den Trainingsschritt aus
        train_func(P, step, model, criterion, optimizer, train_batch,
                   metric_logger=metric_logger, logger=logger, **kwargs)

        """ evaluation & save the best model """
        if step % P.eval_step == 0:
            # Führe den Testschritt aus und erhalte die Genauigkeit
            acc = test_func(P, model, test_loader, criterion, step, logger=logger, **kwargs_test)

            # Speichere das Modell, wenn die aktuelle Genauigkeit die beste ist
            if best < acc:
                best = acc
                save_checkpoint(P, step, best, model.state_dict(),
                                optimizer.state_dict(), logger.logdir, is_best=True)

            # Logge die beste Genauigkeit
            logger.scalar_summary('eval/best_acc', best, step)
            logger.log('[EVAL] [Step %3d] [Acc %5.2f] [Best %5.2f]' % (step, acc, best))

        """ save model per save_step steps """
        if step % P.save_step == 0:
            # Speichere das Modell alle P.save_step Schritte
            save_checkpoint_step(P, step, best, model.state_dict(),
                                 optimizer.state_dict(), logger.logdir)

    """ save last model """
    # Speichere das letzte Modell nach Abschluss des Trainings
    save_checkpoint(P, P.outer_steps, best, model.state_dict(),
                    optimizer.state_dict(), logger.logdir)
