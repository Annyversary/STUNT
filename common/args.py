from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for train."""

    # Initialisiere den ArgumentParser mit einer Beschreibung des Programms
    parser = ArgumentParser(
        description='PyTorch implementation of Self-generated Tasks from UNlabeled Tables (STUNT)'
    )

    # Hinzufügen von Argumenten zum Parser

    # Argument für den Datensatz
    parser.add_argument('--dataset', help='Dataset',
                        type=str)

    # Argument für den Trainingsmodus, standardmäßig 'protonet'
    parser.add_argument('--mode', help='Training mode',
                        default='protonet', type=str)

    # Argument für den Zufallsgenerator-Seed zur Reproduzierbarkeit
    parser.add_argument("--seed", type=int,
                        default=0, help='random seed')

    # Argument für den lokalen Rang für verteiltes Lernen
    parser.add_argument("--rank", type=int,
                        default=0, help='Local rank for distributed learning')

    # Argument, das automatisch auf True gesetzt wird, wenn GPUs > 1 vorhanden sind
    parser.add_argument('--distributed', help='automatically change to True for GPUs > 1',
                        default=False, type=bool)

    # Argument für den Pfad zum Fortsetzen des Trainings von einem Checkpoint
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)

    # Argument für den Pfad zum Laden eines Checkpoints
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default=None, type=str)

    # Argument, um state_dicts nicht strikt zu laden
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')

    # Argument für das Suffix des Log-Verzeichnisses
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)

    # Argument für die Anzahl der Epochen-Schritte zur Berechnung der Genauigkeit/Fehler
    parser.add_argument('--eval_step', help='Epoch steps to compute accuracy/error',
                        default=50, type=int)

    # Argument für die Anzahl der Epochen-Schritte zum Speichern eines Checkpoints
    parser.add_argument('--save_step', help='Epoch steps to save checkpoint',
                        default=2500, type=int)

    # Argument für die Anzahl der Epochen-Schritte zum Drucken/Verfolgen der Trainingsstatistik
    parser.add_argument('--print_step', help='Epoch steps to print/track training stat',
                        default=50, type=int)

    # Argument zur Verwendung des MSE-Loss für Regressionstasks
    parser.add_argument("--regression", help='Use MSE loss (automatically turns to true for regression tasks)',
                        action='store_true')

    # Argument, das das Speichern der Daten verhindert
    parser.add_argument("--baseline", help='do not save the date',
                        action='store_true')

    # Training Konfigurationen
    # Argument für die Anzahl der Meta-Learning-Außen-Schritte
    parser.add_argument('--outer_steps', help='meta-learning outer-step',
                        default=2500, type=int)

    # Argument für die Lernrate
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')

    # Argument für die Batch-Größe
    parser.add_argument('--batch_size', help='Batch size',
                        default=4, type=int)

    # Argument für die Batch-Größe des Testloaders
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=4, type=int)

    # Argument für die maximale Anzahl von Aufgaben für die Inferenz
    parser.add_argument('--max_test_task', help='Max number of task for inference',
                        default=1000, type=int)

    # Meta Learning Konfigurationen
    # Argument für die Anzahl der Klassen (N Wege)
    parser.add_argument('--num_ways', help='N ways',
                        default=10, type=int)

    # Argument für die Anzahl der Support-Shots (K Shots)
    parser.add_argument('--num_shots', help='K (support) shot',
                        default=1, type=int)

    # Argument für die Anzahl der Query-Shots
    parser.add_argument('--num_shots_test', help='query shot',
                        default=15, type=int)

    # Argument für die Anzahl der globalen (oder distill) Shots
    parser.add_argument('--num_shots_global', help='global (or distill) shot',
                        default=0, type=int)

    # Classifier Konfigurationen
    # Argument für den Modelltyp, standardmäßig 'mlp' (Multi-Layer Perceptron)
    parser.add_argument('--model', help='model type',
                        type=str, default='mlp')

    # Überprüfen, ob Standardwerte verwendet werden sollen
    if default:
        return parser.parse_args('')  # leere Zeichenkette
    else:
        return parser.parse_args()
