from collections import OrderedDict

import torch
import torch.nn.functional as F

# Bestimme das Gerät für die Berechnungen (GPU oder CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup(mode, P):
    """
    Stellt die Trainingsfunktion, den Dateinamen und das heutige Datum basierend auf dem Modus und den Parametern bereit.

    Args:
        mode (str): Der Modus, in dem das Modell ausgeführt wird (z.B. 'protonet').
        P: Parameter für das Setup.

    Returns:
        train_func (function): Die Funktion, die den Trainingsschritt ausführt.
        fname (str): Der generierte Dateiname basierend auf den Parametern.
        today (bool): Gibt an, ob das heutige Datum in den Dateinamen aufgenommen werden soll.
    """
    # Generiere den Dateinamen basierend auf den Parametern
    fname = f'{P.dataset}_{P.model}_{mode}_{P.num_ways}way_{P.num_shots}shot_{P.num_shots_test}query'

    if mode == 'protonet':
        assert not P.regression  # Überprüfe, dass es sich nicht um eine Regression handelt

        # Importiere die Trainingsfunktion und die Check-Funktion für den 'protonet'-Modus
        from train.metric_based.protonet import protonet_step as train_func
        from train.metric_based.protonet import check
    else:
        # Werfe einen Fehler, wenn der Modus nicht implementiert ist
        raise NotImplementedError()

    # Überprüfe das heutige Datum mithilfe der Check-Funktion
    today = check(P)
    if P.baseline:
        today = False

    # Füge einen Suffix zum Dateinamen hinzu, wenn angegeben
    if P.suffix is not None:
        fname += f'_{P.suffix}'

    return train_func, fname, today

def copy_model_param(model, params=None):
    """
    Kopiert die Parameter des Modells und erstellt eine neue geordnete Liste von Parametern.

    Args:
        model: Das Modell, dessen Parameter kopiert werden sollen.
        params (OrderedDict, optional): Die Parameter, die kopiert werden sollen. Wenn None, werden die Parameter des Modells verwendet.

    Returns:
        OrderedDict: Eine geordnete Liste der kopierten Parameter.
    """
    if params is None:
        params = OrderedDict(model.meta_named_parameters())  # Hole die Parameter des Modells
    copy_params = OrderedDict()  # Initialisiere ein neues OrderedDict für die kopierten Parameter

    for (name, param) in params.items():
        # Klone und separiere jeden Parameter, damit er unabhängig ist
        copy_params[name] = param.clone().detach()
        copy_params[name].requires_grad_()

    return copy_params

def dropout_eval(m):
    """
    Setzt alle Dropout-Schichten im Modell in den Evaluierungsmodus.

    Args:
        m: Das Modell oder das Modul, das evaluiert werden soll.
    """
    def _is_leaf(model):
        # Überprüft, ob das Modell ein Blattknoten ist (keine Kinder hat)
        return len(list(model.children())) == 0

    if hasattr(m, 'dropout'):
        m.dropout.eval()  # Setzt die Dropout-Schicht in den Evaluierungsmodus

    for child in m.children():
        if not _is_leaf(child):
            # Rekursive Anwendung der Funktion auf alle Kindermodelle
            dropout_eval(child)
