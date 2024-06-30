from models.protonet_model.mlp import MLPProto


def get_model(P, modelstr):
    """
    Stellt das Modell basierend auf dem angegebenen Modelltyp und den Parametern bereit.

    Args:
        P: Parameter für das Setup.
        modelstr (str): Der Typ des Modells, das erstellt werden soll.

    Returns:
        nn.Module: Das erstellte Modell.
    """

    # Überprüfe, ob der Modelltyp 'mlp' ist
    if modelstr == 'mlp':
        # Überprüfe, ob 'protonet' im Modus enthalten ist
        if 'protonet' in P.mode:
            # Überprüfe, ob der Datensatz 'income' ist
            if P.dataset == 'income':
                # Erstelle ein MLPProto-Modell mit den angegebenen Eingabe-, Hidden- und Ausgabedimensionen
                model = MLPProto(105, 1024, 1024)
    else:
        # Werfe einen Fehler, wenn der Modelltyp nicht implementiert ist
        raise NotImplementedError()

    return model  # Gib das erstellte Modell zurück
