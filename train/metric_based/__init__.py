import torch


def get_accuracy(prototypes, embeddings, targets):
    """
    Berechnet die Genauigkeit des Prototypen-Netzwerks auf den Test- oder Abfragepunkten.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        Ein Tensor, der die Prototypen für jede Klasse enthält. Dieser Tensor hat die Form
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        Ein Tensor, der die Einbettungen der Abfragepunkte enthält. Dieser Tensor hat die Form
        `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        Ein Tensor, der die Zielwerte der Abfragepunkte enthält. Dieser Tensor hat die Form
        `(meta_batch_size, num_examples)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Durchschnittliche Genauigkeit auf den Abfragepunkten.
    """
    # Berechne die quadratischen Distanzen zwischen den Prototypen und den Einbettungen der Abfragepunkte
    sq_distances = torch.sum((prototypes.unsqueeze(1)
                              - embeddings.unsqueeze(2)) ** 2, dim=-1)

    # Bestimme die Klasse mit der kleinsten quadratischen Distanz (nächstgelegener Prototyp)
    _, predictions = torch.min(sq_distances, dim=-1)

    # Berechne die Genauigkeit als den Prozentsatz der korrekt vorhergesagten Klassen
    return torch.mean(predictions.eq(targets).float()) * 100.
