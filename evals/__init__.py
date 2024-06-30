def setup(mode, P):
    """
    Stellt die Funktion zum Testen des Klassifikators basierend auf dem angegebenen Modus bereit.

    Args:
        mode (str): Der Modus, in dem das Modell ausgeführt wird (z.B. 'protonet').
        P: Parameter für das Setup.

    Returns:
        function: Die Funktion zum Testen des Klassifikators.
    """
    if mode == 'protonet':
        from evals.metric_based.protonet import test_classifier as test_func  # Importiere die Testfunktion für den Protonet-Modus

    return test_func  # Gib die Testfunktion zurück


def accuracy(output, target, topk=(1,)):
    """
    Berechnet die Genauigkeit für die k besten Vorhersagen für die angegebenen Werte von k.

    Args:
        output (torch.Tensor): Die Ausgaben des Modells.
        target (torch.Tensor): Die Zielwerte.
        topk (tuple): Die Werte von k, für die die Genauigkeit berechnet werden soll.

    Returns:
        list: Eine Liste der Genauigkeitswerte für die angegebenen Werte von k.
    """
    maxk = min(max(topk), output.size()[1])  # Bestimme das maximale k, beschränkt auf die Anzahl der Klassen
    batch_size = target.size(0)  # Bestimme die Batch-Größe

    _, pred = output.topk(maxk, 1, True, True)  # Bestimme die k besten Vorhersagen
    pred = pred.t()  # Transponiere die Vorhersagen

    correct = pred.eq(target.reshape(1, -1).expand_as(pred))  # Vergleiche die Vorhersagen mit den Zielwerten

    # Berechne die Genauigkeit für jedes k in topk
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
