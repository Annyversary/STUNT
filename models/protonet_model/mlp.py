# importiert das torch.nn Modul, das in PyTorch Funktionen
# und Klassen für die Definition von neuronalen Netzwerken bereitstellt.
import torch.nn as nn

# Definiert eine neue Klasse MLPProto, die von nn.Module erbt.
# nn.Module ist die Basisklasse für alle neuronalen Netzwerkmodule in PyTorch.
class MLPProto(nn.Module):

    # Konstruktor (__init__-Methode) der Klasse.
    # in_features (Anzahl der Eingabefeatures), out_features (Anzahl der Ausgabefeatures),
    # hidden_sizes (Anzahl der Neuronen in den versteckten Schichten) und drop_p (Dropout-Rate, standardmäßig 0).
    def __init__(self, in_features, out_features, hidden_sizes, drop_p=0.):

        # super(MLPProto, self).__init__() ruft den Konstruktor der Basisklasse nn.Module auf und initialisiert ihn.
        super(MLPProto, self).__init__()

        # Speichern der Parameter als Instanzvariablen
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.drop_p = drop_p

        # Hier wird ein sequenzielles Modell (nn.Sequential) definiert und self.encoder zugewiesen.
        self.encoder = nn.Sequential(

            # Erste vollständig verbundene (fully connected) Schicht mit in_features Eingabeneuronen
            # und hidden_sizes Ausgabeneuronen. Die Schicht hat auch einen Bias-Term.
            nn.Linear(in_features, hidden_sizes, bias=True),

            # Fügt eine ReLU-Aktivierungsfunktion (Rectified Linear Unit) hinzu,
            # die auf die Ausgaben der vorhergehenden Schicht angewendet wird.
            nn.ReLU(),

            # Zweite vollständig verbundene Schicht mit hidden_sizes Eingabe- und Ausgabeneuronen.
            nn.Linear(hidden_sizes, hidden_sizes, bias=True)
        )

    # Definition der Forward-Methode für den Forward Pass des Netzwerks
    # Diese Methode beschreibt, wie die Eingaben durch die Netzwerk-Schichten fließen.
    def forward(self, inputs):
        # Umformen der Eingaben und Durchlaufen des Encoders
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        # Umformen der Ausgaben in die gewünschte Form und Rückgabe
        return embeddings.view(*inputs.shape[:2], -1)
