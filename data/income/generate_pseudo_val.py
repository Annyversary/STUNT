import numpy as np
from sklearn.cluster import KMeans

# Setze den Zufallsgenerator-Seed für Reproduzierbarkeit
np.random.seed(0)

# Lade die Trainingsdaten und Labels
x = np.load('xtrain.npy')
y = np.load('ytrain.npy')

# Bestimme die Anzahl der Trainingsbeispiele (80% der Daten)
num_train = int(len(x) * 0.8)

# Erstelle eine zufällige Permutation der Indizes der Daten
idx = np.random.permutation(len(x))

# Teile die Indizes in Trainings- und Validierungsindizes auf
train_idx = idx[:num_train]
val_idx = idx[num_train:]

# Teile die Daten basierend auf den Indizes in Trainings- und Validierungsdaten auf
train_x = x[train_idx]
val_x = x[val_idx]

# Speichere die Trainings- und Validierungsdaten
np.save('train_x.npy', train_x)
np.save('val_x.npy', val_x)

# Erstelle ein KMeans-Modell mit 2 Clustern
model = KMeans(n_clusters=2)

# Fitte das Modell auf den Validierungsdaten
model.fit(val_x)

# Sage die Cluster-Labels der Validierungsdaten voraus
labels = model.predict(val_x)

# Speichere die vorhergesagten Cluster-Labels als pseudo-Labels für die Validierungsdaten
np.save('pseudo_val_y.npy', labels)
