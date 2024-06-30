import torch
from torchmeta.utils.prototype import get_prototypes
from train.metric_based import get_accuracy
from utils import MetricLogger
import numpy as np
import argparse
import torch.nn as nn

# Argumentparser für Kommandozeilenargumente
parser = argparse.ArgumentParser(description='STUNT')
parser.add_argument('--data_name', default='income', type=str)  # Name des Datensatzes
parser.add_argument('--shot_num', default=1, type=int)  # Anzahl der Shots für Few-Shot Learning
parser.add_argument('--load_path', default='', type=str)  # Pfad zum Laden des Modells
parser.add_argument('--seed', default=0, type=int)  # Seed für die Reproduzierbarkeit
args = parser.parse_args()

# Datenspezifische Konfigurationen
if args.data_name == 'income':
    input_size = 105  # Anzahl der Eingabefeatures
    output_size = 2  # Anzahl der Ausgabeklassen
    hidden_dim = 1024  # Dimension der versteckten Schicht

# Definition des Modells
class MLPProto(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes, drop_p=0.):
        super(MLPProto, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes
        self.drop_p = drop_p

        # Sequenzielles Modell bestehend aus zwei vollständig verbundenen Schichten und ReLU-Aktivierung
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_sizes, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_sizes, hidden_sizes, bias=True)
        )

    def forward(self, inputs):
        # Forward-Pass durch den Encoder
        embeddings = self.encoder(inputs)
        return embeddings

# Initialisiere das Modell und lade die Modellparameter
model = MLPProto(input_size, hidden_dim, hidden_dim)
model.load_state_dict(torch.load(args.load_path))

# Lade die Trainings- und Testdaten
train_x = np.load('./data/' + args.data_name + '/xtrain.npy')
train_y = np.load('./data/' + args.data_name + '/ytrain.npy')
test_x = np.load('./data/' + args.data_name + '/xtest.npy')
test_y = np.load('./data/' + args.data_name + '/ytest.npy')
train_idx = np.load('./data/' + args.data_name + '/index{}/train_idx_{}.npy'.format(args.shot_num, args.seed))

# Berechne die Einbettungen für den Trainings- und Testdatensatz
few_train = model(torch.tensor(train_x[train_idx]).float())
support_x = few_train.detach().numpy()
support_y = train_y[train_idx]
few_test = model(torch.tensor(test_x).float())
query_x = few_test.detach().numpy()
query_y = test_y

# Definiere die Genauigkeitsfunktion
def get_accuracy(prototypes, embeddings, targets):
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float()) * 100.

# Konvertiere die Daten in PyTorch Tensoren
train_x = torch.tensor(support_x.astype(np.float32)).unsqueeze(0)
train_y = torch.tensor(support_y.astype(np.int64)).unsqueeze(0).type(torch.LongTensor)
val_x = torch.tensor(query_x.astype(np.float32)).unsqueeze(0)
val_y = torch.tensor(query_y.astype(np.int64)).unsqueeze(0).type(torch.LongTensor)

# Berechne die Prototypen und die Genauigkeit
prototypes = get_prototypes(train_x, train_y, output_size)
acc = get_accuracy(prototypes, val_x, val_y).item()

# Drucke und speichere die Ergebnisse
print(args.seed, acc)

out_file = 'result/{}_{}shot/test'.format(args.data_name, args.shot_num)
with open(out_file, 'a+') as f:
    f.write('seed: ' + str(args.seed) + ' test: ' + str(acc))
    f.write('\n')
