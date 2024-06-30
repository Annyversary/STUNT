from re import I
import numpy as np
import torch
import os
import copy
import faiss

class Income(object):
    def __init__(self, tabular_size, seed, source, shot, tasks_per_batch, test_num_way, query):
        super().__init__()
        self.num_classes = 2  # Anzahl der Klassen
        self.tabular_size = tabular_size  # Größe der Tabellendaten
        self.source = source  # Quelle der Daten (train oder val)
        self.shot = shot  # Anzahl der Support-Shots
        self.query = query  # Anzahl der Query-Shots
        self.tasks_per_batch = tasks_per_batch  # Anzahl der Aufgaben pro Batch

        # Lade die unlabeled Trainingsdaten
        self.unlabeled_x = np.load('./data/income/train_x.npy')
        # Lade die Testdaten und Labels
        self.test_x = np.load('./data/income/xtest.npy')
        self.test_y = np.load('./data/income/ytest.npy')
        # Lade die Validierungsdaten und pseudo-Labels
        self.val_x = np.load('./data/income/val_x.npy')
        self.val_y = np.load('./data/income/val_y.npy')  # pseudo-Labels für Validierung
        self.test_num_way = test_num_way  # Anzahl der Klassen im Test
        self.test_rng = np.random.RandomState(seed)  # Zufallsgenerator für Test
        self.val_rng = np.random.RandomState(seed)  # Zufallsgenerator für Validierung

    def __next__(self):
        # Ermöglicht die Nutzung des Objekts als Iterator
        return self.get_batch()

    def __iter__(self):
        # Gibt sich selbst als Iterator zurück
        return self

    def get_batch(self):
        # Initialisiere Listen für Support- und Query-Daten
        xs, ys, xq, yq = [], [], [], []

        if self.source == 'train':
            # Verwende die unlabeled Trainingsdaten
            x = self.unlabeled_x
            num_way = self.test_num_way

        elif self.source == 'val':
            # Verwende die Validierungsdaten und Labels
            x = self.val_x
            y = self.val_y
            # Bestimme die Liste der Klassen
            class_list, _ = np.unique(y, return_counts=True)
            num_val_shot = 1  # Anzahl der Shots für Validierung
            num_way = 2  # Anzahl der Klassen für Validierung

        for _ in range(self.tasks_per_batch):
            # Initialisiere Support- und Query-Sets
            support_set = []
            query_set = []
            support_sety = []
            query_sety = []

            if self.source == 'val':
                # Wähle zufällig Klassen für die Validierung
                classes = np.random.choice(class_list, num_way, replace=False)
                support_idx = []
                query_idx = []

                for k in classes:
                    # Indexe der Datenpunkte für die Klasse k
                    k_idx = np.where(y == k)[0]
                    # Permutiere die Indexe zufällig
                    permutation = np.random.permutation(len(k_idx))
                    k_idx = k_idx[permutation]
                    # Wähle Support- und Query-Indexe
                    support_idx.append(k_idx[:num_val_shot])
                    query_idx.append(k_idx[num_val_shot:num_val_shot + 30])

                # Kombiniere die Indexe
                support_idx = np.concatenate(support_idx)
                query_idx = np.concatenate(query_idx)

                # Erstelle Support- und Query-Datensätze
                support_x = x[support_idx]
                query_x = x[query_idx]
                s_y = y[support_idx]
                q_y = y[query_idx]
                support_y = copy.deepcopy(s_y)
                query_y = copy.deepcopy(q_y)

                # Mappe die Labels auf neue Werte
                i = 0
                for k in classes:
                    support_y[s_y == k] = i
                    query_y[q_y == k] = i
                    i += 1

                # Füge die Datensätze zu den Sets hinzu
                support_set.append(support_x)
                support_sety.append(support_y)
                query_set.append(query_x)
                query_sety.append(query_y)

            elif self.source == 'train':
                tmp_x = copy.deepcopy(x)
                min_count = 0

                while min_count < (self.shot + self.query):
                    # Bestimme die Anzahl der zu maskierenden Spalten
                    min_col = int(x.shape[1] * 0.2)
                    max_col = int(x.shape[1] * 0.5)
                    col = np.random.choice(range(min_col, max_col), 1, replace=False)[0]
                    # Wähle zufällig die Spalten aus, die maskiert werden sollen
                    task_idx = np.random.choice([i for i in range(x.shape[1])], col, replace=False)
                    masked_x = np.ascontiguousarray(x[:, task_idx], dtype=np.float32)
                    # Führe k-means Clustering auf den maskierten Daten durch
                    kmeans = faiss.Kmeans(masked_x.shape[1], num_way, niter=20, nredo=1, verbose=False, min_points_per_centroid=self.shot + self.query, gpu=1)
                    kmeans.train(masked_x)
                    D, I = kmeans.index.search(masked_x, 1)
                    y = I[:, 0].astype(np.int32)
                    class_list, counts = np.unique(y, return_counts=True)
                    min_count = min(counts)

                # Permutiere einige der maskierten Spalten zufällig
                num_to_permute = x.shape[0]
                for t_idx in task_idx:
                    rand_perm = np.random.permutation(num_to_permute)
                    tmp_x[:, t_idx] = tmp_x[:, t_idx][rand_perm]

                # Wähle zufällig Klassen für das Training
                classes = np.random.choice(class_list, num_way, replace=False)
                support_idx = []
                query_idx = []

                for k in classes:
                    k_idx = np.where(y == k)[0]
                    permutation = np.random.permutation(len(k_idx))
                    k_idx = k_idx[permutation]
                    support_idx.append(k_idx[:self.shot])
                    query_idx.append(k_idx[self.shot:self.shot + self.query])

                support_idx = np.concatenate(support_idx)
                query_idx = np.concatenate(query_idx)

                support_x = tmp_x[support_idx]
                query_x = tmp_x[query_idx]
                s_y = y[support_idx]
                q_y = y[query_idx]
                support_y = copy.deepcopy(s_y)
                query_y = copy.deepcopy(q_y)

                i = 0
                for k in classes:
                    support_y[s_y == k] = i
                    query_y[q_y == k] = i
                    i += 1

                support_set.append(support_x)
                support_sety.append(support_y)
                query_set.append(query_x)
                query_sety.append(query_y)

            xs_k = np.concatenate(support_set, 0)
            xq_k = np.concatenate(query_set, 0)
            ys_k = np.concatenate(support_sety, 0)
            yq_k = np.concatenate(query_sety, 0)

            xs.append(xs_k)
            xq.append(xq_k)
            ys.append(ys_k)
            yq.append(yq_k)

        # Staple die Datensätze und forme sie um
        xs, ys = np.stack(xs, 0), np.stack(ys, 0)
        xq, yq = np.stack(xq, 0), np.stack(yq, 0)

        if self.source == 'val':
            xs = np.reshape(xs, [self.tasks_per_batch, num_way * num_val_shot, self.tabular_size])
        else:
            xs = np.reshape(xs, [self.tasks_per_batch, num_way * self.shot, self.tabular_size])

        if self.source == 'val':
            xq = np.reshape(xq, [self.tasks_per_batch, num_way * 30, self.tabular_size])
        else:
            xq = np.reshape(xq, [self.tasks_per_batch, num_way * self.query, self.tabular_size])

        # Konvertiere die Daten in Float32-Typen
        xs = xs.astype(np.float32)
        xq = xq.astype(np.float32)
        ys = ys.astype(np.float32)
        yq = yq.astype(np.float32)

        # Konvertiere die Daten in Torch-Tensoren
        xs = torch.from_numpy(xs).type(torch.FloatTensor)
        xq = torch.from_numpy(xq).type(torch.FloatTensor)
        ys = torch.from_numpy(ys).type(torch.LongTensor)
        yq = torch.from_numpy(yq).type(torch.LongTensor)

        # Erstelle ein Batch-Dictionary mit Trainings- und Test-Sets
        batch = {'train': (xs, ys), 'test': (xq, yq)}

        return batch
