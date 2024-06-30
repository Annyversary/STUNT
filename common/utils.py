import os

import torch
import torch.optim as optim

from utils import load_checkpoint

# Bestimme das Gerät für die Berechnungen (GPU oder CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_optimizer(P, model):
    """Erstellt und gibt einen Adam-Optimizer zurück."""
    params = model.parameters()
    optimizer = optim.Adam(params, lr=P.lr)
    return optimizer

def is_resume(P, model, optimizer):
    """Überprüft, ob das Training von einem Checkpoint fortgesetzt werden soll und lädt diesen falls vorhanden."""
    if P.resume_path is not None:
        # Lade den letzten Checkpoint
        model_state, optim_state, config, lr_dict, ema_dict = load_checkpoint(P.resume_path, mode='last')
        # Lade den Modellzustand
        model.load_state_dict(model_state, strict=not P.no_strict)
        # Lade den Optimizer-Zustand
        optimizer.load_state_dict(optim_state)
        # Setze die Trainingsparameter
        start_step = config['step']
        best = config['best']
        is_best = False
        acc = 0.0
        # Falls vorhanden, setze die Lernrate und den gleitenden Durchschnitt
        if lr_dict is not None:
            P.inner_lr = lr_dict
        if ema_dict is not None:
            P.moving_average = ema_dict
    else:
        # Initialisiere die Parameter, falls kein Checkpoint vorhanden ist
        is_best = False
        start_step = 1
        best = -100.0
        acc = 0.0
    return is_best, start_step, best, acc

def load_model(P, model, logger=None):
    """Lädt das Modell von einem angegebenen Pfad."""
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if P.load_path is not None:
        log_(f'Load model from {P.load_path}')
        # Lade das Modell von der angegebenen Datei
        checkpoint = torch.load(P.load_path)
        # Falls verteiltes Training, initialisiere das Modell für den entsprechenden Rang
        if P.rank != 0:
            model.__init_low_rank__(rank=P.rank)
        # Lade den Zustand des Modells
        model.load_state_dict(checkpoint, strict=P.no_strict)
