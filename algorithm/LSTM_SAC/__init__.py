"""LSTM-SAC representation registration for XuanCe."""

from xuance.torch.representations import REGISTRY_Representation

from .representation import LSTM_HIDDEN, HistoryLSTM

REGISTRY_Representation['History_LSTM'] = HistoryLSTM

__all__ = ['LSTM_HIDDEN', 'HistoryLSTM']
