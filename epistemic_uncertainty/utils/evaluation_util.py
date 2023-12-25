import numpy as np
from torch.utils.data import DataLoader
from ..model.dc.deep_clustering import DCModel


def get_clustering_by_label(dc_model: DCModel, train_data_loader: DataLoader, label, dev='cuda') -> np.ndarray:
    """
    Calculates the number of assignee
    :param dc_model:
    :param train_data_loader:
    :param label: real/fake
    :param dev:
    :return: List of the clusters and the population assigned to them
    """
    clusters_assignments = []
    for itr, data_array in enumerate(train_data_loader):
        data, actions, labels, _ = data_array
        labels = np.array(labels)
        indices = np.argwhere(labels == label)
        data = data[indices, ...].squeeze().to(dev)
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        if len(indices) > 0:
            pred = list(dc_model.predict(data).cpu().detach().numpy())
            clusters_assignments = clusters_assignments.__add__(pred)
    clusters_assignments = np.array(clusters_assignments)
    return clusters_assignments
