import torch
import logging
import numpy as np
import sklearn.cluster
from . import evaluation
from . import faissext
from . import data
from . import utils

def assign_cluster_labels(model, data_loader, use_penultimate_layer, num_clusters, gpu_id=None, clustering_backend='faiss'):
    """
    Assigns cluster labels to the dataset based on the model's output embeddings.
    """
    is_single_cluster = (num_clusters == 1)

    if not is_single_cluster:
        if not use_penultimate_layer:
            logging.debug('Using the final layer for clustering')

        embeddings, labels, indices = utils.predict_batchwise(
            model=model,
            dataloader=data_loader,
            use_penultimate=use_penultimate_layer,
            is_dry_run=is_single_cluster
        )

        sorted_indices = np.argsort(indices)
        embeddings = embeddings[sorted_indices]
        indices = indices[sorted_indices]
        labels = labels[sorted_indices]

        if clustering_backend == 'torch+sklearn':
            clustering_algorithm = sklearn.cluster.KMeans(n_clusters=num_clusters)
            cluster_assignments = clustering_algorithm.fit(embeddings).labels_
        else:
            cluster_assignments = faissext.do_clustering(
                embeddings,
                num_clusters=num_clusters,
                gpu_ids=None if clustering_backend != 'faiss-gpu' else torch.cuda.current_device(),
                niter=100,
                nredo=5,
                verbose=0
            )
    else:
        labels = np.array(data_loader.dataset.ys)
        indices = np.array(data_loader.dataset.I)
        cluster_assignments = np.zeros(len(labels), dtype=int)

    return cluster_assignments, labels, indices

def create_clustered_dataloaders(model, initial_dataloader, config, reassign_clusters=False, previous_indices=None, previous_assignments=None, log=None):
    """
    Creates dataloaders for each cluster by assigning samples based on cluster labels.
    """
    def sort_indices(indices):
        return torch.sort(torch.LongTensor(indices))[1]

    current_assignments, current_labels, current_indices = assign_cluster_labels(
        model,
        initial_dataloader,
        use_penultimate_layer=True,
        num_clusters=config['num_clusters'],
        clustering_backend=config['backend']
    )

    if reassign_clusters:
        sorted_current_indices = sort_indices(current_indices)
        current_indices = current_indices[sorted_current_indices]
        current_labels = current_labels[sorted_current_indices]
        current_assignments = current_assignments[sorted_current_indices]

        sorted_previous_indices = sort_indices(previous_indices)
        previous_indices = previous_indices[sorted_previous_indices]
        previous_assignments = previous_assignments[sorted_previous_indices]

        log.debug('Reassigning clusters...')
        log.debug('Calculating NMI for consecutive cluster assignments...')
        log.debug(str(
            evaluation.calc_normalized_mutual_information(
                current_assignments[current_indices],
                previous_assignments[previous_indices]
            )
        ))

        current_assignments, reassignment_costs = data.loader.reassign_clusters(
            previous_assignments=previous_assignments,
            current_assignments=current_assignments,
            previous_indices=previous_indices,
            current_indices=current_indices
        )
        log.debug('Reassignment costs:')
        log.debug(str(reassignment_costs))

    # Remove clusters with fewer than 2 samples per class
    for cluster_id in range(config['num_clusters']):
        for class_label in np.unique(current_labels[current_assignments == cluster_id]):
            if (current_labels[current_assignments == cluster_id] == class_label).sum().item() == 1:
                current_assignments[(current_labels == class_label) & (current_assignments == cluster_id)] = -1

    dataloaders = data.loader.make_from_clusters(
        C=current_assignments,
        subset_indices=current_indices,
        model=model,
        config=config
    )

    return dataloaders, current_assignments, current_labels, current_indices