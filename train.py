from __future__ import print_function
from __future__ import division

import collections
import os
import matplotlib
import numpy as np
import logging
import torch
import time
import json
import random
import shelve
from tqdm import tqdm
import lib
from lib.clustering import make_clustered_dataloaders
import warnings

warnings.simplefilter("ignore", category=PendingDeprecationWarning)
os.putenv("OMP_NUM_THREADS", "8")

def load_configuration(config_file):
    """Load and parse the configuration file."""
    with open(config_file, 'r') as file:
        config = json.load(file)

    def evaluate_json(config):
        for key, value in config.items():
            if isinstance(value, str):
                if value.startswith(('range', 'float')):
                    config[key] = eval(value)
            elif isinstance(value, dict):
                evaluate_json(value)

    evaluate_json(config)
    return config

def pretty_json_dump(**kwargs):
    """Dump JSON with readable formatting."""
    return json.dumps(**kwargs).replace('\\n', '\n    ')

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for additional data types."""
    def default(self, obj):
        if isinstance(obj, range):
            return f'range({obj.start}, {obj.stop})'
        return super().default(obj)

def evaluate_model(model, dataloaders, logger, backend='faiss', config=None):
    """Evaluate the model on the given dataloaders."""
    if config and config['dataset_selected'] == 'inshop':
        query_loader = lib.data.loader.make(config, model, 'eval', inshop_type='query')
        gallery_loader = lib.data.loader.make(config, model, 'eval', inshop_type='gallery')
        scores = lib.utils.evaluate_in_shop(
            model,
            dl_query=query_loader,
            dl_gallery=gallery_loader,
            use_penultimate=False,
            backend=backend
        )
    else:
        scores = lib.utils.evaluate(
            model,
            dataloaders['eval'],
            use_penultimate=False,
            backend=backend
        )
    return scores

def train_single_batch(model, loss_function, optimizer, config, batch, dataset, epoch):
    """Train the model on a single batch."""
    images, labels, ids = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True), batch[2]

    optimizer.zero_grad()
    embeddings = model(images)

    if epoch < config['finetune_epoch'] * 8 / 19:
        embeddings = embeddings.split(config['sz_embedding'] // config['nb_clusters'], dim=1)[dataset.id]

    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    loss = loss_function[dataset.id](embeddings, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def get_loss_function(config):
    """Initialize the loss function."""
    dataset_name = config['dataset_selected']
    num_classes = len(config['dataset'][dataset_name]['classes']['train'])
    logging.debug(f'Creating margin loss with {num_classes} classes.')
    return [
        lib.loss.MarginLoss(num_classes).cuda()
        for _ in range(config['nb_clusters'])
    ]

def get_optimizer(config, model, loss_function):
    """Initialize the optimizer."""
    return torch.optim.Adam([
        {
            'params': model.parameters_dict['backbone'],
            **config['opt']['backbone']
        },
        {
            'params': model.parameters_dict['embedding'],
            **config['opt']['embedding']
        }
    ])

def initialize_training(config):
    """Set up and start the training process."""
    import matplotlib.pyplot as plt

    metrics = {}
    faiss_memory_manager = lib.faissext.MemoryReserver()
    os.makedirs(config['log']['path'], exist_ok=True)

    log_file_path = os.path.join(config['log']['path'], config['log']['name'])
    if os.path.exists(log_file_path):
        warnings.warn(f'Log file exists: {log_file_path}. Appending underscore.')
        config['log']['name'] += '_'

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.DEBUG if config['verbose'] else logging.INFO,
        handlers=[
            logging.FileHandler(f"{config['log']['path']}/{config['log']['name']}.log"),
            logging.StreamHandler()
        ]
    )

    logging.info(pretty_json_dump(obj=config, indent=4, cls=CustomJSONEncoder, sort_keys=True))

    torch.cuda.set_device(config['cuda_device'])

    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed_all(config['random_seed'])

    faiss_memory_manager.lock(config['backend'])
    model = lib.model.make(config).cuda()

    start_epoch = 0
    best_epoch = -1
    best_recall = 0

    dataloaders = {}
    for loader_type in ['init', 'eval']:
        if config['dataset_selected'] == 'inshop':
            if loader_type == 'init':
                dataloaders[loader_type] = lib.data.loader.make(config, model, loader_type, inshop_type='train')
        else:
            dataloaders[loader_type] = lib.data.loader.make(config, model, loader_type)

    loss_function = get_loss_function(config)
    optimizer = get_optimizer(config, model, loss_function)

    faiss_memory_manager.release()
    logging.info("Evaluating initial model...")
    metrics[-1] = {'score': evaluate_model(model, dataloaders, logging, backend=config['backend'], config=config)}

    dataloaders['train'], cluster_assignments, cluster_labels, cluster_indices = make_clustered_dataloaders(
        model, dataloaders['init'], config, reassign_clusters=False, log=logging
    )
    faiss_memory_manager.lock(config['backend'])

    metrics[-1].update({'C': cluster_assignments, 'T': cluster_labels, 'I': cluster_indices})

    logging.info(f"Training for {config['nb_epochs']} epochs.")
    training_losses = []
    start_time = time.time()

    for epoch in range(start_epoch, config['nb_epochs']):
        metrics[epoch] = {}
        epoch_start_time = time.time()
        epoch_losses = []

        if epoch >= config['finetune_epoch']:
            if epoch == config['finetune_epoch'] or epoch == start_epoch:
                logging.info("Starting fine-tuning...")
                config['nb_clusters'] = 1
                faiss_memory_manager.release()
                dataloaders['train'], cluster_assignments, cluster_labels, cluster_indices = make_clustered_dataloaders(
                    model, dataloaders['init'], config, log=logging
                )
                assert len(dataloaders['train']) == 1

        elif epoch > 0 and config['recluster']['enabled'] and config['nb_clusters'] > 0:
            if epoch % config['recluster']['mod_epoch'] == 0:
                logging.info("Reclustering dataloaders...")
                faiss_memory_manager.release()
                dataloaders['train'], cluster_assignments, cluster_labels, cluster_indices = make_clustered_dataloaders(
                    model, dataloaders['init'], config, reassign_clusters=True, log=logging
                )
                faiss_memory_manager.lock(config['backend'])

        train_loader = lib.data.loader.merge(dataloaders['train'])
        total_batches = sum(len(dl) for dl in dataloaders['train'])

        for batch, dataset in tqdm(train_loader, total=total_batches, desc=f"Train epoch {epoch}"):
            loss = train_single_batch(model, loss_function, optimizer, config, batch, dataset, epoch)
            epoch_losses.append(loss)

        epoch_end_time = time.time()
        training_losses.append(np.mean(epoch_losses[-20:]))

        logging.info(
            f"Epoch {epoch}, loss: {training_losses[-1]}, time: {epoch_end_time - epoch_start_time:.2f}s"
        )

        faiss_memory_manager.release()
        metrics[epoch]['score'] = evaluate_model(model, dataloaders, logging, backend=config['backend'], config=config)
        metrics[epoch]['loss'] = {'train': training_losses[-1]}

        recall_at_1 = metrics[epoch]['score']['recall'][0]
        if recall_at_1 > best_recall:
            best_recall = recall_at_1
            best_epoch = epoch
            logging.info("Best epoch!")

        model.current_epoch = epoch

        with shelve.open(os.path.join(config['log']['path'], config['log']['name']), writeback=True) as db:
            if 'config' not in db:
                db['config'] = config
            if 'metrics' not in db:
                db['metrics'] = {}
                if -1 in metrics:
                    db['metrics'][-1] = metrics[-1]
            db['metrics'][epoch] = metrics[epoch]

        if config['save_model'] and recall_at_1 > best_recall:
            torch.save(
                model.state_dict(),
                os.path.join(config['log']['path'], f"{config['log']['name']}.pt")
            )
            logging.info("Checkpoint saved.")

    total_time = time.time() - start_time
    logging.info(f"Total training time: {total_time / 60:.2f} minutes")
    logging.info(f"Best R@1: {best_recall} at epoch {best_epoch}")
