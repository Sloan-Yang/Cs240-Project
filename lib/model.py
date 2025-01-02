import torchvision
import torch
from math import ceil
import logging
import numpy as np
from torch.nn import Linear, Dropout, AvgPool2d
from torch.nn.init import xavier_normal_

def build_resnet50(pretrained=True):
    resnet_model = torchvision.models.resnet50(pretrained=pretrained)

    resnet_model.feature_extractor = torch.nn.Sequential(
        resnet_model.conv1, resnet_model.bn1, resnet_model.relu, resnet_model.maxpool,
        resnet_model.layer1, resnet_model.layer2, resnet_model.layer3, resnet_model.layer4
    )

    resnet_model.feature_dim = 2048

    for module in filter(
        lambda m: isinstance(m, torch.nn.BatchNorm2d), resnet_model.modules()
    ):
        module.eval()
        module.train = lambda _: None

    return resnet_model

def organize_parameters(model, custom_module_names):
    """
    Separates model parameters into 'backbone' and other specified modules.
    """
    param_groups = {k: [] for k in ['backbone', *custom_module_names]}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        if module_name not in custom_module_names:
            param_groups['backbone'].append(param)
        else:
            param_groups[module_name].append(param)

    total_params = len(list(model.parameters()))
    grouped_params = sum(len(param_groups[group]) for group in param_groups)
    assert total_params == grouped_params, "Parameter count mismatch!"

    return param_groups

def initialize_split_layer(linear_layer, num_clusters, embedding_size):
    """Initialize parts of the embedding layer corresponding to each cluster."""
    for cluster_idx in range(num_clusters):
        indices = torch.arange(
            cluster_idx * ceil(embedding_size / num_clusters),
            min((cluster_idx + 1) * ceil(embedding_size / num_clusters), embedding_size)
        ).long()
        temp_layer = torch.nn.Linear(linear_layer.weight.shape[1], len(indices))
        linear_layer.weight.data[indices] = xavier_normal_(temp_layer.weight.data, gain=1)
        linear_layer.bias.data[indices] = temp_layer.bias.data

def enhance_model_with_embedding(model, config, embedding_size, normalize_output=True):
    """Enhance the ResNet model by adding an embedding layer and configuring clusters."""
    model.global_pooling = AvgPool2d(7, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
    model.dropout_layer = Dropout(0.01)

    device = next(model.parameters()).device
    model.embedding_layer = Linear(model.feature_dim, embedding_size).to(device)

    torch.manual_seed(config['random_seed'] + 1)
    np.random.seed(config['random_seed'] + 1)

    initialize_split_layer(
        model.embedding_layer, config['num_clusters'], config['embedding_size']
    )

    model.parameter_groups = organize_parameters(
        model=model,
        custom_module_names=['embedding_layer']
    )

    model.cluster_neurons = [
        np.arange(
            cluster_idx * ceil(embedding_size / config['num_clusters']),
            min((cluster_idx + 1) * ceil(embedding_size / config['num_clusters']), embedding_size)
        ) for cluster_idx in range(config['num_clusters'])
    ]

    def forward_pass(input_tensor, use_penultimate=False):
        x = model.feature_extractor(input_tensor)
        x = model.global_pooling(x)
        x = model.dropout_layer(x)
        x = x.view(x.size(0), -1)

        if not use_penultimate:
            x = model.embedding_layer(x)
            for neuron_indices in model.cluster_neurons:
                x[:, neuron_indices] = torch.nn.functional.normalize(
                    x[:, neuron_indices], p=2, dim=1
                )
        else:
            x = torch.nn.functional.normalize(x, p=2, dim=1)

        return x

    model.forward = forward_pass

def build_model(config):
    base_model = build_resnet50(pretrained=True)
    enhance_model_with_embedding(
        model=base_model,
        config=config,
        embedding_size=config['embedding_size'],
        normalize_output=True
    )
    return base_model
