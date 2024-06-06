import os
import sys
import json
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.loader import LinkNeighborLoader, HGTLoader
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

sys.path.append("./.")

from src.models import GNN
from src.matrix_factorization import *

SEED = 42

def validate_arguments():
    # Create parser
    parser = argparse.ArgumentParser(description='Train a model given a dataset and architecture')
    parser.add_argument('--data_path', type=str, help='Path to the folder containing the datasets')
    parser.add_argument('--output_dir', type=str, help='Root folder where model, logs and config will be saved')
    parser.add_argument('--model_type', type=str, help='Type of model to train. Either "GNN" or "MatrixFactorization"')
    parser.add_argument('--num_conv_layers', type=int, help='Number of SAGE convolutional layers')
    parser.add_argument('--hidden_channels', type=int, help='Number of hidden channels in the SAGE convolutional layers')
    parser.add_argument('--use_embedding_layers', action='store_true', help='Whether to use embedding layers or not')
    parser.add_argument('--num_decoder_layers', type=int, help='Number of decoder layers, if 0 the decoding will be done by a dot product')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, help='Learning rate for the optimizer', default=0.01)
    parser.add_argument('--sampler_type', type=str, help='Type of sampler to use to create batches. Either "link-neighbor" (LinkNeighborLoader) or "HGT" (HGTLoader)')
    parser.add_argument('--device', type=str, help='Device to use for training.')
    parser.add_argument('--do_neg_sampling', action='store_true', help='Whether to do negative sampling or not')
    parser.add_argument('--num_neighbors_in_sampling', type=int, help='Number of neighbors to sample in the sampling process, if applicable. Default is 25', default=25)
    parser.add_argument('--batch_size', type=int, help='Batch size to use for training. Default is 1024', default=1024)
    parser.add_argument('--verbose', action='store_true', help='Print progress messages')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_path):
        raise ValueError(f"Path to the data folder does not exist: {args.data_path}")
    if args.num_conv_layers < 1:
        raise ValueError("Number of convolutional layers must be greater than 0")
    if args.hidden_channels < 1:
        raise ValueError("Number of hidden channels must be greater than 0")
    if args.num_decoder_layers == 0:
        print("WARNING: number of decoder layers is 0, the model will use a dot product to decode the embeddings")
    if args.num_epochs < 1:
        raise ValueError("Number of epochs must be greater than 0")
    if args.lr <= 0:
        raise ValueError("Learning rate must be greater than 0")
    if args.sampler_type not in ['link-neighbor', 'HGT']:
        raise ValueError("Sampler type must be either 'link-neighbor' or 'HGT'")
    if args.model_type not in ['GNN', 'MatrixFactorization']:
        raise ValueError("Model type must be either 'GNN' or 'MatrixFactorization'")
    if "cuda" in args.device and not torch.cuda.is_available():
        raise ValueError("CUDA is not available, please use a CPU device")
    else:
        # Check if device is valid
        try:
            tmp_device = torch.device(args.device)
        except:
            raise ValueError("Device {} is not valid".format(args.device))
        
    return args


def get_model(data: HeteroData, **kwargs):
    if kwargs['model_type'] == "GNN":
        model = GNN(
            data=data,
            conv_hidden_channels=kwargs['hidden_channels'],
            lin_hidden_channels=kwargs['hidden_channels'],
            num_conv_layers=kwargs['num_conv_layers'],
            num_decoder_layers=kwargs['num_decoder_layers'],
            use_embedding_layers=kwargs['use_embedding_layers'],
        )
    elif kwargs['model_type'] == "MatrixFactorization":
        NotImplementedError("Matrix Factorization model is not implemented yet")
    return model
    


def main(**kwargs):
    ####################### Configuration #######################
    torch.manual_seed(SEED)

    # Create output directory
    root_output_dir = kwargs['output_dir']
    output_dir = os.path.join(root_output_dir, "model_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config = kwargs
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
        
    if kwargs['verbose']:
        print("Configuration saved at: {}".format(config_path))

    ####################### Load Data #######################
    if kwargs['verbose']:
        print("Loading data...")
    
    # Load data
    train_data = torch.load(os.path.join(kwargs['data_path'], "train_hetero.pt"))
    val_data = torch.load(os.path.join(kwargs['data_path'], "val_hetero.pt"))
    data = torch.load(os.path.join(kwargs['data_path'], "data_hetero.pt"))
    
    # Create Loaders
    if kwargs['sampler_type'] == "link-neighbor":
        num_neighbors = [kwargs['num_neighbors_in_sampling']] * kwargs['num_conv_layers']
        train_loader =  LinkNeighborLoader(
            data=train_data,
            num_neighbors=num_neighbors,
            neg_sampling_ratio=2 if kwargs["do_neg_sampling"] else None,
            edge_label_index=(("user", "rates", "book"), train_data["user", "rates", "book"].edge_label_index),
            edge_label=train_data["user", "rates", "book"].edge_label,
            batch_size=kwargs['batch_size'],
            shuffle=True,
        )
        val_loader =  LinkNeighborLoader(
            data=val_data,
            num_neighbors=num_neighbors,
            neg_sampling_ratio=2 if kwargs["do_neg_sampling"] else None,
            edge_label_index=(("user", "rates", "book"), val_data["user", "rates", "book"].edge_label_index),
            edge_label=val_data["user", "rates", "book"].edge_label,
            batch_size=kwargs['batch_size'],
            shuffle=True,
        )
    elif kwargs['sampler_type'] == "HGT":
        train_loader = HGTLoader(
            train_data,
            num_samples=[1024] * 4,  
            shuffle=True,
            batch_size=128,
            input_nodes=("user", None),
        )
        val_loader = HGTLoader(
            val_data,
            num_samples=[1024] * 4,
            shuffle=False,
            batch_size=128,
            input_nodes=("user", None),
        )
        
    if kwargs['verbose']:
        print("\nData loaded successfully")
        print("\nTrain HeteroData:\n\n{}", train_data)
        print("\n\nValidation HeteroData:\n\n{}", val_data)
    
    
    ####################### Training #######################
    log_dir = os.path.join(output_dir, "logs")
    device = torch.device(kwargs['device'])
    writer = SummaryWriter(log_dir=log_dir)
    
    # Load model and optimizer
    model = get_model(data, **kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])
    
    if kwargs['verbose']:
        print("\nModel created successfully and loaded on device: {}".format(device))
        print(model)
        print("\n\nStarting training...\n")
    
    # Start training
    model.train_loop(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=kwargs['num_epochs'],
        writer=writer,
        device=device,
        seed=SEED
    )
    
    # Save model
    if kwargs['verbose']:
        print("\nTraining completed successfully. Saving model...")
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print("Training completed successfully. Model saved at: {}".format(model_path))
    
    # Close writer
    writer.close()
    

if __name__ == "__main__":
    args = validate_arguments()
    if args.verbose:
        print("Arguments validated successfully")
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        print("\n\n")
    main(**vars(args))