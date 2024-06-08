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

sys.path.append(".")

from src.models import GNN
from src.matrix_factorization import *

SEED = 42

def validate_arguments():
    # Create parser
    parser = argparse.ArgumentParser(description='Train a model given a dataset and architecture')
    parser.add_argument('--data_path', type=str, help='Path to the folder containing the datasets')
    parser.add_argument('--output_dir', type=str, help='Root folder where model, logs and config will be saved')
    parser.add_argument('--num_conv_layers', type=int, help='Number of SAGE convolutional layers')
    parser.add_argument('--hidden_channels', type=int, help='Number of hidden channels in the SAGE convolutional layers')
    parser.add_argument('--use_embedding_layers', action='store_true', help='Whether to use embedding layers or not')
    parser.add_argument('--num_decoder_layers', type=int, help='Number of decoder layers, if 0 the decoding will be done by a dot product')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train the model')
    parser.add_argument('--validation_steps', type=int, help='Number of steps between each validation. Default 500.', default=500)
    parser.add_argument('--lr', type=float, help='Learning rate for the optimizer', default=0.01)
    parser.add_argument('--loss', type=str, help='Loss function to use for training. Either "mse", "mae" or "nll"')
    parser.add_argument('--device', type=str, help='Device to use for training.')
    parser.add_argument('--encoder_arch', type=str, help='Either GAT or SAGE', default='SAGE')
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
    if "cuda" in args.device and not torch.cuda.is_available():
        raise ValueError("CUDA is not available, please use a CPU device")
    else:
        # Check if device is valid
        try:
            tmp_device = torch.device(args.device)
        except:
            raise ValueError("Device {} is not valid".format(args.device))
        
    return args
    


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
    test_data = torch.load(os.path.join(kwargs['data_path'], "test_hetero.pt"))
    data = torch.load(os.path.join(kwargs['data_path'], "data_hetero.pt"))
    
    
    if kwargs['verbose']:
        print("\nData loaded successfully")
        print("\nTrain HeteroData:\n\n{}", train_data)
        print("\n\nValidation HeteroData:\n\n{}", val_data)
    
    
    ####################### Training #######################
    log_dir = os.path.join(output_dir, "logs")
    device = torch.device(kwargs['device'])
    writer = SummaryWriter(log_dir=log_dir)
    
    # Load model and optimizer
    model = GNN(
        data=data,
        conv_hidden_channels=kwargs['hidden_channels'],
        lin_hidden_channels=kwargs['hidden_channels'],
        num_conv_layers=kwargs['num_conv_layers'],
        num_decoder_layers=kwargs['num_decoder_layers'],
        use_embedding_layers=kwargs['use_embedding_layers'],
        encoder_arch=kwargs['encoder_arch'],
        out_dim=1 if kwargs["loss"] != "nll" else 5,    # if we want to see the problem as a classification problem we need to output 5 classes
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=kwargs['lr'])
    
    # Choose loss function
    if kwargs['loss'] == "mse":
        loss = torch.nn.MSELoss()
    elif kwargs['loss'] == "mae":
        loss = torch.nn.L1Loss()
    elif kwargs['loss'] == "nll":
        loss = torch.nn.NLLLoss()
    else:
        raise ValueError("Loss function {} is not valid".format(kwargs['loss']))
    
    if kwargs['verbose']:
        print("\nModel created successfully and loaded on device: {}".format(device))
        print(model)
        print("\n\nStarting training...\n")
    
    # Start training
    model.train_loop_full_batch(
        train_data,
        val_data,
        criterion=loss,
        optimizer=optimizer,
        num_epochs=kwargs['num_epochs'],
        writer=writer,
        device=device,
        output_dir=output_dir,
        val_steps=kwargs['validation_steps'],
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