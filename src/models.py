import os
import sys
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn import functional as F
import torch_geometric as pyg
from torch_geometric.data import HeteroData

os.environ['TORCH'] = torch.__version__

from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn.models.basic_gnn import GAT
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

sys.path.append("./.")

from src.evaluation_metrics import * 


class SAGEConvEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        """
        SAGEConvEncoder is a simple GNN encoder that uses SAGEConv layers.
        :param num_layers: Number of SAGEConv layers to use
        :param hidden_channels: Number of hidden channels in the SAGEConv layers
        :param out_channels: Number of output channels in the SAGEConv layers
        """
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv((-1, -1), hidden_channels))
        self.convs.append(SAGEConv((-1, -1), out_channels))

    def forward(self, x_dict, edge_index):
        """
        Forward pass of the model.
        :param x_dict: Input features
        :param edge_index: Edge index tensor
        """
        # Takes the edge_index (not the edge_label_index) as input, and performs
        # message passing on the graph.
        for i, conv in enumerate(self.convs):
            if i != len(self.convs) - 1:
                x_dict = conv(x_dict, edge_index).relu()
            else:
                x_dict = conv(x_dict, edge_index)
        return x_dict

class EdgeDecoder(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, out_dim):
        """
        EdgeDecoder is a simple decoder that uses either linear layers or simply a dot product
        of the embeddings, to compute the edge scores.
        :param input_channels: Number of input channels in the decoder layers
        :param hidden_channels: Number of hidden channels in the decoder layers
        :param num_layers: Number of decoder layers. If 0, the model will use a dot product to decode the embeddings.
        """
        super().__init__()
        if out_dim > 1 and num_layers == 0:
            raise ValueError("Number of decoder layers must be greater than 0 to perform multi-class classification")
        self.is_classifier = out_dim > 1
        
        if num_layers > 0:  # TODO: small fix: if I ask for 1 layer I would always get 2
            self.is_dot_prod = False
            self.lins = torch.nn.ModuleList()
            self.lins.append(torch.nn.Linear(2 * input_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_dim))
        else:
            self.is_dot_prod = True
            print("WARNING: number of decoder layers is 0, the model will use a dot product to decode the embeddings")
            
    def forward(self, z_dict, edge_label_index):
        """
        Forward pass of the model.
        :param z_dict: Dictionary containing the embeddings of all node types
        :param edge_label_index: Edge label index tensor
        """
        row, col = edge_label_index
        
        if self.is_dot_prod:
            return (z_dict['user'][row] * z_dict['book'][col]).sum(dim=-1)
        else:
            z = torch.cat([z_dict['user'][row], z_dict['book'][col]], dim=-1)
            for i, lin in enumerate(self.lins):
                if i != len(self.lins) - 1:
                    z = lin(z).relu()
                else:
                    z = lin(z)
            
            if self.is_classifier:
                return F.log_softmax(z, dim=-1)
            else:
                return z.view(-1)
        
        
class GNN(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        conv_hidden_channels: int,
        lin_hidden_channels: int,
        num_conv_layers: int,
        use_embedding_layers: bool = False,
        book_channels: int = 384,
        user_channels: int = 3,
        num_decoder_layers: int = 1,
        encoder_arch: str = 'SAGE',
        out_dim: int = 1,
    ):
        """
        General Architecture used for our GNN-based recommender system.
        :param data: HeteroData containing the graph
        :param conv_hidden_channels: Number of hidden channels in the SAGEConv layers
        :param lin_hidden_channels: Number of hidden channels in the decoder layers
        :param num_conv_layers: Number of SAGEConv layers to use
        :param book_channels: Number of channels in the book embeddings
        :param user_channels: Number of channels in the user embeddings
        :param num_decoder_layers: Number of decoder layers. If 0, the model will use a dot product to decode the embeddings.
        :param encoder_arch: Type of encoder to use. Either 'SAGE' or 'GAT'
        :param out_dim: If bigger than 1 the model will be performing a multi-class classification.
        """
        super().__init__()
        
        # Define embeddings for the user and book nodes, and linear layers to transform original features
        self.use_embedding_layers = use_embedding_layers
        self.is_classifier = out_dim > 1
        
        if use_embedding_layers:
            self.user_emb = torch.nn.Embedding(data["user"].num_nodes, conv_hidden_channels)
            self.book_emb = torch.nn.Embedding(data["book"].num_nodes, conv_hidden_channels)
        self.user_lin = torch.nn.Linear(user_channels, conv_hidden_channels)
        self.book_lin = torch.nn.Linear(book_channels, conv_hidden_channels)
        
        # Define the encoder and decoder
        if encoder_arch == 'SAGE': 
            self.encoder = SAGEConvEncoder(conv_hidden_channels, conv_hidden_channels, num_conv_layers)
        elif encoder_arch == 'GAT':
             self.encoder = GAT(in_channels = -1, 
                                hidden_channels = conv_hidden_channels, 
                                num_layers = num_conv_layers,
                                out_channels = conv_hidden_channels,
                                add_self_loops = False)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(
            input_channels=conv_hidden_channels,
            hidden_channels=lin_hidden_channels,
            num_layers=num_decoder_layers,
            out_dim=out_dim
        )
        
    def forward(self, data: HeteroData):
        """
        Forward pass of the model.
        :param data: HeteroData containing the graph or a subgraph obtained from it by sampling.
        """
        
        # Create the feature matrices for the user and book nodes by combining embeddings
        # and linearly transformed features
        x_dict = {
            "user": self.user_lin(data["user"].x) + self.user_emb(data["user"].n_id) if self.use_embedding_layers else self.user_lin(data["user"].x),
            "book": self.book_lin(data["book"].x) + self.book_emb(data["book"].n_id) if self.use_embedding_layers else self.book_lin(data["book"].x),
        }
        
        # Perform message passing on the graph
        x_dict = self.encoder(x_dict, data.edge_index_dict)
        
        # Compute the edge scores over the edge_label_index (to be compared with the edge_label)
        return self.decoder(x_dict, data["user", "rates", "book"].edge_label_index)
    
    
    def evaluation(self, val_loader, device, criterion): 
        self.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_val_examples = 0
            predictions = []
            labels = []
            for i, batch in tqdm(enumerate(val_loader), desc=f"Validation", total=len(val_loader)):
                batch = batch.to(device)
                preds = self.forward(batch)
                
                predictions.append(preds)
                labels.append(batch["user", "rates", "book"].edge_label.to(torch.float32))

                # Loss computation
                if isinstance(criterion, torch.nn.MSELoss) and not self.is_classifier:
                    loss = criterion(
                        input=preds.unsqueeze(-1) if preds.dim() == 1 else preds,
                        target=batch["user", "rates", "book"].edge_label.to(torch.float32).unsqueeze(-1)
                    )
                elif isinstance(criterion, torch.nn.NLLLoss) and self.is_classifier:
                    loss = criterion(
                        input=preds,
                        targets = (batch["user", "rates", "book"].edge_label - 1).to(torch.long)
                    )
                elif isinstance(criterion, torch.nn.L1Loss) and not self.is_classifier:
                    loss = criterion(
                        input=preds.unsqueeze(-1) if preds.dim() == 1 else preds,
                        target=batch["user", "rates", "book"].edge_label.to(torch.float32).unsqueeze(-1)
                    )
                else:
                    raise ValueError("Criterion {} not supported with model {}being a classifier".format(
                        criterion.__class__.__name__, 
                        "" if self.is_classifier else "not ",
                    ))
                
                total_val_loss += float(loss) * preds.numel()
                total_val_examples += preds.numel()
                
            avg_val_loss = total_val_loss / total_val_examples

        return avg_val_loss, predictions
    

    def evaluation_full_batch(self, val_data, device, criterion): 
        self.eval()
        with torch.no_grad():
            predictions = []
            labels = []
            batch = val_data.to(device)
            preds = self.forward(batch)
            
            predictions.append(preds)
            labels.append(batch["user", "rates", "book"].edge_label.to(torch.float32))

            # Loss computation
            if isinstance(criterion, torch.nn.MSELoss) and not self.is_classifier:
                loss = criterion(
                    input=preds.unsqueeze(-1) if preds.dim() == 1 else preds,
                    target=batch["user", "rates", "book"].edge_label.to(torch.float32).unsqueeze(-1)
                )
            elif isinstance(criterion, torch.nn.NLLLoss) and self.is_classifier:
                loss = criterion(
                    input=preds,
                    targets = (batch["user", "rates", "book"].edge_label - 1).to(torch.long)
                )
            elif isinstance(criterion, torch.nn.L1Loss) and not self.is_classifier:
                loss = criterion(
                    input=preds.unsqueeze(-1) if preds.dim() == 1 else preds,
                    target=batch["user", "rates", "book"].edge_label.to(torch.float32).unsqueeze(-1)
                )
            else:
                raise ValueError("Criterion {} not supported with model {}being a classifier".format(
                    criterion.__class__.__name__, 
                    "" if self.is_classifier else "not ",
                ))

        return loss.item(), predictions

    
    
    def train_loop(
        self,
        train_loader,
        val_loader,
        criterion,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        writer: SummaryWriter,
        device: torch.device,
        seed: int = 42,
    ):
        """
        Train the model using the given data and optimizer while logging the training process.
        Validation is performed at the end of each epoch.
        :param data: HeteroData containing the graph
        :param optimizer: Optimizer to use for training
        :param criterion: Loss function to use for training
        :param num_epochs: Number of epochs to train the model
        :param writer: SummaryWriter to log the training process
        :param device: Device to use for training
        """
        self.train()
        
        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            total_examples = 0
            
            ######################## Train one epoch ########################
            torch.manual_seed(seed + epoch) # Ensure reproducibility, but with different seeds for each epoch
            for i, batch in tqdm(enumerate(train_loader), desc=f"Training Epoch {epoch + 1}/{num_epochs}", total=len(train_loader)):
                optimizer.zero_grad()
                batch = batch.to(device)
                
                # Forward pass
                preds = self.forward(batch)
                
                # Loss computation
                if isinstance(criterion, torch.nn.MSELoss) and not self.is_classifier:
                    loss = criterion(
                        input=preds.unsqueeze(-1) if preds.dim() == 1 else preds,
                        target=batch["user", "rates", "book"].edge_label.to(torch.float32).unsqueeze(-1)
                    )
                elif isinstance(criterion, torch.nn.NLLLoss) and self.is_classifier:
                    loss = criterion(
                        input=preds,
                        targets = (batch["user", "rates", "book"].edge_label - 1).to(torch.long)
                    )
                elif isinstance(criterion, torch.nn.L1Loss) and not self.is_classifier:
                    loss = criterion(
                        input=preds.unsqueeze(-1) if preds.dim() == 1 else preds,
                        target=batch["user", "rates", "book"].edge_label.to(torch.float32).unsqueeze(-1)
                    )
                else:
                    raise ValueError("Criterion {} not supported with model {}being a classifier".format(
                        criterion.__class__.__name__, 
                        "" if self.is_classifier else "not ",
                    ))

                # Update weights
                loss.backward()
                optimizer.step()
                
                # Log the loss
                if writer is not None:
                    writer.add_scalar(
                        tag="train/loss",
                        scalar_value=loss.item(),
                        global_step=epoch * len(train_loader) + i
                    )
                    
                # Update total loss and number of examples
                total_loss += float(loss) * preds.numel()
                total_examples += preds.numel()
            
            # Compute the average loss
            avg_loss = total_loss / total_examples
            print(f"\nEpoch {epoch + 1}/{num_epochs} - Average Train Loss: {avg_loss}")
            
            ######################## Validate the model ########################

            avg_val_loss, _  = self.evaluation(val_loader, device)
            print(f"Epoch {epoch + 1}/{num_epochs} - Average Validation Loss: {avg_val_loss}")
            
            if writer is not None:
                writer.add_scalar(
                    tag="val/loss",
                    scalar_value=avg_val_loss,
                    global_step=epoch
                )

            self.train()
        

    def train_loop_full_batch(
        self,
        train_data,
        val_data,
        criterion,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        writer: SummaryWriter,
        device: torch.device,
        val_steps: int = 1000,
        seed: int = 42,
    ):
        """
        Train the model full batch using the given data and optimizer while logging the training process.
        Validation is performed at the end of each epoch.
        :param data: HeteroData containing the graph
        :param criterion: Loss function to use for training
        :param optimizer: Optimizer to use for training
        :param num_epochs: Number of epochs to train the model
        :param writer: SummaryWriter to log the training process
        :param device: Device to use for training
        """
        self.train()
        
        for epoch in (range(num_epochs)):
            ######################## Train one epoch ########################
            optimizer.zero_grad()
            batch = train_data.to(device)
            
            # Forward pass
            preds = self.forward(batch)

            # Loss computation
            if isinstance(criterion, torch.nn.MSELoss) and not self.is_classifier:
                loss = criterion(
                    input=preds.unsqueeze(-1) if preds.dim() == 1 else preds,
                    target=batch["user", "rates", "book"].edge_label.to(torch.float32).unsqueeze(-1)
                )
            elif isinstance(criterion, torch.nn.NLLLoss) and self.is_classifier:
                loss = criterion(
                    input=preds,
                    target = (batch["user", "rates", "book"].edge_label - 1).to(torch.long)
                )
            elif isinstance(criterion, torch.nn.L1Loss) and not self.is_classifier:
                loss = criterion(
                    input=preds.unsqueeze(-1) if preds.dim() == 1 else preds,
                    target=batch["user", "rates", "book"].edge_label.to(torch.float32).unsqueeze(-1)
                )
            else:
                raise ValueError("Criterion {} not supported with model {}being a classifier".format(
                    criterion.__class__.__name__, 
                    "" if self.is_classifier else "not ",
                ))

            # Update weights
            loss.backward()
            optimizer.step()
            
            # Log the loss
            if writer is not None:
                writer.add_scalar(
                    tag="train/loss",
                    scalar_value=loss.item(),
                    global_step=epoch 
                )
            
            print(len(batch))
            print(f"\nEpoch {epoch + 1}/{num_epochs} - Train Loss: {float(loss)}")
        
            if (epoch + 1) % val_steps == 0:
                ######################## Validate the model ########################
                # Compute validation loss
                avg_val_loss, predictions  = self.evaluation_full_batch(val_data, device, criterion)
                print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {avg_val_loss}")
                
                if writer is not None:
                    writer.add_scalar(
                        tag="val/loss",
                        scalar_value=avg_val_loss,
                        global_step=epoch
                    )
                    
                # Extract most likely class if the model is a classifier
                if self.is_classifier:
                    predictions = torch.cat(predictions, dim=0).argmax(dim=-1).cpu().numpy()
                else:
                    predictions = torch.cat(predictions, dim=0).cpu().numpy()
            
                # Compute metrics
                k = 5
                threshold = 4
                big_k = 10

                results_df = pd.DataFrame([
                    {
                        "user_id": int(user_id),
                        "book_id": int(book_id),
                        "rating": int(true_label),
                        "predicted_rating": int(predicted_label),
                    }
                    for user_id, book_id, true_label, predicted_label in zip(
                        val_data[("user", "rates", "book")].edge_label_index[0],
                        val_data[("user", "rates", "book")].edge_label_index[1],
                        val_data[("user", "rates", "book")].edge_label,
                        predictions,
                    )
                ])


                # Evaluate the recommendations  
                mean_precision, mean_recall, mean_f1, map_k = evaluate_recommendations(results_df, threshold, k, big_k)
                print(f"Mean Precision@{k}: {mean_precision}")
                print(f"Mean Recall@{k}: {mean_recall}")
                print(f"Mean F1 Score@{k}: {mean_f1}")
                print(f"Mean Average Precision@{big_k}: {map_k}")
                if writer is not None:
                    writer.add_scalar(
                        tag=f"val/precision@{k}",
                        scalar_value=mean_precision,
                        global_step=epoch
                    )
                    writer.add_scalar(
                        tag=f"val/recall@{k}",
                        scalar_value=mean_recall,
                        global_step=epoch
                    )
                    writer.add_scalar(
                        tag=f"val/f1@{k}",
                        scalar_value=mean_f1,
                        global_step=epoch
                    )
                    writer.add_scalar(
                        tag=f"val/map@{big_k}",
                        scalar_value=map_k,
                        global_step=epoch
                    )

                self.train()
