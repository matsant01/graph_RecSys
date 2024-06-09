import os
import sys
import json
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch.utils.tensorboard import SummaryWriter

os.environ['TORCH'] = torch.__version__

from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn.models.basic_gnn import GAT

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

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass of the model.
        :param x_dict: Input features
        :param edge_index_dict: Edge index tensor
        """
        # Takes the edge_index_dict (subgraph) and x_dict (features) as input and performs
        # message passing on the graph.
        for i, conv in enumerate(self.convs):
            if i != len(self.convs) - 1:
                x_dict = conv(x_dict, edge_index_dict).relu()
            else:
                x_dict = conv(x_dict, edge_index_dict)
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
        author_channels: int = 384,
        language_channels: int = 26,
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
        
        # Check if node types "author" and "language" exist as well, and if so add linear layers for them too
        if "author" in data.metadata()[0]:
            self.author_lin = torch.nn.Linear(author_channels, conv_hidden_channels)
            if use_embedding_layers:
                raise NotImplementedError("Embedding layers are not supported yet unless you're using only book and user nodes.")
        else:
            self.author_lin = None
            
        if "language" in data.metadata()[0]:
            self.language_lin = torch.nn.Linear(language_channels, conv_hidden_channels)
            if use_embedding_layers:
                raise NotImplementedError("Embedding layers are not supported yet unless you're using only book and user nodes.")
        else:
            self.language_lin = None
        
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
        # Add the basic node types, with embdeddings if specified
        x_dict = {
            "user": self.user_lin(data["user"].x) + self.user_emb(data["user"].n_id) if self.use_embedding_layers else self.user_lin(data["user"].x),
            "book": self.book_lin(data["book"].x) + self.book_emb(data["book"].n_id) if self.use_embedding_layers else self.book_lin(data["book"].x),
        }
            
        # Add more node types if they exist
        if self.author_lin is not None:
            x_dict["author"] = self.author_lin((data["author"].x).to(torch.float32))
        if self.language_lin is not None:
            x_dict["language"] = self.language_lin((data["language"].x).to(torch.float32))

        
        # Perform message passing on the graph
        x_dict = self.encoder(x_dict, data.edge_index_dict)
        
        # Compute the edge scores over the edge_label_index (to be compared with the edge_label)
        return self.decoder(x_dict, data["user", "rates", "book"].edge_label_index)
    
    
    def evaluation_batched(self, val_loader, device, criterion, k=5, big_k=10): 
        self.eval()
        with torch.no_grad():
            val_predictions = []
            val_labels = []
            val_edge_label_index = []
            total_val_loss = 0
            total_val_samples = 0
            
            for i, batch in tqdm(enumerate(val_loader), desc=f"Validation", total=len(val_loader)):
                batch = batch.to(device)
                preds = self.forward(batch)
                
                # Loss computation
                if ((isinstance(criterion, torch.nn.MSELoss) and not self.is_classifier) or
                    (isinstance(criterion, torch.nn.L1Loss) and not self.is_classifier)):
                    inputs = preds.unsqueeze(-1) if preds.dim() == 1 else preds
                    targets = batch["user", "rates", "book"].edge_label.to(torch.float32).unsqueeze(-1)
                elif isinstance(criterion, torch.nn.NLLLoss) and self.is_classifier:
                    inputs = preds
                    targets = (batch["user", "rates", "book"].edge_label - 1).to(torch.long)
                elif criterion is None:
                    pass
                else:
                    raise ValueError("Criterion {} not supported with model {}being a classifier".format(
                        criterion.__class__.__name__, 
                        "" if self.is_classifier else "not ",
                    ))

                loss = criterion(input=inputs, target=targets).item() if criterion is not None else -1
                total_val_loss += float(loss) * preds.numel()
                total_val_samples += preds.numel()
                
                val_predictions.append(preds)
                val_labels.append(batch["user", "rates", "book"].edge_label.to(torch.float32))
                val_edge_label_index.append(batch["user", "rates", "book"].edge_label_index)
                
            # Compute the average loss
            loss = total_val_loss / total_val_samples
                
            # Prepare val_predictions (sampling in case of multi-class classification)
            val_labels = torch.cat(val_labels, dim=0).cpu().numpy()
            val_edge_label_index = torch.cat(val_edge_label_index, dim=1)
            if self.is_classifier:
                val_predictions = torch.cat(val_predictions, dim=0).argmax(dim=-1).cpu().numpy()
            else:
                val_predictions = torch.cat(val_predictions, dim=0).cpu().numpy()

            # Compute metrics
            results_df = pd.DataFrame([
                {
                    "user_id": int(user_id),
                    "book_id": int(book_id),
                    "rating": int(true_label),
                    "predicted_rating": int(predicted_label),
                }
                for user_id, book_id, true_label, predicted_label in zip(
                    val_edge_label_index[0],
                    val_edge_label_index[1],
                    val_labels,
                    val_predictions,
                )
            ])

            # Evaluate the recommendations  
            mean_precision, mean_recall, mean_f1, map_k = evaluate_recommendations(results_df, threshold=4, k=k, n_precisions=big_k)
            metrics = {
                f"precision@{k}": mean_precision,
                f"recall@{k}": mean_recall,
                f"f1@{k}": mean_f1,
                f"map@{big_k}": map_k,
            }

        return loss, metrics


    def evaluation_full_batch(self, val_data, device, criterion, k=5, big_k=10): 
        self.eval()
        print("Validating...")
        with torch.no_grad():
            predictions = []
            labels = []
            batch = val_data.to(device)
            preds = self.forward(batch)
            
            predictions.append(preds)
            labels.append(batch["user", "rates", "book"].edge_label.to(torch.float32))

            # Loss computation
            if ((isinstance(criterion, torch.nn.MSELoss) and not self.is_classifier) or
                (isinstance(criterion, torch.nn.L1Loss) and not self.is_classifier)):
                inputs = preds.unsqueeze(-1) if preds.dim() == 1 else preds
                targets = batch["user", "rates", "book"].edge_label.to(torch.float32).unsqueeze(-1)
            elif isinstance(criterion, torch.nn.NLLLoss) and self.is_classifier:
                inputs = preds
                targets = (batch["user", "rates", "book"].edge_label - 1).to(torch.long)
            elif criterion is None:
                pass
            else:
                raise ValueError("Criterion {} not supported with model {}being a classifier".format(
                    criterion.__class__.__name__, 
                    "" if self.is_classifier else "not ",
                ))

            loss = criterion(input=inputs, target=targets).item() if criterion is not None else 0
            
            # Prepare predictions (sampling in case of multi-class classification)
            if self.is_classifier:
                predictions = torch.cat(predictions, dim=0).argmax(dim=-1).cpu().numpy()
            else:
                predictions = torch.cat(predictions, dim=0).cpu().numpy()

            # Compute metrics
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
            mean_precision, mean_recall, mean_f1, map_k = evaluate_recommendations(results_df, threshold=4, k=k, n_precisions=big_k)
            metrics = {
                f"precision@{k}": mean_precision,
                f"recall@{k}": mean_recall,
                f"f1@{k}": mean_f1,
                f"map@{big_k}": map_k,
            }

        return loss, metrics

    
    def train_loop_batched(
        self,
        train_loader,
        val_data,
        criterion,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        writer: SummaryWriter,
        device: torch.device,
        val_steps: int = -1,
        output_dir: str = "./output",
        seed: int = 42,
    ):
        """
        Train the model using the data loaders and optimizer while logging the training process.
        Validation is performed every val_steps steps.
        :param train_loader: DataLoader containing the training data
        :param val_data: HeteroData containing the validation data
        :param criterion: Loss function to use for training
        :param optimizer: Optimizer to use for training
        :param num_epochs: Number of epochs to train the model
        :param writer: SummaryWriter to log the training process
        :param device: Device to use for training
        :param val_steps: Number of steps between each validation. If -1, validation is performed at the end of each epoch
        :param output_dir: Directory where to save the model
        """
        self.train()
        
        best_map_at_k = 0
        if val_steps < 0:
            val_steps = abs(val_steps) * len(train_loader)
            print("Validation steps set to {}".format(val_steps))
        
        for epoch in range(num_epochs):
            pbar = tqdm(enumerate(train_loader), desc=f"Training - Epoch {epoch + 1}/{num_epochs} - Loss: 0", total=len(train_loader))
            
            for i, batch in pbar:
                curr_step = epoch * len(train_loader) + i
                
                ######################## Train one epoch ########################
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
                        global_step=curr_step,
                    )
                
                # Update progress bar
                pbar.set_description(f"Training - Epoch {epoch + 1}/{num_epochs} - Loss: {float(loss):.5f}")
            
                if curr_step % val_steps == 0 and curr_step > 0:
                    ######################## Validate the model ########################
                    # Compute validation loss and metrics
                    k = 5
                    big_k = 15
                    avg_val_loss, metrics  = self.evaluation_full_batch(val_data, device, criterion, k=k, big_k=big_k)
                    print(f"Validation - Epoch {epoch + 1}/{num_epochs} - Val Loss: {avg_val_loss:.5f} - MAP@{big_k}: {metrics['map@{}'.format(big_k)]:.3f}")
                    
                    if writer is not None:
                        writer.add_scalar(
                            tag="val/loss",
                            scalar_value=avg_val_loss,
                            global_step=epoch
                        )
                        for metric in metrics:
                            writer.add_scalar(
                                tag=f"val/{metric}",
                                scalar_value=metrics[metric],
                                global_step=epoch
                            )
                            
                    # Save the model only if the MAP@k is better than the previous best
                    if metrics["map@{}".format(big_k)] > best_map_at_k:
                        best_map_at_k = metrics["map@{}".format(big_k)]
                        # Save model
                        model_path = os.path.join(output_dir, "best_model.pt")
                        torch.save(self.state_dict(), model_path)
                        print("\tBest model updated at: {}\n".format(model_path))
                        # Update config.json adding the best epoch value
                        config_path = os.path.join(output_dir, "config.json")
                        with open(config_path, "r") as f:
                            config = json.load(f)
                        config["best_step"] = curr_step
                        with open(config_path, "w") as f:
                            json.dump(config, f)

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
        output_dir: str = "./output",
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
        
        best_map_at_k = 0
        pbar = tqdm(range(num_epochs), desc="Training")
        
        for epoch in pbar:
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
            
            # Update progress bar
            pbar.set_description(f"Training - Epoch {epoch + 1}/{num_epochs} - Train Loss: {float(loss):.5f}")
        
            if (epoch + 1) % val_steps == 0:
                ######################## Validate the model ########################
                # Compute validation loss and metrics
                k = 5
                big_k = 15
                avg_val_loss, metrics  = self.evaluation_full_batch(val_data, device, criterion, k=k, big_k=big_k)
                print(f"Validation - Epoch {epoch + 1}/{num_epochs} - Val Loss: {avg_val_loss:.5f} - MAP@{big_k}: {metrics['map@{}'.format(big_k)]:.3f}")
                
                if writer is not None:
                    writer.add_scalar(
                        tag="val/loss",
                        scalar_value=avg_val_loss,
                        global_step=epoch
                    )
                    for metric in metrics:
                        writer.add_scalar(
                            tag=f"val/{metric}",
                            scalar_value=metrics[metric],
                            global_step=epoch
                        )
                        
                # Save the model only if the MAP@k is better than the previous best
                if metrics["map@{}".format(big_k)] > best_map_at_k:
                    best_map_at_k = metrics["map@{}".format(big_k)]
                    # Save model
                    model_path = os.path.join(output_dir, "best_model.pt")
                    torch.save(self.state_dict(), model_path)
                    print("\tBest model updated at: {}\n".format(model_path))
                    # Update config.json adding the best epoch value
                    config_path = os.path.join(output_dir, "config.json")
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    config["best_epoch"] = epoch
                    with open(config_path, "w") as f:
                        json.dump(config, f)

                self.train()
