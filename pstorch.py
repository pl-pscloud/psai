import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score, f1_score, root_mean_squared_log_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')

class RMSLELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSLELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # Ensure predictions are non-negative
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)

        # Compute the logarithm of (y + 1)
        log_pred = torch.log(y_pred + 1 + self.eps)
        log_true = torch.log(y_true + 1 + self.eps)

        # Compute the squared differences
        loss = torch.sqrt(torch.mean((log_pred - log_true) ** 2))
        return loss

class MAPE_Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.abs((target - pred) / (target + self.eps)))

#pytorch classifier
class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self, 
            learning_rate=0.001, 
            optimizer_name='adam', 
            batch_size=32, 
            max_epochs=1000, 
            patience=10, 
            net=[(1, 256, 256, 1, 0),(1, 256, 1, 0, 0)], 
            embedding_info=None,  
            loss='bcelogit', 
            verbose=1,
            weight_init = 'default',
            device = 'cpu',
            num_threads = os.cpu_count(),
        ):
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.max_epochs = max_epochs
            self.patience = patience
            self.model = None
            self.best_model = None
            self.device = device
            self.input_dim = None
            self.net = net
            self.is_fitted_ = False
            self.optimizer_name = optimizer_name
            self.optimizer = None
            self.criterion = None
            self.verbose = verbose
            self.embedding_info = embedding_info  # Embedding info for categorical features
            self.loss = loss
            self.eval_info = {}
            self.weight_init = weight_init
            self._classes = None  # Initialize classes attribute
            self.num_threads = num_threads

            if self.device == 'cpu':
                torch.set_num_threads(self.num_threads)

    def build_model(self):
        layers = []
        embedding_layers = []
        
        # Add embedding layers for high cardinality categorical features if embedding_info is provided
        total_embedding_dim = 0
        if self.embedding_info:
            for num_categories, embedding_dim in self.embedding_info.values():
                embedding_layer = nn.Embedding(num_categories, embedding_dim)
                embedding_layers.append(embedding_layer.to(self.device))
                total_embedding_dim += embedding_dim

        self.embedding_layers = nn.ModuleList(embedding_layers)

        # First layer: concatenate embeddings and numerical input
        first_layer_input_dim = total_embedding_dim + (self.input_dim if self.input_dim is not None else 0)

        # Define the rest of the network
        if self.net[0][0] == 1:
            l = nn.Linear(first_layer_input_dim, self.net[0][2])
            l = self.configure_weight(l)
            layers.append(l)
        if self.net[0][4] == 1:
            layers.append(nn.BatchNorm1d(self.net[0][2]))
        elif self.net[0][4] == 2:
            layers.append(nn.LayerNorm(self.net[0][2]))

        if self.net[0][3] == 1:
            layers.append(nn.ReLU())
        elif self.net[0][3] == 2:
            layers.append(nn.LeakyReLU(negative_slope=0.01))
        elif self.net[0][3] == 3:
            layers.append(nn.Sigmoid())
        elif self.net[0][3] == 4:
            layers.append(nn.Tanh())
        elif self.net[0][3] == 5:
            layers.append(nn.SiLU())
        elif self.net[0][3] == 6:
            layers.append(nn.Softmax())
        elif self.net[0][3] == 7:
            layers.append(nn.ELU(alpha=1.0))
        elif self.net[0][3] == 8:
            layers.append(nn.SELU())
        elif self.net[0][3] == 0:
            pass  # No activation

        # Iterate over the rest of the net configuration
        for layer in self.net[1:]:
            #add layer
            if layer[0] == 1:
                #Dense
                l = nn.Linear(layer[1], layer[2])
                l = self.configure_weight(l)
                layers.append(l)

                #add normalization
                if layer[4] == 1:
                    layers.append(nn.BatchNorm1d(layer[2]))
                elif layer[4] == 2:
                    layers.append(nn.LayerNorm(layer[2]))
                    
                #add activation function
                if layer[3] == 1:
                    layers.append(nn.ReLU())
                elif layer[3] == 2:
                    layers.append(nn.LeakyReLU(negative_slope=0.01))
                elif layer[3] == 3:
                    layers.append(nn.Sigmoid())
                elif layer[3] == 4:
                    layers.append(nn.Tanh())
                elif layer[3] == 5:
                    layers.append(nn.SiLU())
                elif layer[3] == 6:
                    layers.append(nn.Softmax())
                elif layer[3] == 7:
                    layers.append(nn.ELU(alpha=1.0))
                elif layer[3] == 9:
                    layers.append(nn.SELU())
                elif layer[3] == 0:
                    pass  # No activation
                    
            elif layer[0] == 0:
                #Dropout
                layers.append(nn.Dropout(layer[1]))
                
        if self.verbose >= 2:
            print(f"Embedding layers: {embedding_layers}")
            print(f"Layers: {layers}")
            
        return nn.Sequential(*layers).to(self.device)

    def forward_embeddings(self, cat_features):
        if not self.embedding_layers:
            return None

        # Debugging to print the range of cat_features
        if self.verbose >= 3:
            print(f"Categorical Features Tensor Before Clamping: {cat_features}")

        # Replace -1 (unknown categories) with 0 to safely use the "unknown" index in embedding
        cat_features = torch.where(cat_features == -1, torch.tensor(0).to(self.device), cat_features)

        # Clip values to ensure they are within the valid range of the embeddings
        for i, embedding_layer in enumerate(self.embedding_layers):
            num_categories = embedding_layer.num_embeddings
            cat_features[:, i] = torch.clamp(cat_features[:, i], min=0, max=num_categories - 1)

        # Debugging to ensure the features are correctly within bounds
        if self.verbose >= 3:
            print(f"Categorical Features Tensor After Clamping: {cat_features}")

        embeddings = [embedding(cat_features[:, i]) for i, embedding in enumerate(self.embedding_layers)]
        return torch.cat(embeddings, dim=1)
    
    def configure_weight(self, layer):
        if self.weight_init =='xavier_uniform':
            nn.init.xavier_uniform_(layer.weight)
        elif self.weight_init =='kaiming_uniform':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif self.weight_init =='xavier_normal':
            nn.init.xavier_normal_(layer.weight)
        elif self.weight_init =='kaiming_normal':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)

        if self.verbose == 3:
            print(f'Layer initialization: {layer} with method {self.weight_init}')
        return layer

    def configure_optimizer(self):
        if self.optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_name == 'nadam':
            return optim.NAdam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'adamax':
            return optim.Adamax(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_name}")
        
    def configure_loss(self):
        if self.loss == 'bce':
            return nn.BCELoss()
        elif self.loss == 'bcelogit':
            return nn.BCEWithLogitsLoss()
        elif self.loss == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")

    def fit(self, X, y, eval_set = [None, None]):
        if self.verbose >= 2:
            start_time = datetime.now()
            print(f"Fitting started at {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")

        # Convert X and y to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values
        # Convert X and y to numpy arrays
        if isinstance(eval_set[0], pd.DataFrame):
            eval_set[0] = eval_set[0].values
        if isinstance(eval_set[1], pd.Series) or isinstance(eval_set[1], pd.DataFrame):
            eval_set[1] = eval_set[1].values

        # Separate categorical and numerical features
        if self.embedding_info:
            cat_train_features = X[:, :len(self.embedding_info)]
            num_train_features = X[:, len(self.embedding_info):]
            if eval_set[0] is not None:
                cat_val_features = eval_set[0][:, :len(self.embedding_info)]
                num_val_features = eval_set[0][:, len(self.embedding_info):]
            else:
                cat_val_features = None
                num_val_features = None
        else:
            cat_train_features = None
            num_train_features = X
            cat_val_features = None
            num_val_features = eval_set[0] if eval_set[0] is not None else None

        # Set input dimension for numerical features
        self.input_dim = num_train_features.shape[1] if num_train_features is not None else 0

        # Convert to tensors
        if cat_train_features is not None:
            cat_train_features_tensor = torch.tensor(cat_train_features, dtype=torch.long).to(self.device)
        num_train_features_tensor = torch.tensor(num_train_features, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        # Create validation tensors only if validation data is provided
        if num_val_features is not None and eval_set[1] is not None:
            if cat_val_features is not None:
                cat_val_features_tensor = torch.tensor(cat_val_features, dtype=torch.long).to(self.device)
            num_val_features_tensor = torch.tensor(num_val_features, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(eval_set[1], dtype=torch.float32).view(-1, 1).to(self.device)
        else:
            cat_val_features_tensor = None
            num_val_features_tensor = None
            y_val_tensor = None

        # Build datasets
        if cat_train_features is not None:
            train_dataset = TensorDataset(*(t for t in [cat_train_features_tensor, num_train_features_tensor, y_train_tensor] if t is not None))
        else:
            train_dataset = TensorDataset(*(t for t in [num_train_features_tensor, y_train_tensor] if t is not None))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Create validation loader only if validation data exists
        if num_val_features_tensor is not None and y_val_tensor is not None:
            if cat_val_features is not None:
                val_dataset = TensorDataset(*(t for t in [cat_val_features_tensor, num_val_features_tensor, y_val_tensor] if t is not None))
            else:
                val_dataset = TensorDataset(*(t for t in [num_val_features_tensor, y_val_tensor] if t is not None))
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        self.model = self.build_model().to(self.device)
        self.criterion = self.configure_loss()
        self.optimizer = self.configure_optimizer()

        # Configure scheduler only if validation is available
        if val_loader is not None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=10, verbose=True
            )
        else:
            scheduler = None

        # Early stopping parameters
        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        # Training loop
        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            all_train_outputs = []
            all_train_labels = []

            for batch in train_loader:
                if cat_train_features is not None:
                    batch_cat, batch_num, batch_y = batch
                    embedded_features = self.forward_embeddings(batch_cat)
                    inputs = torch.cat([embedded_features, batch_num], dim=1)
                else:
                    batch_num, batch_y = batch
                    inputs = batch_num

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, batch_y)

                # Store outputs and labels for metrics calculation
                if len(outputs.shape) == 1 or outputs.shape[1] == 1:  # Binary classification
                    probs = torch.sigmoid(outputs).squeeze()
                    all_train_outputs.extend(probs.detach().cpu().numpy().reshape(-1))
                else:  # Multi-class classification
                    probs = torch.softmax(outputs, dim=1)
                    all_train_outputs.extend(probs.detach().cpu().numpy())
                all_train_labels.extend(batch_y.cpu().numpy().reshape(-1))

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item() * batch_y.size(0)
            train_loss /= len(train_loader.dataset)

            # Calculate training metrics
            all_train_outputs = np.array(all_train_outputs)
            all_train_labels = np.array(all_train_labels)
            
            if len(outputs.shape) == 1 or outputs.shape[1] == 1:  # Binary classification
                train_acc = accuracy_score(all_train_labels, (all_train_outputs >= 0.5).astype(int))
                train_auc = roc_auc_score(all_train_labels, all_train_outputs)
                train_f1 = f1_score(all_train_labels, (all_train_outputs >= 0.5).astype(int), average='binary')
            else:  # Multi-class classification
                train_acc = accuracy_score(all_train_labels, np.argmax(all_train_outputs.reshape(-1, outputs.shape[1]), axis=1))
                train_auc = roc_auc_score(all_train_labels, all_train_outputs.reshape(-1, outputs.shape[1]), multi_class='ovr')
                train_f1 = f1_score(all_train_labels, np.argmax(all_train_outputs.reshape(-1, outputs.shape[1]), axis=1), average='macro')

            # Store training metrics
            self.eval_info[epoch] = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_auc': train_auc,
                'train_f1': train_f1
            }
            
            # Validation phase if validation data is available
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                all_outputs = []
                all_labels = []

                with torch.no_grad():
                    for batch in val_loader:
                        if cat_val_features is not None:
                            batch_cat, batch_num, batch_y = batch
                            embedded_features = self.forward_embeddings(batch_cat)
                            inputs = torch.cat([embedded_features, batch_num], dim=1)
                        else:
                            batch_num, batch_y = batch
                            inputs = batch_num

                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_y.size(0)

                        # Store outputs and labels for metrics calculation
                        if len(outputs.shape) == 1 or outputs.shape[1] == 1:  # Binary classification
                            probs = torch.sigmoid(outputs).squeeze()
                            all_outputs.extend(probs.cpu().numpy().reshape(-1))
                        else:  # Multi-class classification
                            probs = torch.softmax(outputs, dim=1)
                            all_outputs.extend(probs.cpu().numpy())
                        all_labels.extend(batch_y.cpu().numpy().reshape(-1))

                val_loss /= len(val_loader.dataset)

                # Calculate validation metrics
                all_outputs = np.array(all_outputs)
                all_labels = np.array(all_labels)
                
                if len(outputs.shape) == 1 or outputs.shape[1] == 1:  # Binary classification
                    val_acc = accuracy_score(all_labels, (all_outputs >= 0.5).astype(int))
                    val_auc = roc_auc_score(all_labels, all_outputs)
                    val_f1 = f1_score(all_labels, (all_outputs >= 0.5).astype(int), average='binary')
                else:  # Multi-class classification
                    val_acc = accuracy_score(all_labels, np.argmax(all_outputs.reshape(-1, outputs.shape[1]), axis=1))
                    val_auc = roc_auc_score(all_labels, all_outputs.reshape(-1, outputs.shape[1]), multi_class='ovr')
                    val_f1 = f1_score(all_labels, np.argmax(all_outputs.reshape(-1, outputs.shape[1]), axis=1), average='macro')

                # Update evaluation info with validation metrics
                self.eval_info[epoch].update({
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'val_f1': val_f1
                })

                # Scheduler step
                if scheduler is not None:
                    scheduler.step(val_loss)

                # Early stopping check using validation loss
                current_loss = val_loss
            else:
                # If no validation set, use training loss for early stopping
                current_loss = train_loss

            # Logging
            if self.verbose >= 2:
                log_msg = (f"Epoch {epoch+1}/{self.max_epochs} - "
                          f"Train Loss: {train_loss:.5f} - "
                          f"Train Acc: {train_acc:.5f} - "
                          f"Train AUC: {train_auc:.5f} - "
                          f"Train F1: {train_f1:.5f}")
                if val_loader is not None:
                    log_msg += (f" - Val Loss: {val_loss:.5f} - "
                              f"Val Acc: {val_acc:.5f} - "
                              f"Val AUC: {val_auc:.5f} - "
                              f"Val F1: {val_f1:.5f}")
                print(log_msg)

            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                patience_counter = 0
                self.best_model = self.model
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} -> best epoch {best_epoch} with {'val' if val_loader else 'train'}_loss = {best_loss:.5f}")
                    break

        # Load the best model
        self.model = self.best_model
        self.best_model = None
        self.model.eval()
        self.is_fitted_ = True

        # Store unique classes for scikit-learn compatibility
        self._classes = np.unique(y)

        if self.verbose >= 2:
            end_time = datetime.now()
            execution_time = end_time - start_time
            print(f"Fitting ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')} and took: {execution_time.total_seconds()} seconds")

    @property
    def classes_(self):
        """Return the classes labels for scikit-learn compatibility."""
        if self._classes is None:
            raise AttributeError("Model has not been fitted yet, call 'fit' before using this estimator.")
        return self._classes

    def predict_proba(self, X):
        # Convert X to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.embedding_info:
            cat_features = X[:, :len(self.embedding_info)]
            num_features = X[:, len(self.embedding_info):]
        else:
            cat_features = None
            num_features = X

        self.model.eval()
        with torch.no_grad():
            if cat_features is not None:
                cat_features_tensor = torch.tensor(cat_features, dtype=torch.long).to(self.device)
                embedded_features = self.forward_embeddings(cat_features_tensor)
            num_features_tensor = torch.tensor(num_features, dtype=torch.float32).to(self.device)

            if cat_features is not None:
                inputs = torch.cat([embedded_features, num_features_tensor], dim=1)
            else:
                inputs = num_features_tensor

            predictions = self.model(inputs).detach().cpu().numpy().flatten()
        
        if self.loss == 'bcelogit':
            predictions = 1 / (1 + np.exp(-predictions))
        
        return np.column_stack([1 - predictions, predictions])
    
    def predict(self, X):
        # Convert X to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.embedding_info:
            cat_features = X[:, :len(self.embedding_info)]
            num_features = X[:, len(self.embedding_info):]
        else:
            cat_features = None
            num_features = X

        self.model.eval()
        with torch.no_grad():
            if cat_features is not None:
                cat_features_tensor = torch.tensor(cat_features, dtype=torch.long).to(self.device)
                embedded_features = self.forward_embeddings(cat_features_tensor)
            num_features_tensor = torch.tensor(num_features, dtype=torch.float32).to(self.device)

            if cat_features is not None:
                inputs = torch.cat([embedded_features, num_features_tensor], dim=1)
            else:
                inputs = num_features_tensor

            predictions = self.model(inputs).detach().cpu().numpy().flatten()
        
        if self.loss == 'bcelogit':
            proba = 1 / (1 + np.exp(-predictions))
            return np.where(proba >= 0.5, 1, 0)

        return predictions
    
    def plot_training(self):

        # Set Seaborn style
        sns.set_style("whitegrid")  # Options: 'darkgrid', 'white', 'ticks', etc.

        # Define a cubehelix palette
        palette = sns.color_palette('GnBu')


        df = pd.DataFrame.from_dict(self.eval_info, orient='index')
        df.index.name = 'Epoch'
        df.reset_index(inplace=True)

        # Create subplots: 1 row, 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        # Plot 1: Training and Validation Loss
        sns.lineplot(
            x='Epoch', y='train_loss', data=df,
            label='Training Loss', marker='o',
            color=palette[2], linewidth=2, ax=axes[0]
        )
        if 'val_loss' in df.columns:
            sns.lineplot(
                x='Epoch', y='val_loss', data=df,
                label='Validation Loss', marker='s',
                color=palette[3], linewidth=2, ax=axes[0]
            )
        axes[0].set_title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        #axes[0].set_ylim(0, df[['train_loss', 'val_loss']].max().max() * 1.1)
        axes[0].grid(True)

        # Plot 2: Validation Accuracy and AUC-ROC
        if 'val_acc' in df.columns:
            sns.lineplot(
                x='Epoch', y='val_acc', data=df,
                label='Validation Accuracy', marker='o',
                color=palette[2], linewidth=2, ax=axes[1]
            )
        if 'val_auc' in df.columns:
            sns.lineplot(
                x='Epoch', y='val_auc', data=df,
                label='Validation AUC-ROC', marker='s',
                color=palette[3], linewidth=2, ax=axes[1]
            )
        axes[1].set_title('Validation Accuracy and AUC-ROC Over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Metric Value', fontsize=12)
        axes[1].legend(fontsize=10)
        #axes[1].set_ylim(0, 1)
        axes[1].grid(True)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the combined plots
        plt.show()

    def _encode_labels(self, y):
        self.label_mapping_ = {label: idx for idx, label in enumerate(self.classes_)}
        return np.array([self.label_mapping_[label] for label in y])
    
# PyTorch Regressor
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        learning_rate=0.001,
        optimizer_name='adam',
        batch_size=32,
        max_epochs=1000,
        patience=10,
        net=[(1, 256, 256, 1, 0), (1, 256, 1, 0, 0)],
        embedding_info=None,
        loss='mse',
        weight_init = 'default',
        verbose=1,
        device = 'cpu',
        num_threads = os.cpu_count(),
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.model = None
        self.best_model = None
        self.device = device
        self.input_dim = None
        self.net = net
        self.is_fitted_ = False
        self.optimizer_name = optimizer_name
        self.optimizer = None
        self.criterion = None
        self.verbose = verbose
        self.embedding_info = embedding_info  # Embedding info for categorical features
        self.loss = loss
        self.eval_info = {}
        self.weight_init = weight_init
        self.num_threads = num_threads

        if self.device == 'cpu':
            torch.set_num_threads(self.num_threads)

    def build_model(self):
        layers = []
        embedding_layers = []

        # Add embedding layers for high cardinality categorical features if embedding_info is provided
        total_embedding_dim = 0
        if self.embedding_info:
            for num_categories, embedding_dim in self.embedding_info.values():
                embedding_layer = nn.Embedding(num_categories, embedding_dim)
                embedding_layers.append(embedding_layer.to(self.device))
                total_embedding_dim += embedding_dim

        self.embedding_layers = nn.ModuleList(embedding_layers)

        # First layer: concatenate embeddings and numerical input
        first_layer_input_dim = total_embedding_dim + (self.input_dim if self.input_dim is not None else 0)

        # Define the rest of the network
        if self.net[0][0] == 1:
            l = nn.Linear(first_layer_input_dim, self.net[0][2])
            l = self.configure_weight(l)
            layers.append(l)
           
        if self.net[0][4] == 1:
            layers.append(nn.BatchNorm1d(self.net[0][2]))
        elif self.net[0][4] == 2:
            layers.append(nn.LayerNorm(self.net[0][2]))

        if self.net[0][3] == 1:
            layers.append(nn.ReLU())
        elif self.net[0][3] == 2:
            layers.append(nn.LeakyReLU(negative_slope=0.01))
        elif self.net[0][3] == 3:
            layers.append(nn.Sigmoid())
        elif self.net[0][3] == 4:
            layers.append(nn.Tanh())
        elif self.net[0][3] == 5:
            layers.append(nn.SiLU())
        elif self.net[0][3] == 6:
            layers.append(nn.Softmax())
        elif self.net[0][3] == 7:
            layers.append(nn.ELU(alpha=1.0))
        elif self.net[0][3] == 8: 
            layers.append(nn.SELU())
        elif self.net[0][3] == 0:
            pass  # No activation

        # Iterate over the rest of the net configuration
        for layer in self.net[1:]:
            # Add layer
            if layer[0] == 1:
                # Dense
                l = nn.Linear(layer[1], layer[2])
                l = self.configure_weight(l)
                layers.append(l)

                # Add normalization
                if layer[4] == 1:
                    layers.append(nn.BatchNorm1d(layer[2]))
                elif layer[4] == 2:
                    layers.append(nn.LayerNorm(layer[2]))

                # Add activation function
                if layer[3] == 1:
                    layers.append(nn.ReLU())
                elif layer[3] == 2:
                    layers.append(nn.LeakyReLU(negative_slope=0.01))
                elif layer[3] == 3:
                    layers.append(nn.Sigmoid())
                elif layer[3] == 4:
                    layers.append(nn.Tanh())
                elif layer[3] == 5:
                    layers.append(nn.SiLU())
                elif layer[3] == 6:
                    layers.append(nn.Softmax())
                elif layer[3] == 7:
                    layers.append(nn.ELU(alpha=1.0))
                elif layer[3] == 8:
                    layers.append(nn.SELU())
                elif layer[3] == 0:
                    pass  # No activation

            elif layer[0] == 0:
                # Dropout
                layers.append(nn.Dropout(layer[1]))

        if self.verbose >= 2:
            print(f"Embedding layers: {embedding_layers}")
            print(f"Layers: {layers}")

        return nn.Sequential(*layers).to(self.device)

    def forward_embeddings(self, cat_features):
        if not self.embedding_layers:
            return None
        
        # Debugging to print the range of cat_features
        if self.verbose >= 3:
            print(f"Categorical Features Tensor Before Clamping: {cat_features}")

        # Replace -1 (unknown categories) with 0 to safely use the "unknown" index in embedding
        cat_features = torch.where(cat_features == -1, torch.tensor(0).to(self.device), cat_features)

        # Clip values to ensure they are within the valid range of the embeddings
        for i, embedding_layer in enumerate(self.embedding_layers):
            num_categories = embedding_layer.num_embeddings
            cat_features[:, i] = torch.clamp(cat_features[:, i], min=0, max=num_categories - 1)

        # Debugging to ensure the features are correctly within bounds
        if self.verbose >= 3:
            print(f"Categorical Features Tensor After Clamping: {cat_features}")

        embeddings = [embedding(cat_features[:, i]) for i, embedding in enumerate(self.embedding_layers)]
        return torch.cat(embeddings, dim=1)

    def configure_weight(self, layer):
        if self.weight_init =='xavier_uniform':
            nn.init.xavier_uniform_(layer.weight)
        elif self.weight_init =='kaiming_uniform':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        elif self.weight_init =='xavier_normal':
            nn.init.xavier_normal_(layer.weight)
        elif self.weight_init =='kaiming_normal':
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)

        if self.verbose == 3:
            print(f'Layer initialization: {layer} with method {self.weight_init}')
        return layer

    def configure_optimizer(self):
        if self.optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_name == 'nadam':
            return optim.NAdam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'adamax':
            return optim.Adamax(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_name}")

    def configure_loss(self):
        if self.loss == 'mse':
            return nn.MSELoss()
        elif self.loss == 'mae':
            return nn.L1Loss()
        elif self.loss == 'huber':
            return nn.SmoothL1Loss()
        elif self.loss == 'rmsle':
            return RMSLELoss()
        elif self.loss == 'mape':
            return MAPE_Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")

    def fit(self, X, y, eval_set = [None, None]):
        if self.verbose >= 2:
            start_time = datetime.now()
            print(f"Fitting started at {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")

        # Convert X and y to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values

        # Separate categorical and numerical features
        if self.embedding_info:
            cat_train_features = X[:, :len(self.embedding_info)]
            num_train_features = X[:, len(self.embedding_info):]
            if eval_set[0] is not None:
                cat_val_features = eval_set[0][:, :len(self.embedding_info)]
                num_val_features = eval_set[0][:, len(self.embedding_info):]
            else:
                cat_val_features = None
                num_val_features = None
        else:
            cat_train_features = None
            num_train_features = X
            cat_val_features = None
            num_val_features = eval_set[0] if eval_set[0] is not None else None

        # Set input dimension for numerical features
        self.input_dim = num_train_features.shape[1] if num_train_features is not None else 0

        # Convert to tensors
        if cat_train_features is not None:
            cat_train_features_tensor = torch.tensor(cat_train_features, dtype=torch.long).to(self.device)
        num_train_features_tensor = torch.tensor(num_train_features, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        # Create validation tensors only if validation data is provided
        if num_val_features is not None and eval_set[1] is not None:
            if cat_val_features is not None:
                cat_val_features_tensor = torch.tensor(cat_val_features, dtype=torch.long).to(self.device)
            num_val_features_tensor = torch.tensor(num_val_features, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(eval_set[1].values, dtype=torch.float32).view(-1, 1).to(self.device)
        else:
            cat_val_features_tensor = None
            num_val_features_tensor = None
            y_val_tensor = None

        # Build datasets
        if cat_train_features is not None:
            train_dataset = TensorDataset(*(t for t in [cat_train_features_tensor, num_train_features_tensor, y_train_tensor] if t is not None))
        else:
            train_dataset = TensorDataset(*(t for t in [num_train_features_tensor, y_train_tensor] if t is not None))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Create validation loader only if validation data exists
        if num_val_features_tensor is not None and y_val_tensor is not None:
            if cat_val_features is not None:
                val_dataset = TensorDataset(*(t for t in [cat_val_features_tensor, num_val_features_tensor, y_val_tensor] if t is not None))
            else:
                val_dataset = TensorDataset(*(t for t in [num_val_features_tensor, y_val_tensor] if t is not None))
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        self.model = self.build_model().to(self.device)
        self.criterion = self.configure_loss()
        self.optimizer = self.configure_optimizer()

        # Configure scheduler only if validation is available
        if val_loader is not None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=10, verbose=True
            )
        else:
            scheduler = None

        # Early stopping parameters
        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        # Training loop
        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            all_train_outputs = []
            all_train_labels = []

            for batch in train_loader:
                if cat_train_features is not None:
                    batch_cat, batch_num, batch_y = batch
                    embedded_features = self.forward_embeddings(batch_cat)
                    inputs = torch.cat([embedded_features, batch_num], dim=1)
                else:
                    batch_num, batch_y = batch
                    inputs = batch_num

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, batch_y)

                # Store outputs and labels for metrics calculation
                all_train_outputs.extend(outputs.detach().cpu().numpy().flatten())
                all_train_labels.extend(batch_y.cpu().numpy().flatten())

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item() * batch_y.size(0)
            train_loss /= len(train_loader.dataset)

            # Calculate training metrics
            train_rmse = root_mean_squared_error(all_train_labels, all_train_outputs)
            train_mae = mean_absolute_error(all_train_labels, all_train_outputs)
            train_r2 = r2_score(all_train_labels, all_train_outputs)
            train_mape = mean_absolute_percentage_error(all_train_labels, all_train_outputs)

            all_train_outputs_n = np.maximum(all_train_outputs, 0)
            train_rmsle = root_mean_squared_log_error(all_train_labels, all_train_outputs_n)

            # Store training metrics
            self.eval_info[epoch] = {
                'train_loss': train_loss,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_mape': train_mape,
                'train_rmsle': train_rmsle,
                'train_r2': train_r2
            }
            
            # Validation phase if validation data is available
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                all_outputs = []
                all_labels = []

                with torch.no_grad():
                    for batch in val_loader:
                        if cat_val_features is not None:
                            batch_cat, batch_num, batch_y = batch
                            embedded_features = self.forward_embeddings(batch_cat)
                            inputs = torch.cat([embedded_features, batch_num], dim=1)
                        else:
                            batch_num, batch_y = batch
                            inputs = batch_num

                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_y.size(0)

                        all_outputs.extend(outputs.cpu().numpy().flatten())
                        all_labels.extend(batch_y.cpu().numpy().flatten())

                val_loss /= len(val_loader.dataset)

                # Compute validation metrics
                val_rmse = root_mean_squared_error(all_labels, all_outputs)
                val_mae = mean_absolute_error(all_labels, all_outputs)
                val_r2 = r2_score(all_labels, all_outputs)
                val_mape = mean_absolute_percentage_error(all_labels, all_outputs)

                all_outputs_n = np.maximum(all_outputs, 0)
                val_rmsle = root_mean_squared_log_error(all_labels, all_outputs_n)

                # Update evaluation info with validation metrics
                self.eval_info[epoch].update({
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'val_mape': val_mape,
                    'val_rmsle': val_rmsle,
                    'val_r2': val_r2,
                })

                # Scheduler step
                if scheduler is not None:
                    scheduler.step(val_loss)

                # Early stopping check using validation loss
                current_loss = val_loss
            else:
                # If no validation set, use training loss for early stopping
                current_loss = train_loss

            # Logging
            if self.verbose >= 2:
                log_msg = (f"Epoch {epoch+1}/{self.max_epochs} - "
                          f"Train Loss: {train_loss:.5f} - "
                          f"Train RMSE: {train_rmse:.5f} - "
                          f"Train MAE: {train_mae:.5f} - "
                          f"Train MAPE: {train_mape:.5f} - "
                          f"Train RMSLE: {train_rmsle:.5f} - "
                          f"Train R2: {train_r2:.5f}")
                if val_loader is not None:
                    log_msg += (f" - Val Loss: {val_loss:.5f} - "
                              f"Val RMSE: {val_rmse:.5f} - "
                              f"Val MAE: {val_mae:.5f} - "
                              f"Val MAPE: {val_mape:.5f} - "
                              f"Val RMSLE: {val_rmsle:.5f} - "
                              f"Val R2: {val_r2:.5f}")
                print(log_msg)

            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                patience_counter = 0
                self.best_model = self.model
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} -> best epoch {best_epoch} with {'val' if val_loader else 'train'}_loss = {best_loss:.5f}")
                    break

        # Load the best model
        self.model = self.best_model
        self.best_model = None
        self.model.eval()
        self.is_fitted_ = True

        if self.verbose >= 2:
            end_time = datetime.now()
            execution_time = end_time - start_time
            print(
                f"Fitting ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')} "
                f"and took: {execution_time.total_seconds()} seconds"
            )

        return self

    def predict(self, X):
        # Convert X to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.embedding_info:
            cat_features = X[:, :len(self.embedding_info)]
            num_features = X[:, len(self.embedding_info):]
        else:
            cat_features = None
            num_features = X

        self.model.eval()
        with torch.no_grad():
            if cat_features is not None:
                cat_features_tensor = torch.tensor(cat_features, dtype=torch.long).to(self.device)
                embedded_features = self.forward_embeddings(cat_features_tensor)
            num_features_tensor = torch.tensor(num_features, dtype=torch.float32).to(self.device)

            if cat_features is not None:
                inputs = torch.cat([embedded_features, num_features_tensor], dim=1)
            else:
                inputs = num_features_tensor

            predictions = self.model(inputs).detach().cpu().numpy().flatten()

        return predictions

    def plot_training(self):
      
        # Set Seaborn style
        sns.set_style("whitegrid")

        # Define a color palette
        palette = sns.color_palette('GnBu')

        df = pd.DataFrame.from_dict(self.eval_info, orient='index')
        df.index.name = 'Epoch'
        df.reset_index(inplace=True)

        # Create subplots: 1 row, 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        # Plot 1: Training and Validation Loss
        sns.lineplot(
            x='Epoch',
            y='train_loss',
            data=df,
            label='Training Loss',
            marker='o',
            color=palette[2],
            linewidth=2,
            ax=axes[0],
        )
        if 'val_loss' in df.columns:
            sns.lineplot(
                x='Epoch',
                y='val_loss',
                data=df,
                label='Validation Loss',
                marker='s',
                color=palette[3],
                linewidth=2,
                ax=axes[0],
            )
        axes[0].set_title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True)

        # Plot 2: Validation Metrics
        if 'val_rmse' in df.columns:
            sns.lineplot(
                x='Epoch',
                y='val_rmse',
                data=df,
                label='Validation RMSE',
                marker='o',
                color=palette[2],
                linewidth=2,
                ax=axes[1],
            )
        if 'val_mae' in df.columns:
            sns.lineplot(
                x='Epoch',
                y='val_mae',
                data=df,
                label='Validation MAE',
                marker='s',
                color=palette[3],
                linewidth=2,
                ax=axes[1],
            )
        if 'val_r2' in df.columns:
            sns.lineplot(
                x='Epoch',
                y='val_r2',
                data=df,
                label='Validation R²',
                marker='^',
                color=palette[4],
                linewidth=2,
                ax=axes[1],
            )
        axes[1].set_title('Validation Metrics Over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Metric Value', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the combined plots
        plt.show()