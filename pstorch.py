import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score, f1_score
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


#pytorch classifier
class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.001, optimizer_name='adam', batch_size=32, max_epochs=1000, patience=10, net=[(1, 256, 256, 1, 0),(1, 256, 1, 0, 0)], embedding_info=None,  loss='mse', verbose=1):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.model = None
        self.best_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            layers.append(nn.Linear(first_layer_input_dim, self.net[0][2]))
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


        # Iterate over the rest of the net configuration
        for layer in self.net[1:]:
            #add layer
            if layer[0] == 1:
                #Dense
                layers.append(nn.Linear(layer[1], layer[2]))
                
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
        elif self.loss == 'bce':
            return nn.BCELoss()
        elif self.loss == 'bcelogit':
            return nn.BCEWithLogitsLoss()
        elif self.loss == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")

    def fit(self, X, y):
        if self.verbose == 2:
            start_time = datetime.now()
            print(f"Fitting started at {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        
        # Convert X and y to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values

        self.classes_ = np.unique(y)
        #y = self._encode_labels(y)
            
        # Separate categorical and numerical features
        if self.embedding_info:
            cat_features = X[:, :len(self.embedding_info)]
            num_features = X[:, len(self.embedding_info):]
        else:
            cat_features = None
            num_features = X

        # Set input dimension for numerical features
        self.input_dim = num_features.shape[1] if num_features is not None else 0

        # Convert to tensors
        if cat_features is not None:
            cat_features_tensor = torch.tensor(cat_features, dtype=torch.long).to(self.device)
        num_features_tensor = torch.tensor(num_features, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # Split into training and validation sets
        if cat_features is not None:
            cat_train, cat_val, num_train, num_val, y_train, y_val = train_test_split(cat_features_tensor, num_features_tensor, y_tensor, test_size=0.2, random_state=42, shuffle=True, stratify=y)
        else:
            num_train, num_val, y_train, y_val = train_test_split(num_features_tensor, y_tensor, test_size=0.2, random_state=42, shuffle=True, stratify=y)
            cat_train, cat_val = None, None

        train_dataset = TensorDataset(*(t for t in [cat_train, num_train, y_train] if t is not None))
        val_dataset = TensorDataset(*(t for t in [cat_val, num_val, y_val] if t is not None))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = self.build_model().to(self.device)
        
        self.criterion = self.configure_loss()
        
        self.optimizer = self.configure_optimizer()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        # Early stopping parameters
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                if cat_features is not None:
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
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item() * batch_y.size(0)
            train_loss /= len(train_loader.dataset)
            

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    if cat_features is not None:
                        batch_cat, batch_num, batch_y = batch
                        embedded_features = self.forward_embeddings(batch_cat)
                        inputs = torch.cat([embedded_features, batch_num], dim=1)
                    else:
                        batch_num, batch_y = batch
                        inputs = batch_num
                    
                    outputs = self.model(inputs)  # Assuming outputs are raw logits
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_y.size(0)
                    
                    # If it's a binary classification
                    if outputs.shape[1] == 1:
                        probs = torch.sigmoid(outputs).squeeze()
                        preds = (probs >= 0.5).long()
                        all_probs.extend(probs.cpu().numpy())
                    else:
                        # Multi-class classification
                        probs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        all_probs.extend(probs.cpu().numpy())
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())

            val_loss /= len(val_loader.dataset)

            # Calculate Accuracy
            val_acc = accuracy_score(all_labels, all_preds)

            # Calculate AUC-ROC
            # Handle binary and multi-class cases
            if outputs.shape[1] == 1:
                # Binary classification
                val_auc = roc_auc_score(all_labels, all_probs)
                val_f1 = f1_score(all_labels, all_preds, average='binary')
            else:
                # Multi-class classification
                val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
                val_f1 = f1_score(all_labels, all_preds, average='macro')

            # Scheduler step
            scheduler.step(val_loss)

            self.eval_info[epoch] = {'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc, 'val_f1': val_f1}

            # Logging
            if self.verbose >= 2:
                print(f"Epoch {epoch+1}/{self.max_epochs} - Loss(train: {train_loss:.5f} val: {val_loss:.5f}) - Val(acc: {val_acc:.5f} roc-auc: {val_auc:.5f} f1: {val_f1:.5f})")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                #torch.save(self.model.state_dict(), 'best_model.pth')
                self.best_model = self.model
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

            # Optionally, you can also keep track of best metrics if needed

            # Load the best model
        #self.model.load_state_dict(torch.load('best_model.pth'))
        self.model = self.best_model
        self.best_model = None
        self.model.eval()
        self.is_fitted_ = True


        if self.verbose == 2:
            end_time = datetime.now()
            execution_time = end_time - start_time
            print(f"Fitting ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')} and take: {execution_time.total_seconds()} seconds")

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
            return 1 / (1 + np.exp(-predictions))

        return predictions
    
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
        sns.lineplot(
            x='Epoch', y='val_acc', data=df,
            label='Validation Accuracy', marker='o',
            color=palette[2], linewidth=2, ax=axes[1]
        )
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
    