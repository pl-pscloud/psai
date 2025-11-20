import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
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

def get_activation(act_name):
    if not act_name: return None
    act_name = str(act_name).lower()
    if act_name == 'relu': return nn.ReLU()
    elif act_name == 'leaky_relu': return nn.LeakyReLU(negative_slope=0.01)
    elif act_name == 'sigmoid': return nn.Sigmoid()
    elif act_name == 'tanh': return nn.Tanh()
    elif act_name == 'silu' or act_name == 'swish': return nn.SiLU()
    elif act_name == 'softmax': return nn.Softmax(dim=1)
    elif act_name == 'elu': return nn.ELU(alpha=1.0)
    elif act_name == 'selu': return nn.SELU()
    elif act_name == 'gelu': return nn.GELU()
    return None

def get_norm(norm_name, dim):
    if not norm_name: return None
    norm_name = str(norm_name).lower()
    if norm_name == 'batch_norm': return nn.BatchNorm1d(dim)
    elif norm_name == 'layer_norm': return nn.LayerNorm(dim)
    return None

def initialize_weights(layer, weight_init='default', verbose=0):
    if weight_init == 'default':
        return layer
        
    if weight_init =='xavier_uniform':
        nn.init.xavier_uniform_(layer.weight)
    elif weight_init =='kaiming_uniform':
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    elif weight_init =='xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    elif weight_init =='kaiming_normal':
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

    if verbose >= 3:
        print(f'Layer initialization: {layer} with method {weight_init}')
    return layer

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', norm=None, dropout=0.0, weight_init='default'):
        super(ResidualBlock, self).__init__()
        
        self.lin1 = nn.Linear(in_features, out_features)
        initialize_weights(self.lin1, weight_init)
        
        self.norm1 = get_norm(norm, out_features) if norm else nn.Identity()
        self.act1 = get_activation(activation) if activation else nn.Identity()
        self.drop = nn.Dropout(dropout)
        
        self.lin2 = nn.Linear(out_features, out_features)
        initialize_weights(self.lin2, weight_init)
        
        self.norm2 = get_norm(norm, out_features) if norm else nn.Identity()
        
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            initialize_weights(self.shortcut, weight_init)
        else:
            self.shortcut = nn.Identity()
            
        self.act2 = get_activation(activation) if activation else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.lin1(x)
        if not isinstance(self.norm1, nn.Identity): out = self.norm1(out)
        if not isinstance(self.act1, nn.Identity): out = self.act1(out)
        out = self.drop(out)
        
        out = self.lin2(out)
        if not isinstance(self.norm2, nn.Identity): out = self.norm2(out)
        
        out += residual
        if not isinstance(self.act2, nn.Identity): out = self.act2(out)
        
        return out

class TabularMLP(nn.Module):
    def __init__(self, embedding_info, n_num_features, layers_config, weight_init='default', verbose=0):
        super(TabularMLP, self).__init__()
        self.verbose = verbose
        self.embedding_layers = nn.ModuleList()
        total_embedding_dim = 0
        
        if embedding_info:
            for num_categories, embedding_dim in embedding_info.values():
                self.embedding_layers.append(nn.Embedding(num_categories, embedding_dim))
                total_embedding_dim += embedding_dim
                
        current_input_dim = total_embedding_dim + n_num_features
        
        layers = []
        for layer_config in layers_config:
            layer_type = layer_config.get('type', 'dense')
            
            if layer_type == 'dense':
                out_features = layer_config['out_features']
                l = nn.Linear(current_input_dim, out_features)
                initialize_weights(l, weight_init, verbose)
                layers.append(l)
                
                norm = get_norm(layer_config.get('norm'), out_features)
                if norm: layers.append(norm)
                
                act = get_activation(layer_config.get('activation'))
                if act: layers.append(act)
                
                current_input_dim = out_features
                
            elif layer_type == 'residual':
                out_features = layer_config['out_features']
                res_block = ResidualBlock(
                    in_features=current_input_dim,
                    out_features=out_features,
                    activation=layer_config.get('activation', 'relu'),
                    norm=layer_config.get('norm', None),
                    dropout=layer_config.get('dropout', 0.0),
                    weight_init=weight_init
                )
                layers.append(res_block)
                current_input_dim = out_features

            elif layer_type == 'dropout':
                p = layer_config.get('p', 0.5)
                layers.append(nn.Dropout(p))
                
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embeddings = []
        if len(self.embedding_layers) > 0 and x_cat is not None:
            # Replace -1 (unknown categories) with 0
            x_cat = torch.where(x_cat == -1, torch.tensor(0, device=x_cat.device), x_cat)
            
            for i, layer in enumerate(self.embedding_layers):
                # Clip values to be within range
                num_categories = layer.num_embeddings
                x_col = torch.clamp(x_cat[:, i], min=0, max=num_categories - 1)
                embeddings.append(layer(x_col))
            
            x_emb = torch.cat(embeddings, dim=1)
            if x_num is not None:
                x = torch.cat([x_emb, x_num], dim=1)
            else:
                x = x_emb
        else:
            x = x_num
            
        return self.mlp(x)

class FTTransformer(nn.Module):
    def __init__(
        self, 
        embedding_info, 
        n_num_features, 
        d_token=192, 
        n_layers=3, 
        n_heads=8, 
        d_ffn_factor=1.33, 
        attention_dropout=0.1, 
        ffn_dropout=0.1, 
        residual_dropout=0.0, 
        activation='reglu',
        n_out=1
    ):
        super(FTTransformer, self).__init__()
        
        self.d_token = d_token
        self.embedding_layers = nn.ModuleList()
        
        # Categorical embeddings
        if embedding_info:
            for num_categories, _ in embedding_info.values():
                # We ignore the embedding_dim from info and use d_token
                self.embedding_layers.append(nn.Embedding(num_categories, d_token))
        
        # Numerical embeddings (linear layer to project to d_token)
        self.n_num_features = n_num_features
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=int(d_token * d_ffn_factor),
            dropout=attention_dropout, # Note: PyTorch's dropout arg applies to both self-attn and FFN in some versions, but mainly after attn
            activation=activation if activation != 'reglu' else 'relu', # PyTorch doesn't support reglu natively in older versions
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, n_out)
        )

    def forward(self, x_cat, x_num):
        batch_size = x_cat.shape[0] if x_cat is not None else x_num.shape[0]
        tokens = []
        
        # CLS Token
        tokens.append(self.cls_token.expand(batch_size, -1, -1))
        
        # Categorical embeddings
        if len(self.embedding_layers) > 0 and x_cat is not None:
             # Replace -1 (unknown categories) with 0
            x_cat = torch.where(x_cat == -1, torch.tensor(0, device=x_cat.device), x_cat)
            
            for i, layer in enumerate(self.embedding_layers):
                num_categories = layer.num_embeddings
                x_col = torch.clamp(x_cat[:, i], min=0, max=num_categories - 1)
                tokens.append(layer(x_col).unsqueeze(1))
                
        # Numerical embeddings
        if self.n_num_features > 0 and x_num is not None:
             # We will use a ModuleList of Linear layers for numerical features
             if not hasattr(self, 'num_embeddings_list'):
                 self.num_embeddings_list = nn.ModuleList([nn.Linear(1, self.d_token) for _ in range(self.n_num_features)]).to(x_num.device)
             
             for i in range(self.n_num_features):
                 tokens.append(self.num_embeddings_list[i](x_num[:, i].unsqueeze(-1)).unsqueeze(1))

        x = torch.cat(tokens, dim=1) # (batch, 1 + n_cat + n_num, d_token)
        
        x = self.transformer(x)
        
        # Use CLS token for prediction
        x_cls = x[:, 0, :]
        return self.head(x_cls)


class PyTorchBaseEstimator(BaseEstimator):
    def __init__(
            self, 
            learning_rate=0.001, 
            optimizer_name='adam', 
            batch_size=32, 
            max_epochs=1000, 
            patience=10, 
            model_type='mlp', # 'mlp' or 'ft_transformer'
            net=None, # For MLP
            ft_params=None, # For FT-Transformer
            embedding_info=None,  
            loss=None, 
            verbose=1,
            weight_init='default',
            device='cpu',
            num_threads=None,
        ):
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.max_epochs = max_epochs
            self.patience = patience
            self.model = None
            self.best_model = None
            self.device = device
            self.input_dim = None
            self.model_type = model_type
            
            # Default net structure if None provided
            self.net = net if net is not None else [
                {'type': 'dense', 'out_features': 256, 'activation': 'relu', 'norm': None},
                {'type': 'dense', 'out_features': 1, 'activation': None, 'norm': None}
            ]
            self.ft_params = ft_params if ft_params is not None else {}
            
            self.is_fitted_ = False
            self.optimizer_name = optimizer_name
            self.optimizer = None
            self.criterion = None
            self.verbose = verbose
            self.embedding_info = embedding_info
            self.loss = loss
            self.eval_info = {}
            self.weight_init = weight_init
            self.num_threads = num_threads if num_threads is not None else os.cpu_count()

            if self.device == 'cpu':
                torch.set_num_threads(self.num_threads)

    def build_model(self):
        n_num_features = self.input_dim
        
        if self.model_type == 'mlp':
            return TabularMLP(
                embedding_info=self.embedding_info,
                n_num_features=n_num_features,
                layers_config=self.net,
                weight_init=self.weight_init,
                verbose=self.verbose
            )
        elif self.model_type == 'ft_transformer':
            return FTTransformer(
                embedding_info=self.embedding_info,
                n_num_features=n_num_features,
                **self.ft_params
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def configure_optimizer(self):
        params = self.model.parameters()
        if self.optimizer_name == 'adam': return optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer_name == 'rmsprop': return optim.RMSprop(params, lr=self.learning_rate)
        elif self.optimizer_name == 'sgd': return optim.SGD(params, lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_name == 'nadam': return optim.NAdam(params, lr=self.learning_rate)
        elif self.optimizer_name == 'adamax': return optim.Adamax(params, lr=self.learning_rate)
        elif self.optimizer_name == 'adamw': return optim.AdamW(params, lr=self.learning_rate)
        else: raise ValueError(f"Unsupported optimizer type: {self.optimizer_name}")

    def configure_loss(self):
        raise NotImplementedError("Subclasses must implement configure_loss")

    def _calculate_metrics(self, y_true, y_pred_raw, epoch_loss, phase='train'):
        raise NotImplementedError("Subclasses must implement _calculate_metrics")

    def _prepare_data(self, X, y=None):
        if isinstance(X, pd.DataFrame): X = X.values
        if y is not None and (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)): y = y.values
        return X, y

    def fit(self, X, y, eval_set=[None, None]):
        if self.verbose >= 2:
            start_time = datetime.now()
            print(f"Fitting started at {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")

        X, y = self._prepare_data(X, y)
        
        # Handle eval_set
        eval_X, eval_y = eval_set
        eval_X, eval_y = self._prepare_data(eval_X, eval_y)
        
        # Separate categorical and numerical features
        if self.embedding_info:
            cat_train_features = X[:, :len(self.embedding_info)]
            num_train_features = X[:, len(self.embedding_info):]
            if eval_X is not None:
                cat_val_features = eval_X[:, :len(self.embedding_info)]
                num_val_features = eval_X[:, len(self.embedding_info):]
            else:
                cat_val_features = None
                num_val_features = None
        else:
            cat_train_features = None
            num_train_features = X
            cat_val_features = None
            num_val_features = eval_X

        self.input_dim = num_train_features.shape[1] if num_train_features is not None else 0

        # To tensors
        def to_tensor(data, dtype):
            return torch.tensor(data, dtype=dtype).to(self.device) if data is not None else None

        cat_train_tensor = to_tensor(cat_train_features, torch.long)
        num_train_tensor = to_tensor(num_train_features, torch.float32)
        y_train_tensor = to_tensor(y, torch.float32).view(-1, 1)

        cat_val_tensor = to_tensor(cat_val_features, torch.long)
        num_val_tensor = to_tensor(num_val_features, torch.float32)
        y_val_tensor = to_tensor(eval_y, torch.float32).view(-1, 1) if eval_y is not None else None

        # Datasets
        # We need to be careful with TensorDataset if some tensors are None
        # We'll always pass both cat and num to the model, even if None (handled inside model)
        # But TensorDataset needs actual tensors.
        # Let's create a custom dataset or just handle the loop carefully.
        # Simplest: Create dummy tensors if None? No, waste of memory.
        # Let's just use what we have and unpack in the loop.
        
        tensors_to_pass = []
        if cat_train_tensor is not None: tensors_to_pass.append(cat_train_tensor)
        if num_train_tensor is not None: tensors_to_pass.append(num_train_tensor)
        tensors_to_pass.append(y_train_tensor)
        
        train_dataset = TensorDataset(*tensors_to_pass)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if (num_val_tensor is not None or cat_val_tensor is not None) and y_val_tensor is not None:
            val_tensors = []
            if cat_val_tensor is not None: val_tensors.append(cat_val_tensor)
            if num_val_tensor is not None: val_tensors.append(num_val_tensor)
            val_tensors.append(y_val_tensor)
            val_dataset = TensorDataset(*val_tensors)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = self.build_model().to(self.device)
        self.criterion = self.configure_loss()
        self.optimizer = self.configure_optimizer()

        scheduler = None
        if val_loader is not None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=10
            )

        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            all_train_outputs = []
            all_train_labels = []

            for batch in train_loader:
                # Unpack batch based on what we put in
                if cat_train_tensor is not None and num_train_tensor is not None:
                    batch_cat, batch_num, batch_y = batch
                elif cat_train_tensor is not None:
                    batch_cat, batch_y = batch
                    batch_num = None
                else:
                    batch_num, batch_y = batch
                    batch_cat = None

                self.optimizer.zero_grad()
                outputs = self.model(batch_cat, batch_num)
                loss = self.criterion(outputs, batch_y)
                
                # Store for metrics
                all_train_outputs.append(outputs.detach().cpu())
                all_train_labels.append(batch_y.cpu())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item() * batch_y.size(0)
            
            train_loss /= len(train_loader.dataset)
            all_train_outputs = torch.cat(all_train_outputs).numpy()
            all_train_labels = torch.cat(all_train_labels).numpy()
            
            train_metrics = self._calculate_metrics(all_train_labels, all_train_outputs, train_loss, phase='train')
            self.eval_info[epoch] = train_metrics

            # Validation
            current_loss = train_loss
            val_metrics = {}
            
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                all_val_outputs = []
                all_val_labels = []

                with torch.no_grad():
                    for batch in val_loader:
                        if cat_val_tensor is not None and num_val_tensor is not None:
                            batch_cat, batch_num, batch_y = batch
                        elif cat_val_tensor is not None:
                            batch_cat, batch_y = batch
                            batch_num = None
                        else:
                            batch_num, batch_y = batch
                            batch_cat = None

                        outputs = self.model(batch_cat, batch_num)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_y.size(0)

                        all_val_outputs.append(outputs.cpu())
                        all_val_labels.append(batch_y.cpu())

                val_loss /= len(val_loader.dataset)
                all_val_outputs = torch.cat(all_val_outputs).numpy()
                all_val_labels = torch.cat(all_val_labels).numpy()

                val_metrics = self._calculate_metrics(all_val_labels, all_val_outputs, val_loss, phase='val')
                self.eval_info[epoch].update(val_metrics)
                
                if scheduler is not None:
                    scheduler.step(val_loss)
                current_loss = val_loss

            # Logging
            if self.verbose >= 2:
                log_msg = f"Epoch {epoch+1}/{self.max_epochs} - " + " - ".join([f"{k}: {v:.5f}" for k, v in train_metrics.items()])
                if val_metrics:
                    log_msg += " - " + " - ".join([f"{k}: {v:.5f}" for k, v in val_metrics.items()])
                print(log_msg)

            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                patience_counter = 0
                self.best_model = self.model # Note: this is a reference, might want deepcopy if model changes later but here it's fine
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} -> best epoch {best_epoch} with loss = {best_loss:.5f}")
                    break

        self.model = self.best_model
        self.best_model = None
        self.model.eval()
        self.is_fitted_ = True
        
        # Store classes if classifier
        if hasattr(self, '_classes') and self._classes is None:
             self._classes = np.unique(y)

        if self.verbose >= 2:
            end_time = datetime.now()
            execution_time = end_time - start_time
            print(f"Fitting ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')} and took: {execution_time.total_seconds()} seconds")
        
        return self

    def plot_training(self):
        sns.set_style("whitegrid")
        palette = sns.color_palette('GnBu')
        
        df = pd.DataFrame.from_dict(self.eval_info, orient='index')
        df.index.name = 'Epoch'
        df.reset_index(inplace=True)
        
        # Determine metrics to plot
        metrics = [col for col in df.columns if col.startswith('train_') and col != 'train_loss']
        val_metrics = [col for col in df.columns if col.startswith('val_') and col != 'val_loss']
        
        n_plots = 2
        fig, axes = plt.subplots(1, n_plots, figsize=(20, 6))
        
        # Plot Loss
        sns.lineplot(x='Epoch', y='train_loss', data=df, label='Training Loss', marker='o', color=palette[2], ax=axes[0])
        if 'val_loss' in df.columns:
            sns.lineplot(x='Epoch', y='val_loss', data=df, label='Validation Loss', marker='s', color=palette[3], ax=axes[0])
        axes[0].set_title('Loss Over Epochs')
        
        # Plot other metrics (just taking the first one found for simplicity or all)
        # Actually let's just plot the first non-loss metric found for train/val
        if metrics:
            metric_name = metrics[0]
            sns.lineplot(x='Epoch', y=metric_name, data=df, label=metric_name, marker='o', color=palette[2], ax=axes[1])
            val_metric_name = metric_name.replace('train_', 'val_')
            if val_metric_name in df.columns:
                sns.lineplot(x='Epoch', y=val_metric_name, data=df, label=val_metric_name, marker='s', color=palette[3], ax=axes[1])
            axes[1].set_title(f'{metric_name} Over Epochs')

        plt.tight_layout()
        plt.show()

class PyTorchClassifier(PyTorchBaseEstimator, ClassifierMixin):
    def __init__(self, loss='bcelogit', **kwargs):
        super().__init__(loss=loss, **kwargs)
        self._classes = None

    def configure_loss(self):
        if self.loss == 'bce': return nn.BCELoss()
        elif self.loss == 'bcelogit': return nn.BCEWithLogitsLoss()
        elif self.loss == 'crossentropy': return nn.CrossEntropyLoss()
        else: raise ValueError(f"Unsupported loss function: {self.loss}")

    def _calculate_metrics(self, y_true, y_pred_raw, epoch_loss, phase='train'):
        # Handle output shape
        if len(y_pred_raw.shape) == 1 or y_pred_raw.shape[1] == 1: # Binary
            y_pred_raw = y_pred_raw.reshape(-1)
            y_true = y_true.reshape(-1)
            probs = 1 / (1 + np.exp(-y_pred_raw)) # Sigmoid
            preds = (probs >= 0.5).astype(int)
            
            acc = accuracy_score(y_true, preds)
            try:
                auc = roc_auc_score(y_true, probs)
            except:
                auc = 0.0
            f1 = f1_score(y_true, preds, average='binary')
        else: # Multiclass
            probs = np.exp(y_pred_raw) / np.sum(np.exp(y_pred_raw), axis=1, keepdims=True) # Softmax
            preds = np.argmax(probs, axis=1)
            y_true = y_true.reshape(-1)
            
            acc = accuracy_score(y_true, preds)
            try:
                auc = roc_auc_score(y_true, probs, multi_class='ovr')
            except:
                auc = 0.0
            f1 = f1_score(y_true, preds, average='macro')
            
        return {
            f'{phase}_loss': epoch_loss,
            f'{phase}_acc': acc,
            f'{phase}_auc': auc,
            f'{phase}_f1': f1
        }

    @property
    def classes_(self):
        if self._classes is None:
            raise AttributeError("Model has not been fitted yet.")
        return self._classes

    def predict_proba(self, X):
        X, _ = self._prepare_data(X)
        
        if self.embedding_info:
            cat_features = X[:, :len(self.embedding_info)]
            num_features = X[:, len(self.embedding_info):]
        else:
            cat_features = None
            num_features = X

        self.model.eval()
        with torch.no_grad():
            cat_tensor = torch.tensor(cat_features, dtype=torch.long).to(self.device) if cat_features is not None else None
            num_tensor = torch.tensor(num_features, dtype=torch.float32).to(self.device)
            
            raw_preds = self.model(cat_tensor, num_tensor).detach().cpu().numpy()

        if self.loss == 'bcelogit':
            probs = 1 / (1 + np.exp(-raw_preds.flatten()))
            return np.column_stack([1 - probs, probs])
        elif self.loss == 'crossentropy':
             return np.exp(raw_preds) / np.sum(np.exp(raw_preds), axis=1, keepdims=True)
        return raw_preds

    def predict(self, X):
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return np.where(proba[:, 1] >= 0.5, 1, 0)
        return np.argmax(proba, axis=1)

class PyTorchRegressor(PyTorchBaseEstimator, RegressorMixin):
    def __init__(self, loss='mse', **kwargs):
        super().__init__(loss=loss, **kwargs)

    def configure_loss(self):
        if self.loss == 'mse': return nn.MSELoss()
        elif self.loss == 'mae': return nn.L1Loss()
        elif self.loss == 'huber': return nn.SmoothL1Loss()
        elif self.loss == 'rmsle': return RMSLELoss()
        elif self.loss == 'mape': return MAPE_Loss()
        else: raise ValueError(f"Unsupported loss function: {self.loss}")

    def _calculate_metrics(self, y_true, y_pred_raw, epoch_loss, phase='train'):
        y_pred = y_pred_raw.flatten()
        y_true = y_true.flatten()
        
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        try:
            y_pred_n = np.maximum(y_pred, 0)
            rmsle = root_mean_squared_log_error(y_true, y_pred_n)
        except:
            rmsle = np.nan

        return {
            f'{phase}_loss': epoch_loss,
            f'{phase}_rmse': rmse,
            f'{phase}_mae': mae,
            f'{phase}_mape': mape,
            f'{phase}_rmsle': rmsle,
            f'{phase}_r2': r2
        }

    def predict(self, X):
        X, _ = self._prepare_data(X)
        
        if self.embedding_info:
            cat_features = X[:, :len(self.embedding_info)]
            num_features = X[:, len(self.embedding_info):]
        else:
            cat_features = None
            num_features = X

        self.model.eval()
        with torch.no_grad():
            cat_tensor = torch.tensor(cat_features, dtype=torch.long).to(self.device) if cat_features is not None else None
            num_tensor = torch.tensor(num_features, dtype=torch.float32).to(self.device)
            
            preds = self.model(cat_tensor, num_tensor).detach().cpu().numpy().flatten()
        return preds