import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import optuna
import joblib
import json
import time
from IPython.display import clear_output, display
import threading

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from tqdm.auto import tqdm
    HAS_PYTORCH = True
    print("🧠 PyTorch loaded successfully - Optimized LSTM ready")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Using device: {device}")
except ImportError:
    HAS_PYTORCH = False
    print("❌ PyTorch not found - Install with: pip install torch optuna tqdm")

class TrainingProgressTracker:
    """Training progress tracker for PyTorch models"""
    
    def __init__(self, symbol, model_type, update_freq=10):
        self.symbol = symbol
        self.model_type = model_type
        self.update_freq = update_freq
        self.epoch_losses = []
        self.val_losses = []
        self.epoch_metrics = []
        self.start_time = None
        
    def on_train_begin(self):
        self.start_time = time.time()
        print(f"\n🔥 Starting {self.model_type} training for {self.symbol}")
        print("="*60)
        
    def on_epoch_end(self, epoch, train_loss, val_loss, train_mae, val_mae):
        # Store metrics
        self.epoch_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Display progress every update_freq epochs
        if (epoch + 1) % self.update_freq == 0:
            elapsed = time.time() - self.start_time
            
            print(f"\n📊 Epoch {epoch + 1} Progress:")
            print(f"   Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            print(f"   MAE: {train_mae:.6f} | Val MAE: {val_mae:.6f}")
            print(f"   Time: {elapsed:.1f}s | Avg: {elapsed/(epoch+1):.2f}s/epoch")
            
            # Show improvement
            if len(self.val_losses) > self.update_freq:
                recent_improvement = self.val_losses[-self.update_freq] - self.val_losses[-1]
                if recent_improvement > 0:
                    print(f"   📈 Improvement: {recent_improvement:.6f}")
                else:
                    print(f"   📉 Change: {recent_improvement:.6f}")
            
            # Plot progress
            self.plot_training_progress(epoch + 1)
    
    def plot_training_progress(self, current_epoch):
        """Plot real-time training progress"""
        try:
            plt.figure(figsize=(12, 4))
            
            # Training and validation loss
            plt.subplot(1, 2, 1)
            plt.plot(self.epoch_losses, label='Training Loss', color='blue', alpha=0.7)
            plt.plot(self.val_losses, label='Validation Loss', color='red', alpha=0.7)
            plt.title(f'{self.symbol} - Training Progress (Epoch {current_epoch})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Loss difference (overfitting indicator)
            plt.subplot(1, 2, 2)
            if len(self.epoch_losses) > 1:
                loss_diff = np.array(self.val_losses) - np.array(self.epoch_losses)
                plt.plot(loss_diff, label='Val - Train Loss', color='green', alpha=0.7)
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                plt.title('Overfitting Monitor')
                plt.xlabel('Epoch')
                plt.ylabel('Loss Difference')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Plot error: {e}")

# PyTorch Model Architectures
class TransformerLSTMModel(nn.Module):
    """PyTorch Transformer-LSTM architecture"""
    
    def __init__(self, input_size, params):
        super(TransformerLSTMModel, self).__init__()
        
        self.sequence_length = params.get('sequence_length', 60)
        lstm_units_1 = params.get('lstm_units_1', 128)
        lstm_units_2 = params.get('lstm_units_2', 64)
        lstm_units_3 = params.get('lstm_units_3', 32)
        dropout_rate = params.get('dropout_rate', 0.2)
        
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(input_size, lstm_units_1, batch_first=True, 
                            dropout=dropout_rate, bidirectional=True)
        self.norm1 = nn.LayerNorm(lstm_units_1 * 2)
        
        self.lstm2 = nn.LSTM(lstm_units_1 * 2, lstm_units_2, batch_first=True, 
                            dropout=dropout_rate, bidirectional=True)
        self.norm2 = nn.LayerNorm(lstm_units_2 * 2)
        
        # Multi-head attention
        d_model = lstm_units_2 * 2
        num_heads = params.get('num_heads', 8)
        self.attention = nn.MultiheadAttention(d_model, num_heads, 
                                             dropout=params.get('attention_dropout', 0.1),
                                             batch_first=True)
        self.norm_attn = nn.LayerNorm(d_model)
        
        # CNN branch
        conv_filters_1 = params.get('conv_filters_1', 64)
        conv_filters_2 = params.get('conv_filters_2', 32)
        conv_kernel_size = params.get('conv_kernel_size', 3)
        
        self.conv1 = nn.Conv1d(input_size, conv_filters_1, conv_kernel_size, padding=1)
        self.conv2 = nn.Conv1d(conv_filters_1, conv_filters_2, conv_kernel_size, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv_dropout = nn.Dropout(dropout_rate)
        
        # Calculate conv output size
        conv_output_size = conv_filters_2 * (self.sequence_length // 2)
        
        # Final LSTM
        self.lstm3 = nn.LSTM(d_model, lstm_units_3, batch_first=True, dropout=dropout_rate)
        
        # Dense layers
        dense_units_1 = params.get('dense_units_1', 128)
        dense_units_2 = params.get('dense_units_2', 64)
        dense_units_3 = params.get('dense_units_3', 32)
        
        combined_size = lstm_units_3 + d_model + 64  # lstm3 + attention_pooled + conv_dense
        
        self.conv_dense = nn.Linear(conv_output_size, 64)
        
        self.dense1 = nn.Linear(combined_size, dense_units_1)
        self.dense2 = nn.Linear(dense_units_1, dense_units_2)
        self.dense3 = nn.Linear(dense_units_2, dense_units_3)
        self.output = nn.Linear(dense_units_3, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_dense = nn.LayerNorm(dense_units_1)
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # LSTM branch
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.norm1(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.norm2(lstm2_out)
        
        # Attention
        attn_out, _ = self.attention(lstm2_out, lstm2_out, lstm2_out)
        attn_out = self.norm_attn(attn_out + lstm2_out)
        
        # CNN branch
        conv_input = x.transpose(1, 2)  # (batch, features, seq_len)
        conv_out = F.relu(self.conv1(conv_input))
        conv_out = F.relu(self.conv2(conv_out))
        conv_out = self.pool(conv_out)
        conv_out = self.conv_dropout(conv_out)
        conv_out = conv_out.flatten(1)
        conv_out = F.relu(self.conv_dense(conv_out))
        
        # Final LSTM
        lstm3_out, _ = self.lstm3(attn_out)
        lstm3_out = lstm3_out[:, -1, :]  # Take last output
        
        # Global average pooling for attention
        attn_pooled = torch.mean(attn_out, dim=1)
        
        # Combine branches
        combined = torch.cat([lstm3_out, attn_pooled, conv_out], dim=1)
        
        # Dense layers
        x = F.relu(self.dense1(combined))
        x = self.dropout(x)
        x = self.norm_dense(x)
        
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        
        x = F.relu(self.dense3(x))
        x = self.dropout(x)
        
        output = self.output(x)
        
        return output

class DeepLSTMModel(nn.Module):
    """PyTorch Deep LSTM architecture"""
    
    def __init__(self, input_size, params):
        super(DeepLSTMModel, self).__init__()
        
        lstm_layers = params.get('lstm_layers', 3)
        lstm_units = params.get('lstm_units_base', 256)
        dropout_rate = params.get('dropout_rate', 0.3)
        
        self.lstm_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(nn.LSTM(input_size, lstm_units, batch_first=True, 
                                       dropout=dropout_rate, bidirectional=True))
        self.norms.append(nn.LayerNorm(lstm_units * 2))
        
        # Additional LSTM layers
        for i in range(lstm_layers - 1):
            units = lstm_units // (2 ** i)
            self.lstm_layers.append(nn.LSTM(lstm_units * 2 if i == 0 else units * 2 * 2, 
                                           units, batch_first=True, 
                                           dropout=dropout_rate, bidirectional=True))
            self.norms.append(nn.LayerNorm(units * 2))
        
        # Dense layers
        dense_units = params.get('dense_units_base', 128)
        final_lstm_units = (lstm_units // (2 ** (lstm_layers - 2))) * 2
        
        self.dense1 = nn.Linear(final_lstm_units, dense_units)
        self.dense2 = nn.Linear(dense_units, dense_units // 2)
        self.output = nn.Linear(dense_units // 2, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.norms)):
            x, _ = lstm(x)
            if i < len(self.lstm_layers) - 1:  # Not the last layer
                x = norm(x)
            else:
                x = x[:, -1, :]  # Take last output for final layer
        
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        
        return self.output(x)

class CNNLSTMModel(nn.Module):
    """PyTorch CNN-LSTM architecture"""
    
    def __init__(self, input_size, params):
        super(CNNLSTMModel, self).__init__()
        
        conv_layers = params.get('conv_layers', 2)
        conv_filters = params.get('conv_filters_base', 128)
        conv_kernel_size = params.get('conv_kernel_size', 3)
        dropout_rate = params.get('dropout_rate', 0.2)
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        for i in range(conv_layers):
            filters = conv_filters // (2 ** i)
            if i == 0:
                self.conv_layers.append(nn.Conv1d(input_size, filters, conv_kernel_size, padding=1))
            else:
                prev_filters = conv_filters // (2 ** (i-1))
                self.conv_layers.append(nn.Conv1d(prev_filters, filters, conv_kernel_size, padding=1))
        
        self.pool = nn.MaxPool1d(2)
        self.conv_dropout = nn.Dropout(dropout_rate)
        
        # LSTM layers
        lstm_units = params.get('lstm_units_1', 128)
        final_conv_filters = conv_filters // (2 ** (conv_layers - 1))
        
        self.lstm1 = nn.LSTM(final_conv_filters, lstm_units, batch_first=True, dropout=dropout_rate)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True, dropout=dropout_rate)
        
        # Dense layers
        self.dense1 = nn.Linear(lstm_units // 2, 64)
        self.dense2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # CNN processing
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        x = self.pool(x)
        x = self.conv_dropout(x)
        
        # Back to LSTM format
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        
        # LSTM processing
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take last output
        
        # Dense layers
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        
        return self.output(x)

class GRUAttentionModel(nn.Module):
    """PyTorch GRU with Attention architecture"""
    
    def __init__(self, input_size, params):
        super(GRUAttentionModel, self).__init__()
        
        gru_units_1 = params.get('gru_units_1', 128)
        gru_units_2 = params.get('gru_units_2', 64)
        dropout_rate = params.get('dropout_rate', 0.2)
        
        # GRU layers
        self.gru1 = nn.GRU(input_size, gru_units_1, batch_first=True, 
                          dropout=dropout_rate, bidirectional=True)
        self.norm1 = nn.LayerNorm(gru_units_1 * 2)
        
        self.gru2 = nn.GRU(gru_units_1 * 2, gru_units_2, batch_first=True, 
                          dropout=dropout_rate, bidirectional=True)
        self.norm2 = nn.LayerNorm(gru_units_2 * 2)
        
        # Attention
        d_model = gru_units_2 * 2
        num_heads = params.get('num_heads', 4)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(d_model)
        
        # Dense layers
        dense_units_1 = params.get('dense_units_1', 64)
        dense_units_2 = params.get('dense_units_2', 32)
        
        self.dense1 = nn.Linear(d_model, dense_units_1)
        self.dense2 = nn.Linear(dense_units_1, dense_units_2)
        self.output = nn.Linear(dense_units_2, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # GRU processing
        x, _ = self.gru1(x)
        x = self.norm1(x)
        
        x, _ = self.gru2(x)
        x = self.norm2(x)
        
        # Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm_attn(attn_out + x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Dense layers
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        
        return self.output(x)

class HybridModel(nn.Module):
    """PyTorch Hybrid architecture"""
    
    def __init__(self, input_size, params):
        super(HybridModel, self).__init__()
        
        lstm_units = params.get('lstm_units', 64)
        gru_units = params.get('gru_units', 64)
        conv_filters = params.get('conv_filters', 64)
        conv_kernel_size = params.get('conv_kernel_size', 3)
        dropout_rate = params.get('dropout_rate', 0.2)
        
        # LSTM branch
        self.lstm1 = nn.LSTM(input_size, lstm_units, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units // 2, batch_first=True)
        
        # GRU branch
        self.gru1 = nn.GRU(input_size, gru_units, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(gru_units * 2, gru_units // 2, batch_first=True)
        
        # CNN branch
        self.conv = nn.Conv1d(input_size, conv_filters, conv_kernel_size, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Combine
        combined_size = (lstm_units // 2) + (gru_units // 2) + conv_filters
        dense_units = params.get('dense_units', 64)
        
        self.dense = nn.Linear(combined_size, dense_units)
        self.output = nn.Linear(dense_units, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # LSTM branch
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out[:, -1, :]
        
        # GRU branch
        gru_out, _ = self.gru1(x)
        gru_out, _ = self.gru2(gru_out)
        gru_out = gru_out[:, -1, :]
        
        # CNN branch
        cnn_input = x.transpose(1, 2)
        cnn_out = F.relu(self.conv(cnn_input))
        cnn_out = self.global_pool(cnn_out).squeeze(-1)
        
        # Combine
        combined = torch.cat([lstm_out, gru_out, cnn_out], dim=1)
        
        x = F.relu(self.dense(combined))
        x = self.dropout(x)
        
        return self.output(x)

class OptimizedLSTMPredictor:
    def __init__(self, symbols=['AAPL', 'MSFT', 'GOOGL'], 
                 optimization_trials=100, 
                 use_cache=True,
                 show_progress=True):
        self.symbols = symbols
        self.optimization_trials = optimization_trials
        self.use_cache = use_cache
        self.show_progress = show_progress
        self.device = device
        
        # Progress tracking
        self.optimization_progress = {}
        self.training_progress = {}
        
        # These will be optimized by Optuna
        self.best_params = {}
        self.models = {}
        self.scalers = {}
        self.data = {}
        self.results = {}
        self.feature_importance = {}
        self.studies = {}
        
        # UPDATED DATE CALCULATIONS - Use data up to end of last week, predict for this week
        self.today = datetime.now().date()
        
        # Calculate the last completed Friday (end of last week)
        days_since_monday = self.today.weekday()  # Monday = 0, Sunday = 6
        
        # Calculate end of last week (last Friday)
        if days_since_monday <= 4:  # Monday to Friday - go back to previous Friday
            days_to_last_friday = days_since_monday + 3
        else:  # Saturday or Sunday - go back to Friday of this week
            days_to_last_friday = days_since_monday - 4
        
        # Data cutoff is end of last week (last Friday)
        self.data_cutoff_date = self.today - timedelta(days=days_to_last_friday)
        
        # For prediction target, we predict for THIS WEEK (current week's end)
        # Calculate this Friday
        if days_since_monday <= 4:  # Monday to Friday
            days_to_this_friday = 4 - days_since_monday
        else:  # Saturday or Sunday
            days_to_this_friday = 11 - days_since_monday  # Next Friday
        
        self.prediction_target_date = self.today + timedelta(days=days_to_this_friday)
        
        print(f"📅 Data Usage Policy:")
        print(f"  Today: {self.today}")
        print(f"  Data cutoff (training): {self.data_cutoff_date} (End of last week)")
        print(f"  Prediction target: {self.prediction_target_date} (This week)")
        print(f"  🚫 NO data from current week will be used for training")
        print(f"  🎯 Predicting for current week using last week's data")
        
        self.test_period_days = 14  # Increased for better validation
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def display_optimization_dashboard(self, symbol, study, trial_number, current_value, best_value):
        """Display real-time optimization dashboard"""
        if not self.show_progress:
            return
            
        try:
            clear_output(wait=True)
            
            print(f"🚀 OPTIMIZATION DASHBOARD - {symbol}")
            print("="*70)
            print(f"Trial: {trial_number}/{self.optimization_trials}")
            print(f"Current Value: {current_value:.6f}")
            print(f"Best Value: {best_value:.6f}")
            print(f"Progress: {trial_number/self.optimization_trials*100:.1f}%")
            print("="*70)
            
            # Plot optimization progress
            if len(study.trials) > 1:
                plt.figure(figsize=(15, 5))
                
                # Objective values over trials
                plt.subplot(1, 3, 1)
                values = [trial.value for trial in study.trials if trial.value is not None]
                plt.plot(values, 'b-', alpha=0.6, linewidth=1)
                plt.plot([min(values)]*len(values), 'r--', alpha=0.8, label=f'Best: {min(values):.4f}')
                plt.title(f'{symbol} - Optimization Progress')
                plt.xlabel('Trial')
                plt.ylabel('Objective Value (Lower = Better)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Best value trend (rolling minimum)
                plt.subplot(1, 3, 2)
                best_so_far = np.minimum.accumulate(values)
                plt.plot(best_so_far, 'g-', linewidth=2, label='Best So Far')
                plt.title('Best Value Trend')
                plt.xlabel('Trial')
                plt.ylabel('Best Objective Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Parameter importance (if available)
                plt.subplot(1, 3, 3)
                try:
                    importances = optuna.importance.get_param_importances(study)
                    if importances:
                        params = list(importances.keys())[:10]  # Top 10
                        values = [importances[p] for p in params]
                        plt.barh(params, values)
                        plt.title('Top Parameter Importance')
                        plt.xlabel('Importance')
                except:
                    plt.text(0.5, 0.5, 'Parameter importance\nwill be available\nafter more trials', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Parameter Importance')
                
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"⚠️ Dashboard error: {e}")
    
    def load_market_data(self, start_date="2018-01-01", end_date=None):
        """Load market data with strict cutoff to prevent data leakage"""
        print("🔄 Loading market data for optimization...")
        print(f"🚫 Enforcing data cutoff: {self.data_cutoff_date}")
        
        # Set end_date to our cutoff to prevent future data leakage
        if end_date is None:
            end_date = self.data_cutoff_date
        else:
            # Use the earlier of provided end_date or our cutoff
            provided_end = datetime.strptime(end_date, "%Y-%m-%d").date() if isinstance(end_date, str) else end_date
            end_date = min(provided_end, self.data_cutoff_date)
        
        print(f"📊 Loading data from {start_date} to {end_date}")
        
        # Progress bar for data loading
        pbar = tqdm(self.symbols, desc="Loading data") if self.show_progress else self.symbols
        
        for symbol in pbar:
            try:
                if self.show_progress:
                    pbar.set_description(f"Loading {symbol}")
                
                raw_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if not raw_data.empty:
                    clean_data = pd.DataFrame(index=raw_data.index)
                    
                    for col_name in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                        if isinstance(raw_data.columns, pd.MultiIndex):
                            matching_cols = [col for col in raw_data.columns if col[0] == col_name]
                            if matching_cols:
                                clean_data[col_name] = raw_data[matching_cols[0]].values
                        else:
                            if col_name in raw_data.columns:
                                clean_data[col_name] = raw_data[col_name].values
                    
                    # CRITICAL: Additional filter to ensure no future data
                    cutoff_datetime = pd.Timestamp(self.data_cutoff_date)
                    clean_data = clean_data[clean_data.index <= cutoff_datetime]
                    
                    if len(clean_data) == 0:
                        print(f"❌ No data for {symbol} before cutoff date {self.data_cutoff_date}")
                        continue
                    
                    self.data[symbol] = clean_data
                    
                    latest_date = clean_data.index[-1].date()
                    print(f"✅ Loaded {symbol}: {len(clean_data)} days (Latest: {latest_date})")
                    
                    # Verify no future data leakage
                    if latest_date > self.data_cutoff_date:
                        print(f"⚠️ WARNING: {symbol} has data beyond cutoff! Latest: {latest_date}, Cutoff: {self.data_cutoff_date}")
                        # Remove future data
                        self.data[symbol] = clean_data[clean_data.index.date <= self.data_cutoff_date]
                        print(f"🔧 Fixed: Now ends at {self.data[symbol].index[-1].date()}")
                    
                else:
                    print(f"❌ No data found for {symbol}")
                    
            except Exception as e:
                print(f"❌ Error downloading {symbol}: {e}")
        
        # Final verification
        print(f"\n🔍 Data Validation:")
        for symbol, df in self.data.items():
            latest_date = df.index[-1].date()
            days_safe = (self.data_cutoff_date - latest_date).days
            print(f"  {symbol}: Latest {latest_date}, Safe margin: {days_safe} days")
            if days_safe < 0:
                print(f"  ⚠️ ERROR: {symbol} has future data! This should not happen!")
        
        return self.data
    
    def calculate_optimized_technical_indicators(self, df, params):
        """Calculate technical indicators with progress tracking"""
        data = df.copy()
        close_prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']
        volume = data['Volume']
        open_prices = data['Open']
        
        if self.show_progress:
            print("🔧 Calculating technical indicators...")
        
        # Optimized moving averages
        sma_periods = params.get('sma_periods', [5, 10, 20, 50])
        ema_periods = params.get('ema_periods', [5, 10, 20, 50])
        
        for period in sma_periods:
            data[f'SMA_{period}'] = close_prices.rolling(window=period).mean()
            
        for period in ema_periods:
            data[f'EMA_{period}'] = close_prices.ewm(span=period).mean()
            
        # Optimized MACD
        macd_fast = params.get('macd_fast', 12)
        macd_slow = params.get('macd_slow', 26)
        macd_signal = params.get('macd_signal', 9)
        
        ema_fast = close_prices.ewm(span=macd_fast).mean()
        ema_slow = close_prices.ewm(span=macd_slow).mean()
        macd = ema_fast - ema_slow
        macd_signal_line = macd.ewm(span=macd_signal).mean()
        data['MACD'] = macd
        data['MACD_signal'] = macd_signal_line
        data['MACD_histogram'] = macd - macd_signal_line
        
        # Optimized RSI
        rsi_periods = params.get('rsi_periods', [14, 21])
        for period in rsi_periods:
            data[f'RSI_{period}'] = self.calculate_rsi(close_prices, period)
            
        # Optimized Bollinger Bands
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2.0)
        bb_middle = close_prices.rolling(window=bb_period).mean()
        bb_std_val = close_prices.rolling(window=bb_period).std()
        data['BB_upper'] = bb_middle + (bb_std_val * bb_std)
        data['BB_lower'] = bb_middle - (bb_std_val * bb_std)
        data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / bb_middle
        data['BB_position'] = (close_prices - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'] + 1e-8)
        
        # Optimized Stochastic
        stoch_k_period = params.get('stoch_k_period', 14)
        stoch_d_period = params.get('stoch_d_period', 3)
        stoch_k, stoch_d = self.calculate_stochastic(high_prices, low_prices, close_prices, 
                                                    stoch_k_period, stoch_d_period)
        data['Stoch_K'] = stoch_k
        data['Stoch_D'] = stoch_d
        
        # Optimized ATR
        atr_period = params.get('atr_period', 14)
        data['ATR'] = self.calculate_atr(high_prices, low_prices, close_prices, atr_period)
        
        # Volume indicators
        volume_ma_period = params.get('volume_ma_period', 20)
        data[f'Volume_MA'] = volume.rolling(window=volume_ma_period).mean()
        data['Volume_Ratio'] = volume / (data['Volume_MA'] + 1e-8)
        
        # Advanced indicators
        data['OBV'] = self.calculate_obv(close_prices, volume)
        data['MFI'] = self.calculate_mfi(high_prices, low_prices, close_prices, volume)
        
        # Price action
        data['Gap'] = (open_prices - close_prices.shift(1)) / close_prices.shift(1) * 100
        data['Intraday_Range'] = (high_prices - low_prices) / close_prices * 100
        data['Body_Size'] = abs(close_prices - open_prices) / close_prices * 100
        
        # Returns with optimized periods
        return_periods = params.get('return_periods', [1, 3, 5, 10])
        for period in return_periods:
            data[f'Return_{period}d'] = close_prices.pct_change(period) * 100
            
        # Price relative to moving averages
        for period in sma_periods[:3]:  # Top 3 periods
            if f'SMA_{period}' in data.columns:
                data[f'Price_vs_SMA_{period}'] = (close_prices / data[f'SMA_{period}'] - 1) * 100
                
        # Volatility
        volatility_period = params.get('volatility_period', 20)
        returns = close_prices.pct_change()
        data['Historical_Volatility'] = returns.rolling(window=volatility_period).std() * np.sqrt(252) * 100
        
        # Momentum
        momentum_period = params.get('momentum_period', 10)
        data['Momentum'] = close_prices / close_prices.shift(momentum_period) * 100
        
        # Feature interactions (optimized combinations)
        data['Volume_Price_Trend'] = data['Volume_Ratio'] * data['Return_1d']
        data['RSI_BB_Combo'] = data[f'RSI_{rsi_periods[0]}'] * data['BB_position']
        data['MACD_RSI_Combo'] = data['MACD'] * data[f'RSI_{rsi_periods[0]}'] / 100
        
        return data
    
    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-8))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def calculate_atr(self, high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_obv(self, close, volume):
        obv = volume.copy() * 0
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv
    
    def calculate_mfi(self, high, low, close, volume, period=14):
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        money_ratio = positive_flow / (negative_flow + 1e-8)
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi
    
    def select_features_optimized(self, df, target_col, params):
        """Optimized feature selection with progress tracking"""
        
        # Get all potential features (excluding target and future columns)
        exclude_cols = ['Close_future', 'Return_future', target_col] + \
                      [col for col in df.columns if 'future' in col.lower()]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove columns with too many NaN values
        valid_features = []
        for col in feature_cols:
            if df[col].notna().sum() > len(df) * 0.7:  # At least 70% valid data
                valid_features.append(col)
        
        if len(valid_features) == 0:
            return ['Close', 'Volume']  # Fallback
        
        # Clean data for feature selection
        X = df[valid_features].fillna(method='ffill').fillna(method='bfill')
        y = df[target_col].fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows where target is still NaN
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:  # Not enough data
            return valid_features[:20]  # Return top 20 features
        
        # Feature selection method
        selection_method = params.get('feature_selection_method', 'mutual_info')
        n_features = params.get('n_features', min(25, len(valid_features)))
        
        try:
            if selection_method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
            else:
                selector = SelectKBest(score_func=f_regression, k=n_features)
            
            X_selected = selector.fit_transform(X, y)
            selected_features = [valid_features[i] for i in selector.get_support(indices=True)]
            
            # Always include basic price features
            must_have = ['Close', 'Volume', 'High', 'Low', 'Open']
            for feature in must_have:
                if feature in valid_features and feature not in selected_features:
                    selected_features.append(feature)
            
            return selected_features[:n_features]
            
        except Exception as e:
            if self.show_progress:
                print(f"⚠️ Feature selection error: {e}")
            return valid_features[:n_features]
    
    def build_optimized_model(self, input_size, params):
        """Build optimized PyTorch model with architecture info"""
        
        model_type = params.get('model_type', 'transformer_lstm')
        
        if self.show_progress:
            print(f"🏗️ Building {model_type} architecture...")
            print(f"   Input size: {input_size}")
            print(f"   LSTM units: {params.get('lstm_units_1', 128)}")
            print(f"   Dropout rate: {params.get('dropout_rate', 0.2)}")
            print(f"   Learning rate: {params.get('learning_rate', 0.001)}")
        
        if model_type == 'transformer_lstm':
            model = TransformerLSTMModel(input_size, params)
        elif model_type == 'deep_lstm':
            model = DeepLSTMModel(input_size, params)
        elif model_type == 'cnn_lstm':
            model = CNNLSTMModel(input_size, params)
        elif model_type == 'gru_attention':
            model = GRUAttentionModel(input_size, params)
        else:
            model = HybridModel(input_size, params)
        
        return model.to(self.device)
    
    def create_sequences_optimized(self, features, targets, sequence_length):
        """Create sequences with optimized parameters"""
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            sequence = features[i-sequence_length:i]
            target = targets[i]
            
            if not np.isnan(target) and np.isfinite(sequence).all():
                X.append(sequence)
                y.append(target)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def prepare_data_with_strict_cutoff(self, df_with_indicators, params):
        """Prepare data with strict cutoff to prevent data leakage"""
        
        # Create target variable with STRICT future data handling
        # We predict 5 days ahead, but only use data that doesn't leak into our cutoff
        future_days = 5
        
        # Calculate the latest date we can use for creating targets
        latest_data_date = df_with_indicators.index[-1].date()
        
        # The target needs to be future_days ahead, so we need to limit our training data
        # to ensure the target doesn't go beyond our cutoff
        max_training_date = self.data_cutoff_date - timedelta(days=future_days)
        
        print(f"🔍 Data preparation for current week prediction:")
        print(f"  Data ends at: {latest_data_date}")
        print(f"  Max training date: {max_training_date} (cutoff - {future_days} days)")
        print(f"  Training cutoff: {self.data_cutoff_date} (last week)")
        print(f"  Prediction target: {self.prediction_target_date} (current week)")
        
        # Filter data to ensure no future leakage
        max_training_datetime = pd.Timestamp(max_training_date)
        df_filtered = df_with_indicators[df_with_indicators.index <= max_training_datetime]
        
        if len(df_filtered) == 0:
            raise ValueError(f"No data available before max training date {max_training_date}")
        
        # Now create the target variable safely
        df_filtered = df_filtered.copy()
        df_filtered['Return_future_5d'] = (df_filtered['Close'].shift(-future_days) / 
                                         df_filtered['Close'] - 1) * 100
        
        # Verify no future data leakage in target
        last_valid_target_idx = len(df_filtered) - future_days
        if last_valid_target_idx < 50:  # Need minimum data
            raise ValueError(f"Insufficient data after applying cutoff. Only {last_valid_target_idx} samples available.")
        
        # Additional safety: only use data where we have valid targets that don't exceed cutoff
        df_safe = df_filtered.iloc[:last_valid_target_idx].copy()
        
        print(f"  Final training data: {len(df_safe)} samples")
        print(f"  Training ends at: {df_safe.index[-1].date()}")
        
        return df_safe
    
    def train_pytorch_model(self, model, X_train, y_train, X_val, y_val, params, symbol):
        """Train PyTorch model with progress tracking"""
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=params.get('batch_size', 32), shuffle=True)
        
        # Optimizer and loss function
        learning_rate = params.get('learning_rate', 0.001)
        weight_decay = params.get('weight_decay', 0.01)
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        loss_function = params.get('loss_function', 'mse')
        if loss_function == 'mae':
            criterion = nn.L1Loss()
        elif loss_function == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                        patience=10, min_lr=1e-7)
        
        # Training progress tracker
        progress_tracker = TrainingProgressTracker(symbol, params['model_type'], update_freq=10)
        progress_tracker.on_train_begin()
        
        # Early stopping
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        epochs = params.get('epochs', 100)
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0
            epoch_train_mae = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.get('clipnorm', 1.0))
                
                optimizer.step()
                
                epoch_train_loss += loss.item()
                epoch_train_mae += F.l1_loss(outputs, batch_y).item()
            
            epoch_train_loss /= len(train_loader)
            epoch_train_mae /= len(train_loader)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_mae = F.l1_loss(val_outputs, y_val_tensor).item()
            
            train_losses.append(epoch_train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Progress tracking
            if self.show_progress:
                progress_tracker.on_epoch_end(epoch, epoch_train_loss, val_loss, epoch_train_mae, val_mae)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"⚡ Early stopping at epoch {epoch + 1}")
                break
        
        # Restore best model
        model.load_state_dict(best_model_state)
        
        # Return training history
        return {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
    
    def objective_function(self, trial, symbol):
        """Optuna objective function with progress tracking and strict data cutoff"""
        
        # Show trial progress
        if self.show_progress and hasattr(trial, 'number'):
            study = trial.study
            
            # Safe way to get best value - check for completed trials first
            try:
                completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
                if len(completed_trials) > 0:
                    best_value = study.best_value
                else:
                    best_value = float('inf')
            except (ValueError, AttributeError):
                best_value = float('inf')
            
            current_value = trial.value if hasattr(trial, 'value') and trial.value is not None else float('inf')
            print(f"\n🔬 Trial {trial.number + 1} for {symbol}")
            print(f"   Testing parameters...")
            
        # Suggest hyperparameters (same as TensorFlow version)
        params = {
            # Data preprocessing parameters
            'sequence_length': trial.suggest_int('sequence_length', 30, 120),
            'scaler_type': trial.suggest_categorical('scaler_type', ['robust', 'standard', 'minmax']),
            
            # Technical indicator parameters
            'sma_periods': [
                trial.suggest_int('sma_period_1', 5, 15),
                trial.suggest_int('sma_period_2', 15, 30),
                trial.suggest_int('sma_period_3', 30, 60),
                trial.suggest_int('sma_period_4', 60, 200)
            ],
            'ema_periods': [
                trial.suggest_int('ema_period_1', 5, 15),
                trial.suggest_int('ema_period_2', 15, 30),
                trial.suggest_int('ema_period_3', 30, 60)
            ],
            'macd_fast': trial.suggest_int('macd_fast', 8, 16),
            'macd_slow': trial.suggest_int('macd_slow', 20, 35),
            'macd_signal': trial.suggest_int('macd_signal', 7, 12),
            'rsi_periods': [
                trial.suggest_int('rsi_period_1', 10, 18),
                trial.suggest_int('rsi_period_2', 18, 25)
            ],
            'bb_period': trial.suggest_int('bb_period', 15, 25),
            'bb_std': trial.suggest_float('bb_std', 1.5, 2.5),
            'stoch_k_period': trial.suggest_int('stoch_k_period', 10, 20),
            'stoch_d_period': trial.suggest_int('stoch_d_period', 3, 7),
            'atr_period': trial.suggest_int('atr_period', 10, 20),
            'volume_ma_period': trial.suggest_int('volume_ma_period', 15, 30),
            'return_periods': [1, 3, 5, trial.suggest_int('return_period_4', 7, 15)],
            'volatility_period': trial.suggest_int('volatility_period', 15, 30),
            'momentum_period': trial.suggest_int('momentum_period', 7, 15),
            
            # Feature selection parameters
            'feature_selection_method': trial.suggest_categorical('feature_selection_method', 
                                                                ['mutual_info', 'f_regression']),
            'n_features': trial.suggest_int('n_features', 15, 40),
            
            # Model architecture parameters
            'model_type': trial.suggest_categorical('model_type', 
                                                  ['transformer_lstm', 'deep_lstm', 'cnn_lstm', 
                                                   'gru_attention', 'hybrid']),
            
            # LSTM parameters
            'lstm_units_1': trial.suggest_int('lstm_units_1', 64, 256),
            'lstm_units_2': trial.suggest_int('lstm_units_2', 32, 128),
            'lstm_units_3': trial.suggest_int('lstm_units_3', 16, 64),
            'lstm_layers': trial.suggest_int('lstm_layers', 2, 4),
            'lstm_units_base': trial.suggest_int('lstm_units_base', 128, 512),
            
            # GRU parameters
            'gru_units_1': trial.suggest_int('gru_units_1', 64, 200),
            'gru_units_2': trial.suggest_int('gru_units_2', 32, 100),
            'gru_units': trial.suggest_int('gru_units', 32, 128),
            
            # CNN parameters
            'conv_filters_1': trial.suggest_int('conv_filters_1', 32, 128),
            'conv_filters_2': trial.suggest_int('conv_filters_2', 16, 64),
            'conv_filters_base': trial.suggest_int('conv_filters_base', 64, 256),
            'conv_kernel_size': trial.suggest_int('conv_kernel_size', 2, 5),
            'conv_layers': trial.suggest_int('conv_layers', 1, 3),
            'conv_filters': trial.suggest_int('conv_filters', 32, 128),
            
            # Attention parameters
            'num_heads': trial.suggest_int('num_heads', 4, 12),
            'key_dim': trial.suggest_int('key_dim', 32, 128),
            'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.3),
            
            # Dense layer parameters
            'dense_units_1': trial.suggest_int('dense_units_1', 64, 256),
            'dense_units_2': trial.suggest_int('dense_units_2', 32, 128),
            'dense_units_3': trial.suggest_int('dense_units_3', 16, 64),
            'dense_units_base': trial.suggest_int('dense_units_base', 64, 256),
            'dense_units': trial.suggest_int('dense_units', 32, 128),
            
            # Regularization parameters
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.1, 0.4),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.001, 0.1, log=True),
            
            # Training parameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
            'clipnorm': trial.suggest_float('clipnorm', 0.5, 2.0),
            'batch_size': trial.suggest_int('batch_size', 32, 256),
            'epochs': trial.suggest_int('epochs', 30, 100),  # Reduced for faster optimization
            
            # Loss function
            'loss_function': trial.suggest_categorical('loss_function', ['mae', 'mse', 'huber'])
        }
        
        try:
            # Prepare data with optimized parameters and STRICT cutoff
            df = self.data[symbol].copy()
            df_with_indicators = self.calculate_optimized_technical_indicators(df, params)
            
            # Use strict cutoff preparation
            df_clean = self.prepare_data_with_strict_cutoff(df_with_indicators, params)
            
            # Select features
            feature_cols = self.select_features_optimized(df_clean, 'Return_future_5d', params)
            
            # Prepare final dataset
            df_final = df_clean[feature_cols + ['Return_future_5d']].dropna()
            
            if len(df_final) < 200:  # Need sufficient data
                return float('inf')
            
            # Split data with time series respect
            sequence_length = params['sequence_length']
            test_size = min(60, len(df_final) // 5)  # 20% for testing
            train_end = len(df_final) - test_size
            
            train_data = df_final.iloc[:train_end]
            test_data = df_final.iloc[train_end - sequence_length:]
            
            # Scale data
            if params['scaler_type'] == 'robust':
                feature_scaler = RobustScaler()
                target_scaler = StandardScaler()
            elif params['scaler_type'] == 'standard':
                feature_scaler = StandardScaler()
                target_scaler = StandardScaler()
            else:
                feature_scaler = MinMaxScaler()
                target_scaler = MinMaxScaler()
            
            train_features_scaled = feature_scaler.fit_transform(train_data[feature_cols])
            train_target = train_data['Return_future_5d'].values.reshape(-1, 1)
            train_target_scaled = target_scaler.fit_transform(train_target)
            
            test_features_scaled = feature_scaler.transform(test_data[feature_cols])
            test_target = test_data['Return_future_5d'].values.reshape(-1, 1)
            test_target_scaled = target_scaler.transform(test_target)
            
            # Create sequences
            X_train, y_train = self.create_sequences_optimized(train_features_scaled, 
                                                              train_target_scaled.flatten(), 
                                                              sequence_length)
            X_test, y_test = self.create_sequences_optimized(test_features_scaled, 
                                                            test_target_scaled.flatten(), 
                                                            sequence_length)
            
            if len(X_train) < 30 or len(X_test) < 5:
                return float('inf')
            
            # Build and train model
            input_size = len(feature_cols)
            model = self.build_optimized_model(input_size, params)
            
            # Train model (silent during optimization)
            params_copy = params.copy()
            params_copy['epochs'] = min(50, params['epochs'])  # Reduce for optimization
            
            # Temporarily disable progress for optimization
            original_show_progress = self.show_progress
            self.show_progress = False
            
            history = self.train_pytorch_model(model, X_train, y_train, X_test, y_test, params_copy, symbol)
            
            self.show_progress = original_show_progress
            
            # Make predictions
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                y_pred_scaled = model(X_test_tensor).squeeze().cpu().numpy()
            
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test_original, y_pred)
            mae = mean_absolute_error(y_test_original, y_pred)
            r2 = r2_score(y_test_original, y_pred)
            
            direction_accuracy = np.mean(np.sign(y_test_original) == np.sign(y_pred))
            
            # Combined objective (minimize)
            # We want to minimize MAE and maximize R2 and direction accuracy
            objective = mae - (r2 * 2) - (direction_accuracy * 3)
            
            # Penalty for overfitting
            with torch.no_grad():
                X_train_tensor = torch.FloatTensor(X_train[-len(X_test):]).to(self.device)
                train_pred_scaled = model(X_train_tensor).squeeze().cpu().numpy()
            
            train_pred = target_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
            train_actual = target_scaler.inverse_transform(y_train[-len(X_test):].reshape(-1, 1)).flatten()
            train_mae = mean_absolute_error(train_actual, train_pred)
            
            if mae > train_mae * 1.5:  # Significant overfitting
                objective += 2.0
            
            # Show trial result
            if self.show_progress:
                print(f"   📊 Trial result: {objective:.4f} (R²: {r2:.3f}, Dir: {direction_accuracy:.1%}, MAE: {mae:.3f})")
            
            return objective
            
        except Exception as e:
            if self.show_progress:
                print(f"   ❌ Trial failed: {e}")
            return float('inf')
    
    def optimize_hyperparameters(self, symbol):
        """Optimize hyperparameters with visual progress tracking"""
        
        print(f"🔥 Starting hyperparameter optimization for {symbol}...")
        print(f"🎯 Running {self.optimization_trials} trials...")
        
        # Create study
        study_name = f"pytorch_lstm_optimization_{symbol}"
        
        try:
            # Try to load existing study
            if self.use_cache:
                study = optuna.load_study(study_name=study_name, 
                                        storage=f'sqlite:///optuna_pytorch_{symbol}.db')
                print(f"📖 Loaded existing study with {len(study.trials)} trials")
            else:
                raise Exception("Not using cache")
        except:
            # Create new study
            study = optuna.create_study(direction='minimize', 
                                      study_name=study_name,
                                      storage=f'sqlite:///optuna_pytorch_{symbol}.db',
                                      load_if_exists=True)
            print(f"🆕 Created new optimization study")
        
        # Progress tracking
        initial_trials = len(study.trials)
        
        # Custom callback for progress visualization
        def progress_callback(study, trial):
            if self.show_progress and trial.number % 5 == 0:
                current_value = trial.value if trial.value is not None else float('inf')
                
                # Safe best value access
                try:
                    completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
                    best_value = study.best_value if len(completed_trials) > 0 else float('inf')
                except (ValueError, AttributeError):
                    best_value = float('inf')
                
                print(f"🔬 Trial {trial.number + 1}/{self.optimization_trials} - Current: {current_value:.4f}, Best: {best_value:.4f}")
            
        # Optimize with progress tracking
        if self.show_progress:
            study.optimize(lambda trial: self.objective_function(trial, symbol), 
                          n_trials=self.optimization_trials,
                          timeout=3600,  # 1 hour timeout
                          callbacks=[progress_callback])
        else:
            study.optimize(lambda trial: self.objective_function(trial, symbol), 
                          n_trials=self.optimization_trials,
                          timeout=3600)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\n✅ Optimization completed for {symbol}")
        print(f"🏆 Best objective value: {best_value:.4f}")
        print(f"🎯 Best trial: {study.best_trial.number}")
        print(f"📈 Total trials: {len(study.trials)} ({len(study.trials) - initial_trials} new)")
        
        # Display final optimization summary
        if self.show_progress:
            self.display_optimization_summary(symbol, study)
        
        # Save results
        self.best_params[symbol] = best_params
        self.studies[symbol] = study
        
        # Save best parameters to file
        with open(f'best_params_pytorch_{symbol}.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        return best_params, study
    
    def display_optimization_summary(self, symbol, study):
        """Display comprehensive optimization summary"""
        try:
            plt.figure(figsize=(18, 10))
            
            # 1. Optimization history
            plt.subplot(2, 3, 1)
            values = [trial.value for trial in study.trials if trial.value is not None]
            plt.plot(values, 'b-', alpha=0.6, linewidth=1, label='Trial Values')
            
            # Best value trend
            best_so_far = np.minimum.accumulate(values)
            plt.plot(best_so_far, 'r-', linewidth=2, label='Best So Far')
            
            plt.title(f'{symbol} - Optimization History')
            plt.xlabel('Trial')
            plt.ylabel('Objective Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. Parameter importance
            plt.subplot(2, 3, 2)
            try:
                importances = optuna.importance.get_param_importances(study)
                if importances:
                    params = list(importances.keys())[:10]  # Top 10
                    imp_values = [importances[p] for p in params]
                    y_pos = np.arange(len(params))
                    plt.barh(y_pos, imp_values)
                    plt.yticks(y_pos, params)
                    plt.title('Top 10 Parameter Importance')
                    plt.xlabel('Importance')
            except:
                plt.text(0.5, 0.5, 'Parameter importance\navailable after more trials', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Parameter Importance')
            
            # 3. Best trial parameters (top 10)
            plt.subplot(2, 3, 3)
            best_params = study.best_params
            if best_params:
                # Select most important parameters to display
                important_params = ['model_type', 'learning_rate', 'lstm_units_1', 
                                  'sequence_length', 'dropout_rate', 'batch_size']
                display_params = {}
                for param in important_params:
                    if param in best_params:
                        display_params[param] = best_params[param]
                
                param_names = list(display_params.keys())
                param_values = [str(display_params[p])[:10] for p in param_names]  # Truncate long values
                
                y_pos = np.arange(len(param_names))
                plt.barh(y_pos, [1]*len(param_names), alpha=0.6)
                
                for i, (name, value) in enumerate(zip(param_names, param_values)):
                    plt.text(0.5, i, f'{name}: {value}', ha='center', va='center')
                
                plt.yticks([])
                plt.xlabel('Best Parameters')
                plt.title('Best Trial Configuration')
            
            # 4. Objective value distribution
            plt.subplot(2, 3, 4)
            if len(values) > 10:
                plt.hist(values, bins=20, alpha=0.7, edgecolor='black')
                plt.axvline(study.best_value, color='red', linestyle='--', 
                           label=f'Best: {study.best_value:.4f}')
                plt.xlabel('Objective Value')
                plt.ylabel('Frequency')
                plt.title('Objective Value Distribution')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # 5. Convergence analysis
            plt.subplot(2, 3, 5)
            if len(values) > 10:
                # Rolling average of improvement
                window = min(10, len(values) // 4)
                rolling_best = pd.Series(best_so_far).rolling(window=window).mean()
                plt.plot(rolling_best, 'g-', linewidth=2, label=f'Rolling Mean ({window} trials)')
                plt.plot(best_so_far, 'r--', alpha=0.5, label='Best Value')
                plt.xlabel('Trial')
                plt.ylabel('Objective Value')
                plt.title('Convergence Analysis')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # 6. Summary statistics
            plt.subplot(2, 3, 6)
            stats_text = f"""
OPTIMIZATION SUMMARY for {symbol}

Total Trials: {len(study.trials)}
Best Value: {study.best_value:.6f}
Best Trial: #{study.best_trial.number}

Value Statistics:
Mean: {np.mean(values):.4f}
Std: {np.std(values):.4f}
Min: {min(values):.4f}
Max: {max(values):.4f}

Best Model Type: {best_params.get('model_type', 'N/A')}
Best Learning Rate: {best_params.get('learning_rate', 'N/A'):.6f}
Best Sequence Length: {best_params.get('sequence_length', 'N/A')}

PyTorch Device: {self.device}
Data Cutoff: {self.data_cutoff_date}
            """
            plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
            plt.axis('off')
            
            plt.tight_layout()
            plt.suptitle(f'🚀 PYTORCH OPTIMIZATION SUMMARY - {symbol}', fontsize=16, y=1.02)
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Summary plot error: {e}")
    
    def train_optimized_model(self, symbol):
        """Train the final optimized model with full progress tracking and strict data cutoff"""
        
        if symbol not in self.best_params:
            print(f"❌ No optimized parameters found for {symbol}")
            return None
        
        print(f"\n🚀 Training optimized PyTorch model for {symbol}...")
        print("="*60)
        
        params = self.best_params[symbol]
        
        # Display model configuration
        print(f"🏗️ MODEL CONFIGURATION:")
        print(f"   Architecture: {params.get('model_type', 'N/A')}")
        print(f"   Sequence Length: {params.get('sequence_length', 'N/A')}")
        print(f"   Learning Rate: {params.get('learning_rate', 'N/A')}")
        print(f"   Batch Size: {params.get('batch_size', 'N/A')}")
        print(f"   Dropout Rate: {params.get('dropout_rate', 'N/A')}")
        print(f"   LSTM Units: {params.get('lstm_units_1', 'N/A')}")
        print(f"   Device: {self.device}")
        print(f"   Data Cutoff: {self.data_cutoff_date}")
        print("="*60)
        
        # Prepare data with strict cutoff
        df = self.data[symbol].copy()
        df_with_indicators = self.calculate_optimized_technical_indicators(df, params)
        
        # Use strict cutoff preparation
        df_clean = self.prepare_data_with_strict_cutoff(df_with_indicators, params)
        
        # Select features
        feature_cols = self.select_features_optimized(df_clean, 'Return_future_5d', params)
        print(f"🎯 Selected {len(feature_cols)} features for training")
        
        # Prepare final dataset
        df_final = df_clean[feature_cols + ['Return_future_5d']].dropna()
        
        # Split data for final training
        sequence_length = params['sequence_length']
        test_size = 21  # 3 weeks for final validation
        train_end = len(df_final) - test_size
        
        train_data = df_final.iloc[:train_end]
        test_data = df_final.iloc[train_end - sequence_length:]
        
        print(f"📊 Data split: {len(train_data)} training days, {len(test_data)} test days")
        print(f"📅 Training period: {train_data.index[0].date()} to {train_data.index[-1].date()}")
        print(f"📅 Testing period: {test_data.index[-test_size:].min().date()} to {test_data.index[-1].date()}")
        
        # Verify no future data leakage
        if test_data.index[-1].date() > self.data_cutoff_date:
            print(f"⚠️ WARNING: Test data extends beyond cutoff date!")
            print(f"   Test end: {test_data.index[-1].date()}")
            print(f"   Cutoff: {self.data_cutoff_date}")
        
        # Scale data
        if params['scaler_type'] == 'robust':
            feature_scaler = RobustScaler()
            target_scaler = StandardScaler()
        elif params['scaler_type'] == 'standard':
            feature_scaler = StandardScaler()
            target_scaler = StandardScaler()
        else:
            feature_scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()
        
        train_features_scaled = feature_scaler.fit_transform(train_data[feature_cols])
        train_target = train_data['Return_future_5d'].values.reshape(-1, 1)
        train_target_scaled = target_scaler.fit_transform(train_target)
        
        test_features_scaled = feature_scaler.transform(test_data[feature_cols])
        test_target = test_data['Return_future_5d'].values.reshape(-1, 1)
        test_target_scaled = target_scaler.transform(test_target)
        
        # Create sequences
        X_train, y_train = self.create_sequences_optimized(train_features_scaled, 
                                                          train_target_scaled.flatten(), 
                                                          sequence_length)
        X_test, y_test = self.create_sequences_optimized(test_features_scaled, 
                                                        test_target_scaled.flatten(), 
                                                        sequence_length)
        
        print(f"📈 Training sequences: {X_train.shape}")
        print(f"📉 Testing sequences: {X_test.shape}")
        
        # Build optimized model
        input_size = len(feature_cols)
        model = self.build_optimized_model(input_size, params)
        
        # Display model summary
        if self.show_progress:
            print(f"\n🏗️ Model Architecture Summary:")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
        
        # Train model with full progress tracking
        print(f"\n🔥 Training optimized {params['model_type']} model...")
        start_time = time.time()
        
        history = self.train_pytorch_model(model, X_train, y_train, X_test, y_test, params, symbol)
        
        training_time = time.time() - start_time
        print(f"\n⏱️ Training completed in {training_time:.1f} seconds")
        
        # Evaluate model
        print(f"\n📊 Evaluating model performance...")
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred_scaled = model(X_test_tensor).squeeze().cpu().numpy()
        
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(y_test_original, y_pred)
        
        # Display final training plots
        if self.show_progress:
            self.display_final_training_results(symbol, history, y_test_original, y_pred, metrics)
        
        # Save model
        torch.save(model.state_dict(), f'best_pytorch_model_{symbol}.pth')
        
        # Store results
        self.models[symbol] = {
            'model': model,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'feature_cols': feature_cols,
            'params': params,
            'metrics': metrics,
            'history': history,
            'predictions': y_pred,
            'actual': y_test_original,
            'training_time': training_time,
            'data_cutoff_date': self.data_cutoff_date,
            'training_end_date': train_data.index[-1].date(),
            'test_end_date': test_data.index[-1].date()
        }
        
        print(f"\n✅ {symbol} - Final PyTorch Model Performance:")
        print(f"   R² Score: {metrics['r2']:.4f}")
        print(f"   Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"   Training Time: {training_time:.1f}s")
        print(f"   Device Used: {self.device}")
        print(f"   Data Cutoff: {self.data_cutoff_date}")
        print(f"   Training Data End: {train_data.index[-1].date()}")
        
        return self.models[symbol]
    
    def display_final_training_results(self, symbol, history, y_true, y_pred, metrics):
        """Display comprehensive final training results"""
        try:
            plt.figure(figsize=(20, 12))
            
            # 1. Training history
            plt.subplot(3, 4, 1)
            plt.plot(history['train_loss'], label='Training Loss', alpha=0.8)
            plt.plot(history['val_loss'], label='Validation Loss', alpha=0.8)
            plt.title(f'{symbol} - PyTorch Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. Loss convergence
            plt.subplot(3, 4, 2)
            if len(history['val_loss']) > 1:
                plt.plot(np.array(history['val_loss']) - np.array(history['train_loss']), 
                        label='Val-Train Loss', alpha=0.8, color='green')
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                plt.title('Overfitting Monitor')
                plt.xlabel('Epoch')
                plt.ylabel('Loss Difference')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # 3. Predictions vs Actual
            plt.subplot(3, 4, 3)
            plt.scatter(y_true, y_pred, alpha=0.6, s=30)
            
            # Perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            plt.xlabel('Actual Returns (%)')
            plt.ylabel('Predicted Returns (%)')
            plt.title(f'Predictions vs Actual (R²: {metrics["r2"]:.4f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 4. Residuals plot
            plt.subplot(3, 4, 4)
            residuals = y_true - y_pred
            plt.scatter(y_pred, residuals, alpha=0.6, s=30)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            plt.xlabel('Predicted Returns (%)')
            plt.ylabel('Residuals (%)')
            plt.title('Residuals Plot')
            plt.grid(True, alpha=0.3)
            
            # 5. Prediction timeline
            plt.subplot(3, 4, 5)
            plt.plot(y_true, label='Actual', alpha=0.8, linewidth=2)
            plt.plot(y_pred, label='Predicted', alpha=0.8, linewidth=2)
            plt.title('Prediction Timeline')
            plt.xlabel('Time Step')
            plt.ylabel('Returns (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 6. Error distribution
            plt.subplot(3, 4, 6)
            errors = np.abs(y_true - y_pred)
            plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(errors), color='red', linestyle='--', 
                       label=f'Mean Error: {np.mean(errors):.3f}%')
            plt.xlabel('Absolute Error (%)')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 7. Direction accuracy analysis
            plt.subplot(3, 4, 7)
            actual_direction = np.sign(y_true)
            pred_direction = np.sign(y_pred)
            
            # Confusion matrix for directions
            correct_up = np.sum((actual_direction > 0) & (pred_direction > 0))
            correct_down = np.sum((actual_direction < 0) & (pred_direction < 0))
            wrong_up = np.sum((actual_direction < 0) & (pred_direction > 0))
            wrong_down = np.sum((actual_direction > 0) & (pred_direction < 0))
            
            confusion = np.array([[correct_down, wrong_down], [wrong_up, correct_up]])
            
            plt.imshow(confusion, cmap='Blues')
            plt.colorbar()
            plt.xticks([0, 1], ['Pred Down', 'Pred Up'])
            plt.yticks([0, 1], ['Actual Down', 'Actual Up'])
            plt.title(f'Direction Confusion Matrix\nAccuracy: {metrics["direction_accuracy"]:.1f}%')
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(confusion[i, j]), ha='center', va='center', 
                            color='white' if confusion[i, j] > confusion.max()/2 else 'black')
            
            # 8. Cumulative returns comparison
            plt.subplot(3, 4, 8)
            actual_cum_returns = np.cumprod(1 + y_true/100)
            pred_cum_returns = np.cumprod(1 + y_pred/100)
            
            plt.plot(actual_cum_returns, label='Actual Strategy', alpha=0.8, linewidth=2)
            plt.plot(pred_cum_returns, label='Predicted Strategy', alpha=0.8, linewidth=2)
            plt.title('Cumulative Returns Comparison')
            plt.xlabel('Time Step')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 9. Performance metrics summary
            plt.subplot(3, 4, 9)
            metrics_text = f"""
PYTORCH MODEL METRICS

R² Score: {metrics['r2']:.4f}
Direction Accuracy: {metrics['direction_accuracy']:.1f}%
RMSE: {metrics['rmse']:.4f}%
MAE: {metrics['mae']:.4f}%

FINANCIAL METRICS

Sharpe Ratio: {metrics['sharpe_ratio']:.4f}
Information Ratio: {metrics['information_ratio']:.4f}
Max Drawdown: {metrics['max_drawdown']:.2f}%
Hit Rate: {metrics['hit_rate']:.1f}%
Gain/Loss Ratio: {metrics['gain_loss_ratio']:.2f}
Volatility: {metrics['volatility']:.2f}%

DATA INTEGRITY
Cutoff Date: {self.data_cutoff_date}
No Future Data Used ✅
            """
            plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            plt.axis('off')
            
            # 10. Learning curves analysis
            plt.subplot(3, 4, 10)
            # Overfitting analysis
            train_loss = history['train_loss']
            val_loss = history['val_loss']
            if len(val_loss) >= 10:
                overfitting_score = np.mean(np.array(val_loss[-10:]) - np.array(train_loss[-10:]))
                
                plt.plot(np.array(val_loss) - np.array(train_loss), 
                        label=f'Val-Train Loss (Avg: {overfitting_score:.4f})', alpha=0.8)
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                plt.title('Overfitting Analysis')
                plt.xlabel('Epoch')
                plt.ylabel('Validation - Training Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # 11. Prediction confidence analysis
            plt.subplot(3, 4, 11)
            # Calculate prediction confidence based on consistency
            prediction_strength = np.abs(y_pred)
            actual_strength = np.abs(y_true)
            
            plt.scatter(prediction_strength, actual_strength, alpha=0.6, s=30)
            plt.xlabel('Prediction Strength (|Predicted|)')
            plt.ylabel('Actual Strength (|Actual|)')
            plt.title('Prediction Confidence Analysis')
            plt.grid(True, alpha=0.3)
            
            # 12. Model architecture info
            plt.subplot(3, 4, 12)
            model_info = self.models[symbol]['params']
            architecture_text = f"""
PYTORCH MODEL INFO

Type: {model_info.get('model_type', 'N/A')}
Sequence Length: {model_info.get('sequence_length', 'N/A')}
Features: {len(self.models[symbol]['feature_cols'])}

HYPERPARAMETERS

Learning Rate: {model_info.get('learning_rate', 'N/A'):.6f}
Batch Size: {model_info.get('batch_size', 'N/A')}
Dropout Rate: {model_info.get('dropout_rate', 'N/A'):.3f}
LSTM Units: {model_info.get('lstm_units_1', 'N/A')}
Scaler: {model_info.get('scaler_type', 'N/A')}

TRAINING INFO

Training Time: {self.models[symbol].get('training_time', 0):.1f}s
Epochs Trained: {history.get('epochs_trained', 'N/A')}
Device: {device}
Best Val Loss: {history.get('best_val_loss', 'N/A'):.6f}

DATA CUTOFF
Cutoff Date: {self.data_cutoff_date}
Training End: {self.models[symbol].get('training_end_date', 'N/A')}
            """
            plt.text(0.1, 0.9, architecture_text, transform=plt.gca().transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
            plt.axis('off')
            
            plt.tight_layout()
            plt.suptitle(f'🚀 PYTORCH TRAINING RESULTS - {symbol}', fontsize=16, y=0.98)
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Results plot error: {e}")
    
    def calculate_comprehensive_metrics(self, y_true, y_pred):
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Direction accuracy
        metrics['direction_accuracy'] = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
        
        # Correlation
        metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        
        # Financial metrics
        returns = y_pred / 100  # Convert percentage to decimal
        metrics['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak * 100
        metrics['max_drawdown'] = np.min(drawdown)
        
        # Hit rate
        metrics['hit_rate'] = np.mean(returns > 0) * 100
        
        # Average gains and losses
        gains = returns[returns > 0]
        losses = returns[returns <= 0]
        metrics['avg_gain'] = np.mean(gains) * 100 if len(gains) > 0 else 0
        metrics['avg_loss'] = np.mean(losses) * 100 if len(losses) > 0 else 0
        metrics['gain_loss_ratio'] = abs(metrics['avg_gain'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else float('inf')
        
        # Volatility
        metrics['volatility'] = np.std(returns) * np.sqrt(252) * 100
        
        # Information ratio
        benchmark_return = 0.1 / 252  # Assume 10% annual benchmark
        excess_returns = returns - benchmark_return
        metrics['information_ratio'] = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
        
        return metrics
    
    def predict_optimized(self, symbol, days_ahead=5):
        """Make optimized predictions with confidence analysis and strict data handling"""
        
        if symbol not in self.models:
            print(f"❌ No trained model found for {symbol}")
            return None
        
        print(f"🔮 Making optimized PyTorch prediction for {symbol}...")
        print(f"🚫 Using only data up to: {self.data_cutoff_date} (end of last week)")
        print(f"🎯 Predicting for: current week ending {self.prediction_target_date}")
        
        model_data = self.models[symbol]
        model = model_data['model']
        feature_scaler = model_data['feature_scaler']
        target_scaler = model_data['target_scaler']
        feature_cols = model_data['feature_cols']
        params = model_data['params']
        
        # Prepare recent data with strict cutoff
        df = self.data[symbol].copy()
        
        # Ensure we don't use any data beyond our cutoff
        cutoff_datetime = pd.Timestamp(self.data_cutoff_date)
        df = df[df.index <= cutoff_datetime]
        
        df_with_indicators = self.calculate_optimized_technical_indicators(df, params)
        
        # Get latest features (before cutoff)
        sequence_length = params['sequence_length']
        latest_features = df_with_indicators[feature_cols].tail(sequence_length).values
        
        # Verify we have enough data
        if len(latest_features) < sequence_length:
            print(f"❌ Insufficient data for prediction. Need {sequence_length}, have {len(latest_features)}")
            return None
        
        # Scale features
        latest_scaled = feature_scaler.transform(latest_features)
        X_latest = torch.FloatTensor(latest_scaled).unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_latest).squeeze().cpu().numpy()
        
        pred_return = target_scaler.inverse_transform([[pred_scaled]])[0][0]
        
        # Current price (from our cutoff date)
        current_price = df['Close'].iloc[-1]
        predicted_price = current_price * (1 + pred_return/100)
        
        # Calculate confidence based on model performance
        metrics = model_data['metrics']
        confidence = (metrics['r2'] * 0.4 + metrics['direction_accuracy']/100 * 0.4 + 
                     min(metrics['sharpe_ratio'], 2)/2 * 0.2) * 100
        
        # Risk assessment
        volatility = metrics['volatility']
        max_drawdown = metrics['max_drawdown']
        
        risk_level = 'Low'
        if volatility > 30 or max_drawdown < -10:
            risk_level = 'High'
        elif volatility > 20 or max_drawdown < -5:
            risk_level = 'Medium'
        
        prediction_result = {
            'symbol': symbol,
            'model_type': params['model_type'],
            'current_price': float(current_price),
            'predicted_return_pct': float(pred_return),
            'predicted_price': float(predicted_price),
            'confidence': float(max(0, min(100, confidence))),
            'risk_level': risk_level,
            'direction': 'BUY' if pred_return > 0 else 'SELL',
            'model_metrics': metrics,
            'device': str(self.device),
            'data_cutoff_date': str(self.data_cutoff_date),
            'prediction_date': str(self.prediction_target_date),
            'current_price_date': str(df.index[-1].date()),
            'days_ahead': days_ahead,
            'optimization_score': model_data.get('optimization_score', 'N/A')
        }
        
        print(f"   📅 Prediction based on data ending: {df.index[-1].date()} (last week)")
        print(f"   🎯 Predicting for: {self.prediction_target_date} (current week)")
        print(f"   📊 Current price: ${current_price:.2f} (as of {df.index[-1].date()})")
        print(f"   🔮 Predicted return: {pred_return:+.2f}% for current week")
        print(f"   🎯 Direction: {prediction_result['direction']}")
        
        return prediction_result
    
    def generate_optimized_report(self):
        """Generate comprehensive optimized report with visualizations and strict data handling"""
        
        print("="*100)
        print("🚀 OPTIMIZED PYTORCH LSTM STOCK PREDICTION REPORT")
        print("="*100)
        print(f"📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Analyzed Stocks: {', '.join(self.symbols)}")
        print(f"🧠 Optimization Trials: {self.optimization_trials} per stock")
        print(f"📊 Prediction Period: Current week from {self.data_cutoff_date}")
        print(f"🔥 Device: {self.device}")
        print(f"⚡ CUDA Available: {torch.cuda.is_available()}")
        print(f"🚫 Data Cutoff: {self.data_cutoff_date} (No future data used)")
        
        all_predictions = {}
        
        for symbol in self.symbols:
            if symbol in self.models:
                print(f"\n{'='*80}")
                print(f"📈 {symbol} - OPTIMIZED PYTORCH ANALYSIS")
                print(f"{'='*80}")
                
                model_data = self.models[symbol]
                params = model_data['params']
                metrics = model_data['metrics']
                
                print(f"🏆 BEST MODEL CONFIGURATION:")
                print(f"  Architecture: {params['model_type']}")
                print(f"  Sequence Length: {params['sequence_length']}")
                print(f"  Features: {len(model_data['feature_cols'])}")
                print(f"  Scaler: {params['scaler_type']}")
                print(f"  Training Time: {model_data.get('training_time', 0):.1f}s")
                print(f"  Device: {self.device}")
                print(f"  Data Cutoff: {model_data.get('data_cutoff_date', 'N/A')}")
                
                print(f"\n🎯 OPTIMIZED PERFORMANCE:")
                print(f"  R² Score: {metrics['r2']:.4f}")
                print(f"  Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
                print(f"  Information Ratio: {metrics['information_ratio']:.4f}")
                print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
                print(f"  Hit Rate: {metrics['hit_rate']:.1f}%")
                print(f"  Gain/Loss Ratio: {metrics['gain_loss_ratio']:.2f}")
                
                # Make prediction
                prediction = self.predict_optimized(symbol)
                if prediction:
                    all_predictions[symbol] = prediction
                    print(f"\n🔮 OPTIMIZED PREDICTION:")
                    print(f"  Current Price: ${prediction['current_price']:.2f} (as of {prediction['current_price_date']})")
                    print(f"  Predicted Price (5d): ${prediction['predicted_price']:.2f}")
                    print(f"  Expected Return: {prediction['predicted_return_pct']:+.2f}%")
                    print(f"  Direction: {prediction['direction']}")
                    print(f"  Confidence: {prediction['confidence']:.1f}%")
                    print(f"  Risk Level: {prediction['risk_level']}")
                    print(f"  Data Cutoff: {prediction['data_cutoff_date']} (last week)")
                    print(f"  Prediction Target: {prediction['prediction_date']} (current week)")
        
        # Generate summary visualizations
        if all_predictions and self.show_progress:
            self.display_portfolio_summary(all_predictions)
        
        # Summary recommendations
        if all_predictions:
            print(f"\n{'='*100}")
            print("🎯 OPTIMIZED PYTORCH RECOMMENDATIONS")
            print(f"{'='*100}")
            
            # Sort by confidence and expected return
            high_confidence = [(s, p) for s, p in all_predictions.items() 
                             if p['confidence'] >= 80 and p['risk_level'] in ['Low', 'Medium']]
            medium_confidence = [(s, p) for s, p in all_predictions.items() 
                               if 65 <= p['confidence'] < 80]
            
            print(f"\n🔥 HIGH CONFIDENCE PICKS ({len(high_confidence)} stocks):")
            for symbol, pred in sorted(high_confidence, key=lambda x: x[1]['confidence'], reverse=True):
                print(f"  {symbol}: {pred['direction']} {pred['predicted_return_pct']:+.2f}% "
                      f"(Confidence: {pred['confidence']:.1f}%, Risk: {pred['risk_level']})")
            
            print(f"\n⭐ MEDIUM CONFIDENCE PICKS ({len(medium_confidence)} stocks):")
            for symbol, pred in sorted(medium_confidence, key=lambda x: x[1]['confidence'], reverse=True):
                print(f"  {symbol}: {pred['direction']} {pred['predicted_return_pct']:+.2f}% "
                      f"(Confidence: {pred['confidence']:.1f}%)")
            
            print(f"🔒 DATA INTEGRITY SUMMARY:")
            print(f"  ✅ All training used data only up to: {self.data_cutoff_date} (last week)")
            print(f"  ✅ No current week data leakage in any model")
            print(f"  ✅ Predictions are for current week: {self.prediction_target_date}")
            print(f"  ✅ Safe historical data training maintained")
        
        return all_predictions
    
    def display_portfolio_summary(self, predictions):
        """Display portfolio-level summary visualizations with data integrity info"""
        try:
            plt.figure(figsize=(20, 12))
            
            symbols = list(predictions.keys())
            returns = [predictions[s]['predicted_return_pct'] for s in symbols]
            confidences = [predictions[s]['confidence'] for s in symbols]
            risk_levels = [predictions[s]['risk_level'] for s in symbols]
            current_prices = [predictions[s]['current_price'] for s in symbols]
            
            # 1. Expected returns
            plt.subplot(3, 4, 1)
            colors = ['green' if r > 0 else 'red' for r in returns]
            bars = plt.bar(symbols, returns, color=colors, alpha=0.7)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title('Expected Returns (current week)\n🚫 No Current Week Data Used')
            plt.ylabel('Return (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, return_val in zip(bars, returns):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.1),
                        f'{return_val:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
            
            # 2. Confidence levels
            plt.subplot(3, 4, 2)
            colors = ['darkgreen' if c >= 80 else 'orange' if c >= 65 else 'red' for c in confidences]
            plt.bar(symbols, confidences, color=colors, alpha=0.7)
            plt.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='High Confidence')
            plt.axhline(y=65, color='orange', linestyle='--', alpha=0.7, label='Medium Confidence')
            plt.title('Prediction Confidence\n✅ Training: Last Week, Predicting: This Week')
            plt.ylabel('Confidence (%)')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 3. Risk vs Return
            plt.subplot(3, 4, 3)
            risk_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            for i, (symbol, risk) in enumerate(zip(symbols, risk_levels)):
                plt.scatter(returns[i], confidences[i], 
                           c=risk_colors[risk], s=100, alpha=0.7, label=risk if i == 0 or risk not in [risk_levels[j] for j in range(i)] else "")
                plt.annotate(symbol, (returns[i], confidences[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel('Expected Return (%)')
            plt.ylabel('Confidence (%)')
            plt.title('Risk vs Return Analysis')
            plt.axhline(y=75, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 4. Portfolio allocation suggestion
            plt.subplot(3, 4, 4)
            # Simple allocation based on confidence and positive returns
            positive_returns = [(s, p) for s, p in predictions.items() if p['predicted_return_pct'] > 0]
            if positive_returns:
                weights = []
                for symbol, pred in positive_returns:
                    confidence_weight = pred['confidence'] / 100
                    return_weight = max(0, pred['predicted_return_pct']) / 10  # Scale returns
                    risk_weight = {'Low': 1.0, 'Medium': 0.7, 'High': 0.3}[pred['risk_level']]
                    weights.append(confidence_weight * return_weight * risk_weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight * 100 for w in weights]
                    
                    plt.pie(normalized_weights, labels=[s for s, _ in positive_returns], autopct='%1.1f%%')
                    plt.title('Suggested Portfolio Allocation\n(Positive Returns Only)')
            else:
                plt.text(0.5, 0.5, 'No positive return\npredictions', ha='center', va='center')
                plt.title('Suggested Portfolio Allocation')
            
            # 5. Model performance comparison
            plt.subplot(3, 4, 5)
            r2_scores = [predictions[s]['model_metrics']['r2'] for s in symbols]
            plt.bar(symbols, r2_scores, alpha=0.7, color='skyblue')
            plt.title('Model R² Scores')
            plt.ylabel('R² Score')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 6. Direction accuracy comparison
            plt.subplot(3, 4, 6)
            dir_accuracies = [predictions[s]['model_metrics']['direction_accuracy'] for s in symbols]
            plt.bar(symbols, dir_accuracies, alpha=0.7, color='lightcoral')
            plt.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Baseline (60%)')
            plt.title('Direction Accuracy')
            plt.ylabel('Accuracy (%)')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 7. Sharpe ratios
            plt.subplot(3, 4, 7)
            sharpe_ratios = [predictions[s]['model_metrics']['sharpe_ratio'] for s in symbols]
            colors = ['green' if sr > 1 else 'orange' if sr > 0.5 else 'red' for sr in sharpe_ratios]
            plt.bar(symbols, sharpe_ratios, color=colors, alpha=0.7)
            plt.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Good (>1.0)')
            plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Decent (>0.5)')
            plt.title('Sharpe Ratios')
            plt.ylabel('Sharpe Ratio')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 8. Max drawdown comparison
            plt.subplot(3, 4, 8)
            max_drawdowns = [predictions[s]['model_metrics']['max_drawdown'] for s in symbols]
            colors = ['green' if md > -5 else 'orange' if md > -10 else 'red' for md in max_drawdowns]
            plt.bar(symbols, max_drawdowns, color=colors, alpha=0.7)
            plt.axhline(y=-5, color='orange', linestyle='--', alpha=0.7, label='Caution (-5%)')
            plt.axhline(y=-10, color='red', linestyle='--', alpha=0.7, label='High Risk (-10%)')
            plt.title('Maximum Drawdown')
            plt.ylabel('Max Drawdown (%)')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 9. Current prices
            plt.subplot(3, 4, 9)
            plt.bar(symbols, current_prices, alpha=0.7, color='purple')
            plt.title(f'Current Stock Prices\n(as of {self.data_cutoff_date})')
            plt.ylabel('Price ($)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 10. Model types used
            plt.subplot(3, 4, 10)
            model_types = [predictions[s]['model_type'] for s in symbols]
            model_counts = {}
            for mt in model_types:
                model_counts[mt] = model_counts.get(mt, 0) + 1
            
            plt.pie(model_counts.values(), labels=model_counts.keys(), autopct='%1.0f')
            plt.title('Optimized Model Types')
            
            # 11. Risk distribution
            plt.subplot(3, 4, 11)
            risk_counts = {}
            for rl in risk_levels:
                risk_counts[rl] = risk_counts.get(rl, 0) + 1
            
            risk_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            colors = [risk_colors[risk] for risk in risk_counts.keys()]
            plt.pie(risk_counts.values(), labels=risk_counts.keys(), autopct='%1.0f', colors=colors)
            plt.title('Risk Level Distribution')
            
            # 12. Summary statistics with data integrity
            plt.subplot(3, 4, 12)
            avg_return = np.mean(returns)
            avg_confidence = np.mean(confidences)
            positive_predictions = sum(1 for r in returns if r > 0)
            
            stats_text = f"""
PYTORCH PORTFOLIO SUMMARY

Total Stocks: {len(symbols)}
Positive Predictions: {positive_predictions}/{len(symbols)}
Average Return: {avg_return:+.2f}%
Average Confidence: {avg_confidence:.1f}%

RISK BREAKDOWN
Low Risk: {risk_counts.get('Low', 0)} stocks
Medium Risk: {risk_counts.get('Medium', 0)} stocks
High Risk: {risk_counts.get('High', 0)} stocks

TOP PERFORMER
Best Return: {max(returns):+.2f}%
Stock: {symbols[returns.index(max(returns))]}

HIGHEST CONFIDENCE
Best Confidence: {max(confidences):.1f}%
Stock: {symbols[confidences.index(max(confidences))]}

DATA INTEGRITY ✅
Cutoff: {self.data_cutoff_date}
No Current Week Data Used
Training: Last Week Only
Predicting: Current Week
Device: {device}
CUDA: {torch.cuda.is_available()}
            """
            plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            plt.axis('off')
            
            plt.tight_layout()
            plt.suptitle('🚀 PYTORCH PORTFOLIO ANALYSIS DASHBOARD - DATA INTEGRITY VERIFIED', fontsize=16, y=0.98)
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Portfolio summary plot error: {e}")
    
    def save_optimization_results(self, filename='pytorch_optimization_results.pkl'):
        """Save all optimization results with data integrity info"""
        results = {
            'best_params': self.best_params,
            'models': {k: {
                'params': v['params'],
                'metrics': v['metrics'],
                'feature_cols': v['feature_cols'],
                'training_time': v.get('training_time', 0),
                'data_cutoff_date': v.get('data_cutoff_date', self.data_cutoff_date),
                'training_end_date': v.get('training_end_date', 'N/A'),
                'test_end_date': v.get('test_end_date', 'N/A')
            } for k, v in self.models.items()},
            'symbols': self.symbols,
            'optimization_trials': self.optimization_trials,
            'device': str(self.device),
            'data_cutoff_date': str(self.data_cutoff_date),
            'prediction_target_date': str(self.prediction_target_date),
            'studies_info': {k: {
                'best_value': v.best_value,
                'n_trials': len(v.trials),
                'best_trial_number': v.best_trial.number if v.best_trial else None
            } for k, v in self.studies.items()}
        }
        
        joblib.dump(results, filename)
        print(f"💾 PyTorch optimization results saved to {filename}")
        print(f"🔒 Data integrity preserved - cutoff date: {self.data_cutoff_date}")
    
    def load_optimization_results(self, filename='pytorch_optimization_results.pkl'):
        """Load optimization results"""
        try:
            results = joblib.load(filename)
            self.best_params = results['best_params']
            # Restore data cutoff date if available
            if 'data_cutoff_date' in results:
                self.data_cutoff_date = datetime.strptime(results['data_cutoff_date'], '%Y-%m-%d').date()
            print(f"📖 PyTorch optimization results loaded from {filename}")
            print(f"🔒 Data cutoff preserved: {self.data_cutoff_date}")
            return True
        except:
            print(f"❌ Could not load results from {filename}")
            return False
    
    def create_live_monitoring_dashboard(self):
        """Create a live monitoring dashboard for ongoing predictions"""
        if not self.show_progress:
            return
            
        try:
            # This would create a live dashboard in a Jupyter environment
            from IPython.display import HTML, display
            
            dashboard_html = """
            <div style="border: 2px solid #4CAF50; padding: 20px; margin: 10px; border-radius: 10px;">
                <h2 style="color: #4CAF50;">🚀 PyTorch LSTM Predictor Live Dashboard</h2>
                <p>Real-time monitoring of your optimized PyTorch LSTM models</p>
                <div style="display: flex; justify-content: space-between;">
                    <div style="text-align: center;">
                        <h3>Active Models</h3>
                        <p style="font-size: 24px; color: #2196F3;">{}</p>
                    </div>
                    <div style="text-align: center;">
                        <h3>Avg Confidence</h3>
                        <p style="font-size: 24px; color: #FF9800;">{:.1f}%</p>
                    </div>
                    <div style="text-align: center;">
                        <h3>High Confidence Picks</h3>
                        <p style="font-size: 24px; color: #4CAF50;">{}</p>
                    </div>
                    <div style="text-align: center;">
                        <h3>Device</h3>
                        <p style="font-size: 18px; color: #9C27B0;">{}</p>
                    </div>
                    <div style="text-align: center;">
                        <h3>Data Cutoff</h3>
                        <p style="font-size: 16px; color: #FF5722;">{}</p>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    <p style="color: #4CAF50; font-weight: bold;">✅ Training: Last Week Data | Predicting: Current Week</p>
                </div>
            </div>
            """.format(
                len(self.models),
                np.mean([self.predict_optimized(s)['confidence'] for s in self.models.keys()]) if self.models else 0,
                len([s for s in self.models.keys() if self.predict_optimized(s)['confidence'] > 80]) if self.models else 0,
                str(self.device),
                str(self.data_cutoff_date)
            )
            
            display(HTML(dashboard_html))
            
        except ImportError:
            print("📊 Live dashboard requires Jupyter environment")
            print(f"🔒 Data cutoff: {self.data_cutoff_date} (last week)")
            print(f"✅ Predicting for current week using last week's data")
        except Exception as e:
            print(f"⚠️ Dashboard error: {e}")
    
    def load_pretrained_model(self, symbol, model_path):
        """Load a pretrained PyTorch model"""
        if symbol not in self.best_params:
            print(f"❌ No parameters found for {symbol}")
            return None
        
        try:
            params = self.best_params[symbol]
            input_size = len(params.get('feature_cols', []))
            
            # Build model architecture
            model = self.build_optimized_model(input_size, params)
            
            # Load state dict
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            
            print(f"✅ Loaded pretrained model for {symbol} from {model_path}")
            print(f"🔒 Model trained with data cutoff: {self.data_cutoff_date}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model for {symbol}: {e}")
            return None

def main_optimized():
    """Main function for optimized PyTorch LSTM predictor that trains on last week's data to predict current week"""
    
    print("🌟 OPTIMIZED PYTORCH LSTM STOCK PREDICTOR - CURRENT WEEK PREDICTIONS 🌟")
    print("="*80)
    
    if not HAS_PYTORCH:
        print("❌ PyTorch not available")
        print("Please install: pip install torch optuna scikit-learn tqdm")
        return
    
    print("✅ All dependencies ready - Starting PyTorch optimization")
    print(f"🔥 Device: {device}")
    print(f"⚡ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
    
    # Configuration options
    print("\n🎯 Configuration Options:")
    show_progress = input("Show detailed progress and visualizations? (y/n, default y): ").lower() != 'n'
    
    # Select stocks
    print("\n🎯 Select stocks for optimization:")
    print("1. Tech Giants (AAPL, MSFT, GOOGL)")
    print("2. AI & Semiconductors (NVDA, AMD, INTC)")
    print("3. Growth Stocks (TSLA, NFLX, AMZN)")
    print("4. Financial Sector (JPM, BAC, WFC)")
    print("5. Energy Sector (XOM, CVX, COP)")
    print("6. Custom selection")
    
    choice = input("Choose (1-6): ").strip()
    
    if choice == '1':
        symbols = ['AAPL', 'MSFT', 'GOOGL']
    elif choice == '2':
        symbols = ['NVDA', 'AMD', 'INTC']
    elif choice == '3':
        symbols = ['TSLA', 'NFLX', 'AMZN']
    elif choice == '4':
        symbols = ['JPM', 'BAC', 'WFC']
    elif choice == '5':
        symbols = ['XOM', 'CVX', 'COP']
    elif choice == '6':
        custom_input = input("Enter stock symbols (comma separated): ").strip().upper()
        symbols = [s.strip() for s in custom_input.split(',')]
    else:
        symbols = ['AAPL', 'MSFT']
    
    # Set optimization parameters
    print("\n🔧 Optimization Settings:")
    trials = int(input("Number of optimization trials per stock (20-200, default 50): ") or "50")
    use_cache = input("Use cached results if available? (y/n, default y): ").lower() != 'n'
    
    print(f"\n🎯 Selected stocks: {', '.join(symbols)}")
    print(f"🔥 Optimization trials: {trials} per stock")
    print(f"💾 Cache usage: {'Enabled' if use_cache else 'Disabled'}")
    print(f"📊 Progress visualization: {'Enabled' if show_progress else 'Disabled'}")
    print(f"🚀 Using PyTorch with device: {device}")
    
    # Initialize predictor with strict data handling
    predictor = OptimizedLSTMPredictor(
        symbols=symbols, 
        optimization_trials=trials,
        use_cache=use_cache,
        show_progress=show_progress
    )
    
    print(f"\n🔒 DATA INTEGRITY VERIFICATION:")
    print(f"   Training data cutoff: {predictor.data_cutoff_date} (end of last week)")
    print(f"   Prediction target: {predictor.prediction_target_date} (current week)")
    print(f"   ✅ No current week data will be used in training")
    print(f"   🎯 Model will predict current week using last week's data")
    
    # Load data with strict cutoff
    print("\n📥 Loading market data with strict cutoff...")
    try:
        start_time = time.time()
        data = predictor.load_market_data(start_date="2018-01-01")
        load_time = time.time() - start_time
        
        if not data:
            print("❌ Cannot download data")
            return None
            
        print(f"✅ Data loaded in {load_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None
    
    # Display initial data summary with cutoff verification
    if show_progress:
        print("\n📊 Data Summary with Cutoff Verification:")
        for symbol, df in data.items():
            latest_date = df.index[-1].date()
            days_safe = (predictor.data_cutoff_date - latest_date).days
            status = "✅ SAFE" if days_safe >= 0 else "❌ UNSAFE"
            print(f"  {symbol}: {len(df)} days, from {df.index[0].date()} to {latest_date} [{status}, margin: {days_safe} days]")
    
    # Optimization phase
    print("\n🚀 Starting PyTorch Optimization Phase...")
    print("="*60)
    
    successful_optimizations = []
    optimization_start_time = time.time()
    
    for i, symbol in enumerate(symbols):
        if symbol in data:
            try:
                print(f"\n🔥 Optimizing {symbol} ({i+1}/{len(symbols)})...")
                symbol_start_time = time.time()
                
                best_params, study = predictor.optimize_hyperparameters(symbol)
                
                symbol_optimization_time = time.time() - symbol_start_time
                successful_optimizations.append(symbol)
                
                print(f"✅ {symbol} optimization completed in {symbol_optimization_time:.1f}s")
                print(f"🏆 Best objective: {study.best_value:.4f}")
                print(f"📈 Total trials: {len(study.trials)}")
                
            except Exception as e:
                print(f"❌ Optimization error for {symbol}: {e}")
                import traceback
                if show_progress:
                    traceback.print_exc()
    
    total_optimization_time = time.time() - optimization_start_time
    
    if not successful_optimizations:
        print("❌ No successful optimizations")
        return None
    
    print(f"\n✅ Optimization phase completed in {total_optimization_time:.1f}s")
    print(f"🎯 Successfully optimized: {', '.join(successful_optimizations)}")
    
    # Training phase
    print(f"\n🚀 Training Optimized PyTorch Models...")
    print("="*60)
    
    successful_models = []
    training_start_time = time.time()
    
    for i, symbol in enumerate(successful_optimizations):
        try:
            print(f"\n🔥 Training optimized PyTorch model for {symbol} ({i+1}/{len(successful_optimizations)})...")
            model_start_time = time.time()
            
            model_data = predictor.train_optimized_model(symbol)
            
            model_training_time = time.time() - model_start_time
            
            if model_data:
                successful_models.append(symbol)
                print(f"✅ {symbol} model trained in {model_training_time:.1f}s")
                print(f"🎯 Performance: R²={model_data['metrics']['r2']:.4f}, "
                      f"Dir={model_data['metrics']['direction_accuracy']:.1f}%")
                print(f"🔒 Data cutoff verified: {model_data.get('data_cutoff_date', 'N/A')}")
                
        except Exception as e:
            print(f"❌ Training error for {symbol}: {e}")
            import traceback
            if show_progress:
                traceback.print_exc()
    
    total_training_time = time.time() - training_start_time
    
    if not successful_models:
        print("❌ No successful model training")
        return None
    
    print(f"\n✅ Training phase completed in {total_training_time:.1f}s")
    print(f"🎯 Successfully trained: {', '.join(successful_models)}")
    
    # Generate final report
    print("\n📊 Generating Optimized PyTorch Prediction Report...")
    report_start_time = time.time()
    
    try:
        predictions = predictor.generate_optimized_report()
        
        report_time = time.time() - report_start_time
        
        # Save results
        predictor.save_optimization_results()
        
        # Create live dashboard if in appropriate environment
        if show_progress:
            predictor.create_live_monitoring_dashboard()
        
        total_time = time.time() - optimization_start_time
        
        print(f"\n🎉 PYTORCH OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✅ Optimized PyTorch models for: {', '.join(successful_models)}")
        print(f"⏱️ Total time: {total_time:.1f}s")
        print(f"   - Optimization: {total_optimization_time:.1f}s")
        print(f"   - Training: {total_training_time:.1f}s")
        print(f"   - Report: {report_time:.1f}s")
        print(f"🚀 Device: {device}")
        print(f"⚡ CUDA Used: {torch.cuda.is_available()}")
        print(f"🔒 Data cutoff enforced: {predictor.data_cutoff_date} (last week)")
        print(f"✅ TRAINING USES ONLY LAST WEEK'S DATA")
        print(f"🎯 PREDICTIONS ARE FOR CURRENT WEEK")
        
        # Performance summary
        if predictions:
            avg_confidence = np.mean([p['confidence'] for p in predictions.values()])
            avg_return = np.mean([p['predicted_return_pct'] for p in predictions.values()])
            high_confidence_count = len([p for p in predictions.values() if p['confidence'] > 80])
            
            print(f"\n📈 PYTORCH PERFORMANCE SUMMARY:")
            print(f"   Average Confidence: {avg_confidence:.1f}%")
            print(f"   Average Predicted Return: {avg_return:+.2f}%")
            print(f"   High Confidence Predictions: {high_confidence_count}/{len(predictions)}")
            print(f"   Prediction Period: Current week ending {predictor.prediction_target_date}")
        
        return predictor, predictions
        
    except Exception as e:
        print(f"❌ Report generation error: {e}")
        import traceback
        if show_progress:
            traceback.print_exc()
        return predictor, None

if __name__ == "__main__":
    try:
        print("🌟 LAUNCHING OPTIMIZED PYTORCH LSTM PREDICTOR WITH DATA INTEGRITY 🌟")
        print("🔥 Real-time progress tracking and comprehensive analysis")
        print("📊 Complete training visualization and optimization monitoring")
        print("🚀 Powered by PyTorch with GPU acceleration support")
        print("🔒 STRICT DATA CUTOFF - No future data leakage")
        print("="*80)
        
        result = main_optimized()
        
        if result is not None:
            predictor, predictions = result
            
            if predictor and predictions:
                print("\n🎉 PyTorch optimization and prediction completed successfully!")
                print("\n💡 Available commands for further analysis:")
                print("- predictor.predict_optimized('SYMBOL') for new predictions")
                print("- predictor.generate_optimized_report() for updated report")
                print("- predictor.models['SYMBOL'] for detailed model information")
                print("- predictor.best_params['SYMBOL'] for optimized parameters")
                print("- predictor.studies['SYMBOL'] for optimization study details")
                print("- predictor.display_portfolio_summary(predictions) for portfolio analysis")
                print("- predictor.save_optimization_results() to save all results")
                print("- torch.save(predictor.models['SYMBOL']['model'].state_dict(), 'model.pth') to save model")
                
                print("\n📁 Files created:")
                print("- best_params_pytorch_[SYMBOL].json: Optimal parameters for each stock")
                print("- optuna_pytorch_[SYMBOL].db: Optimization study database")
                print("- best_pytorch_model_[SYMBOL].pth: Trained PyTorch model files")
                print("- pytorch_optimization_results.pkl: Complete results package")
                
                print("\n🔄 For continuous monitoring:")
                print("- Use predictor.create_live_monitoring_dashboard() in Jupyter")
                print("- Run predictor.predict_optimized('SYMBOL') for real-time predictions")
                print("- Use predictor.load_pretrained_model('SYMBOL', 'path') to load saved models")
                
                print(f"\n🚀 System Information:")
                print(f"- Device: {device}")
                print(f"- PyTorch Version: {torch.__version__}")
                print(f"- CUDA Available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"- GPU: {torch.cuda.get_device_name(0)}")
                    print(f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                
                print(f"🔒 DATA INTEGRITY SUMMARY:")
                print(f"- Training data cutoff: {predictor.data_cutoff_date} (last week)")
                print(f"- Prediction target: {predictor.prediction_target_date} (current week)")
                print(f"- ✅ Training uses data up to last week only")
                print(f"- ✅ Predicting for current week using historical data")
                print(f"- ✅ No current week data contamination")
                
            else:
                print("\n⚠️ Optimization completed but with some issues")
                print("Check the error messages above for details")
        else:
            print("\n⚠️ Optimization failed")
            print("Please check your data connection and try again")
            
    except KeyboardInterrupt:
        print("\n\n👋 Optimization cancelled by user")
        print("Partial results may have been saved")
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease report this error for debugging")