import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
import random
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import copy
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients, DeepLift, Deconvolution, GradientShap, Occlusion

seed =50 #2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)



# time series input
features =20
# training epochs
epochs =200 #1000
# synthetic time series dataset
ts_len = 1980
# test dataset size
test_len =106
# temporal casual layer channels
channel_sizes = [9] * 3
# convolution kernel size
kernel_size =3 #5
dropout = 0.1


#ts = generate_time_series(ts_len)
train_ratio=0.7
#ts_diff_y = ts_diff(ts[:, 0])
#ts_diff = copy.deepcopy(ts)
#ts_diff[:, 0] = ts_diff_y
l1_factor=3*10^(-10)
l2_factor=3*10^(-10)

C=[0,1,2]
lr2=[0.0001,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001,0.0002,0.0003]



#E=np.concatenate((X_no3[:,np.newaxis],X_po4[:,np.newaxis]),axis=1)
#E=[X_no3[:,np.newaxis],X_po4[:,np.newaxis]]
#print("E",E.shape)



N=24

prediction1=np.zeros((4,264))
prediction2=np.zeros((4,264))
prediction3=np.zeros((4,264))
prediction4=np.zeros((4,264))
prediction5=np.zeros((4,264))
prediction6=np.zeros((4,264))
prediction7=np.zeros((4,264))
prediction8=np.zeros((4,264))
prediction9=np.zeros((4,264))
prediction10=np.zeros((4,264))



#print("X_no3",X_no3.shape)
#print("X_npp",X_npp.shape)


#npp_index=npp_index.T
#print("npp_index_GFDL",npp_index.shape)



k=0
#M=[3,6,9,12,15,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,156,162,168,174,180,186,192,198,204,210,216,222,228,234,240,246,252,258,264]

#M=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144,147,150,153,156,159,162,165,168,171,174,177,180,183,186,189,192,195,198,201,204,207,210,213,216,219,222,225,228,231,234,237,240,243,246,249,252,255,258,261,264]
M=np.arange(1,24)

years=np.arange(1850,2015,1/12)

class EarlyStopping:

    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt',trace_func=print):

        #Args:
        """
                            Default: False
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.val_loss_min = np.Inf
        self.trace_func = trace_func

        
    def __call__(self, val_loss, model):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.W = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        energy = torch.tanh(self.W(x))  # shape: (batch_size, seq_length, hidden_size)
        energy = energy.view(x.size(0), -1, self.num_heads, self.head_size)  # shape: (batch_size, seq_length, num_heads, head_size)
        energy = energy.permute(0, 2, 1, 3)  # shape: (batch_size, num_heads, seq_length, head_size)

        attention_weights = torch.softmax(self.V(energy), dim=2)  # shape: (batch_size, num_heads, seq_length, 1)
        context_vector = torch.sum(attention_weights * energy, dim=2)  # shape: (batch_size, num_heads, head_size)

        context_vector = context_vector.view(x.size(0), -1)  # shape: (batch_size, hidden_size)
        return context_vector


class TemporalCasualLayer(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout ):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {
            'kernel_size': kernel_size,
            'stride':      1,
            'padding':     padding,
            'dilation':    dilation
        }

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        self.crop1 = Crop(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.conv3 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop3 = Crop(padding)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.conv4 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop4 = Crop(padding)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout)

        self.conv5 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop5 = Crop(padding)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1,
                                 self.conv2, self.crop2, self.relu2, self.dropout2,
                                 self.conv3, self.crop3, self.relu3, self.dropout3,
                                 self.conv4, self.crop4, self.relu4, self.dropout4
                                 
                                
                                 )
        self.residual = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.residual is not None:
            self.residual.weight.data.normal_(0, 0.01)

    def forward(self, x):
        residual = x if self.residual is None else self.residual(x)
        y = self.net(x)

        output = self.relu(y + residual)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.W = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        energy = torch.tanh(self.W(x))  # shape: (batch_size, seq_length, hidden_size)
        energy = energy.view(x.size(0), -1, self.num_heads, self.head_size)  # shape: (batch_size, seq_length, num_heads, head_size)
        energy = energy.permute(0, 2, 1, 3)  # shape: (batch_size, num_heads, seq_length, head_size)

        attention_weights = torch.softmax(self.V(energy), dim=2)  # shape: (batch_size, num_heads, seq_length, 1)
        context_vector = torch.sum(attention_weights * energy, dim=2)  # shape: (batch_size, num_heads, head_size)

        context_vector = context_vector.view(x.size(0), -1)  # shape: (batch_size, hidden_size)
        return context_vector

class Crop(nn.Module):

    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[:, :, :-self.crop_size].contiguous()
    

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        # input is dim (N, in_channels, T) where N is the batch_size, and T is the sequence length
        mask = np.array([[1 if i>j else 0 for i in range(input.size(2))] for j in range(input.size(2))])
        if input.is_cuda:
            mask = torch.ByteTensor(mask).cuda(input.get_device())
        else:
            mask = torch.ByteTensor(mask)
        # mask = mask.bool()
        
        input = input.permute(0,2,1) # input: [N, T, inchannels]
        keys = self.linear_keys(input) # keys: (N, T, key_size)
        query = self.linear_query(input) # query: (N, T, key_size)
        values = self.linear_values(input) # values: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))

        weight_temp = F.softmax(temp / self.sqrt_key_size, dim=1) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp_vert = F.softmax(temp / self.sqrt_key_size, dim=1) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp_hori = F.softmax(temp / self.sqrt_key_size, dim=2) # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        # weight_temp = (weight_temp_hori + weight_temp_vert)/2
        value_attentioned = torch.bmm(weight_temp, values).permute(0,2,1) # shape: (N, T, value_size)
       
        return value_attentioned, weight_temp # value_attentioned: [N, in_channels, T], weight_temp: [N, T, T]
    

class TemporalConvolutionNetwork(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.1):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        tcl_params = {
            'kernel_size': kernel_size,
            'stride': 1,
            'dropout': dropout
        }

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            tcl_params['dilation'] = dilation
            tcl = TemporalCasualLayer(in_channels, out_channels, **tcl_params)
            layers.append(tcl)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class SparseMultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(SparseMultiHeadAttention, self).__init__()
        assert input_size % num_heads == 0, "Input size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_size = input_size // num_heads

        self.W_query = nn.Linear(input_size, input_size)
        self.W_key = nn.Linear(input_size, input_size)
        self.W_value = nn.Linear(input_size, input_size)

    def forward(self, x):
        batch_size, seq_length, input_size = x.size()
        
        queries = self.W_query(x)  # shape: (batch_size, seq_length, input_size)
        keys = self.W_key(x)  # shape: (batch_size, seq_length, input_size)
        values = self.W_value(x)  # shape: (batch_size, seq_length, input_size)

        # Reshape queries, keys, and values for multi-head attention
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        # Perform attention mechanism
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_size)  # shape: (batch_size, num_heads, seq_length, seq_length)
        attention_weights = torch.softmax(energy, dim=-1)  # shape: (batch_size, num_heads, seq_length, seq_length)
        context_vector = torch.matmul(attention_weights, values)  # shape: (batch_size, num_heads, seq_length, head_size)

        # Rearrange context_vector to match the original shape
        context_vector = context_vector.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, input_size)

        return context_vector   
    
class TCNN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout,l1_factor,l2_factor,num_heads):
        super(TCNN, self).__init__()
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.conv1 = TemporalConvolutionNetwork(input_channels, [32, 64, 128], kernel_size, dropout)
        self.attention = SparseMultiHeadAttention(num_channels[-1], num_heads)
        #self.attention = MultiHeadAttention(num_channels[-1], attention_hidden_size, num_heads)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.dropout = nn.Dropout(dropout)
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #x_mag = self.fourier_feature_engineering(x)
        y = self.tcn(x)
        y = self.dropout(y)

        # L1 Regularization
        l1_reg = torch.tensor(0.001)
        if self.l1_factor > 0:
            for param in self.parameters():
                l1_reg += torch.norm(param, p=1)

        # L2 Regularization
        l2_reg = torch.tensor(0.001)
        if self.l2_factor > 0:
            for param in self.parameters():
                l2_reg += torch.norm(param, p=2)

        #attended_features = self.attention(y[:, :, -1])
        y = y.transpose(1, 2)
        attended_features = self.attention(y)
        attended_features = attended_features.transpose(1, 2)  # Restore original shape
        out = self.linear(attended_features[:, :, -1])

        if self.l1_factor > 0:
            out += self.l1_factor * l1_reg

        if self.l2_factor > 0:
            out += 0.5 * self.l2_factor * l2_reg

        return out  
      

from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import copy


def train_model(model, X, y, epochs, optimizer, mse_loss, early_stopping, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_params = None
    min_val_loss = np.inf

    all_training_losses = []
    all_validation_losses = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        x_train, x_val = torch.tensor(X_train).float(), torch.tensor(X_val).float()
        y_train, y_val = torch.tensor(y_train).float(), torch.tensor(y_val).float()

        fold_training_losses = []
        fold_validation_losses = []
        fold_min_val_loss = np.inf

        for t in range(epochs):
            model.train()
            optimizer.zero_grad()

            prediction = model(x_train)
            loss = mse_loss(prediction, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_prediction = model(x_val)
                val_loss = mse_loss(val_prediction, y_val)

            fold_training_losses.append(loss.item())
            fold_validation_losses.append(val_loss.item())

            if val_loss.item() < fold_min_val_loss:
                best_fold_params = copy.deepcopy(model.state_dict())
                fold_min_val_loss = val_loss.item()
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {t} in fold {fold}")
                break

            if t % 10 == 0:
                print(f'Fold {fold}, Epoch {t}. Train Loss: {round(loss.item(), 4)}, Val Loss: {round(val_loss.item(), 4)}')

        all_training_losses.append(fold_training_losses)
        all_validation_losses.append(fold_validation_losses)

        if fold_min_val_loss < min_val_loss:
            best_params = best_fold_params
            min_val_loss = fold_min_val_loss

    return best_params, all_training_losses, all_validation_losses, min_val_loss




def ts_diff(ts):
    diff_ts = [0] * len(ts)
    for i in range(1, len(ts)):
        diff_ts[i] = ts[i] - ts[i - 1]
    return diff_ts


input2=[108]
output2=[108]
skill_so=np.zeros((4,10))


IG=np.zeros((6,4,264))
IG2=np.zeros((6,4,264))

Deconv=np.zeros((6,4,264))
Deconv2=np.zeros((6,4,264))

GB1=np.zeros((6,4,264))
GB2=np.zeros((6,4,264))

O1=np.zeros((6,4,264))
O2=np.zeros((6,4,264))

seed2=list(range(4))

for ii in range(4):
    seed =seed2[ii] #50
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    w=0
    i=0
    Z=[0,12,24,36,48]
    #D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
    D2=np.zeros((1980,6))
    for j in range(6):
        D2[:,j]=ts_diff(input_series[:,j])
    #T2[:,]=target_series[:,]
    T2=np.zeros((1980,))
    for j in range(1):
        T2[:,]=ts_diff(target_series[:,])
    # Use the mask to exclude NaN value
    
    #D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
    #T2=target_series#np.concatenate((no3_index[:,np.newaxis],tos_index[:,np.newaxis]),axis=1)
#   D=np.concatenate((X_no3[:-Z[i],np.newaxis],X_po4[:-Z[i],np.newaxis],pc1_tos[:-Z[i],:10],pc1_so[:-Z[i],:5],pc1_zos[:-Z[i],:5]),axis=1) 
    X_ss, Y_mm =  split_sequences(D2,T2[:,], input2[0],output2[0])
    print("X_ss",X_ss.shape)
    print("y_mm",Y_mm.shape)
    train_ratio=0.75
    train_len = round(len(X_ss[:-(input2[0]+output2[0]+264)]) * train_ratio)
    test_len=input2[0]+output2[0] #150/3
    X_train, Y_train= X_ss[:-(input2[0]+output2[0]+264)],\
                                   Y_mm[:-(input2[0]+output2[0]+264)],\
                                       #X_ss[-test_len:],\
                                       #Y_mm[-test_len:]

    print("X_train",X_train.shape)
    X_train, X_val, Y_train, Y_val = X_train[:train_len],\
                                     X_train[train_len:],\
                                     Y_train[:train_len],\
                                     Y_train[train_len:]
    x_train = torch.tensor(data = X_train).float()
    y_train = torch.tensor(data = Y_train).float()

    x_val = torch.tensor(data = X_val).float()
    y_val = torch.tensor(data = Y_val).float()

    #x_test = torch.tensor(data = X_test).float()
    #y_test = torch.tensor(data = Y_test).float()
    x_train = x_train.transpose(1, 2)
    x_val = x_val.transpose(1, 2)
    #x_test = x_test.transpose(1, 2)

    #y_train = y_train[:, :, 0]
    #y_val = y_val[:,:,0]
    print("x_train",x_train.shape)
    print("y_train",y_train.shape)
    train_len = x_train.size()[0]

    model_params = {
    'input_size': D2.shape[1], #60
    'output_size':  108,
    'num_channels': channel_sizes,
    'kernel_size':  kernel_size,
    'dropout':      dropout,
    'l1_factor':l1_factor,
    'l2_factor':l2_factor,
    'num_heads':3
    }
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNN(**model_params)

    best_params = None
    min_val_loss = sys.maxsize

    training_loss = []
    validation_loss = []

    #model = model.to(device)
    n_splits = 3  # Number of fold

    # Data preparation
    X_ss, Y_mm = split_sequences(D2,target_series[:,], input2[0], output2[0])
    print("X_ss", X_ss.shape)
    print("y_mm", Y_mm.shape)

    # Results storage
    all_train_loss = []
    all_val_loss = []
    #early_stopping = EarlyStopping(patience=10, verbose=True)
    
    # Train the model on the current fold
    kernel_sizes = [3,4,5,6]
    channel_sizes_options = [[6,6,6],[9,9,9],[12,12,12],[18,18,18]]
    dropout_list=[0,0.1,0.2,0.3]
    epochs = 200

    best_config = None
    best_val_loss = np.inf
    best_model_params = None

    train_losses=[]
    val_losses=[]
    best_model_path = 'best_model.pth'
    
    for kernel_size in kernel_sizes:
      for channel_sizes in channel_sizes_options:
         for dropout in dropout_list:

            print(f"Testing Kernel Size: {kernel_size}, Channel Sizes: {channel_sizes}")

            model_params = {
            'input_size': 6,  # Adjust this as needed
            'output_size': 108,
            'num_channels': channel_sizes,
            'kernel_size': kernel_size,
            'dropout': dropout,
            'l1_factor': l1_factor,
            'l2_factor': l2_factor,
            'num_heads': 3
            }

            model = TCNN(**model_params)
            optimizer = torch.optim.AdamW(params=model.parameters(), weight_decay=0.000001, lr=0.007) 
            mse_loss = nn.MSELoss()

            early_stopping = EarlyStopping(patience=10, verbose=True)

            # Train the model and get the validation loss
            best_params, train_loss, val_loss, val_loss_min = train_model(
            model, x_train, y_train, epochs, optimizer, mse_loss,early_stopping
            )

            if val_loss_min < best_val_loss:
               best_val_loss = val_loss_min
               best_config = (kernel_size, channel_sizes, dropout)
               best_model_params = best_params

            train_losses.append(train_loss)
            val_losses.append(val_loss)




    # Optionally load the best model parameters
    model.load_state_dict(best_params)
    
    avg_train_loss = np.mean([np.min(losses) for losses in train_loss])
    avg_val_loss = np.mean([np.min(losses) for losses in val_loss])
    plt.figure()
    plt.title('Training Progress')
    plt.yscale("log")
    plt.plot(avg_train_loss, label = 'train')
    plt.plot(avg_val_loss, label = 'validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    plt.savefig("loss_"+str(w)+".png")


    def ts_int(ts_diff, ts_base, start=0):
    #    """
    #    Integrate a differenced time series using cumulative sum.

    #    Parameters:
    #    - ts_diff (numpy array): The differenced time series.
    #    - ts_base (numpy array): The base time series.
    #    - start (float): The initial value for integration.

    #    Returns:
    #    - ts_integrated (numpy array): The integrated time series.
    #    """
        ts_diff = np.asarray(ts_diff)
        ts_base = np.asarray(ts_base)

        ts_integrated = np.empty_like(ts_diff)
        ts_integrated[0] = start + ts_diff[0]

        # Use cumulative sum for integration
        ts_integrated[1:] = np.cumsum(ts_diff[1:]) + ts_base[:-1]
        return ts_integrated.tolist()
    
    #def ts_int(ts_diff, ts_base, start):
#        ts_int = [start]
#        for i in range(1, len(ts_diff)):
#            ts_int.append(ts_int[i-1] + ts_diff[i-1] + ts_base[i-1])
#        return np.array(ts_int)

    for N in range(264):
        print('N=',N)
        if N==0:
           test_len=input2[0]+output2[0]
           X_test, Y_test= X_ss[-test_len:],Y_mm[-test_len:]
           x_test = torch.tensor(data = X_test).float()
           y_test = torch.tensor(data = Y_test).float()
           x_test = x_test.transpose(1, 2)
           #y_test=y_test[:,:,ii]

           best_model = TCNN(**model_params)
           best_model.eval()
           best_model.load_state_dict(best_params)

           tcn_prediction = best_model(x_test)
           #print('tcn_prediction',tcn_prediction[-1,:].detach().numpy().shape)
           A=0
           years=np.arange(1996-int(A/12),2015,1/12)
           test_len=input2[0]+output2[0]  #150/3

           Z=ts_int(
            tcn_prediction[-1,:].tolist(),
            target_series[-108:,],
            start = target_series[-108-1,]
            )
           #Q=["NO3 anomaly","PO4 anomaly","first pc of SST"]
           #ci = 0.1 * np.std(Z[input2[0]:]) / np.mean(Z[input2[0]:])
           #95% confidence interval
           #plt.figure()
           #plt.fill_between(years[-108:], (Z[input2[0]:]-ci), (Z[input2[0]:]+ci), color='green', alpha=0.5)
           #plt.plot(years[-108:],Z[input2[0]:],label = 'tcn',color='k',linewidth=2.5)
           #plt.plot(years[-108:],T2[-108:], label = 'real',color='r',linewidth=2.5)
           #plt.ylabel(Q[ii],fontsize=13)
           #plt.xlabel("Years",fontsize=13)
           #plt.xticks(fontsize=13)
           #plt.yticks(fontsize=13)
           #plt.legend()
           #plt.show()
           #plt.savefig('forecast_TCNN_GFDL_0.png')

           
           prediction1[ii,N]=Z[-1]
           prediction2[ii,N]=Z[-12]
           prediction3[ii,N]=Z[-24]
           prediction4[ii,N]=Z[-36]
           prediction5[ii,N]=Z[-48]
           prediction6[ii,N]=Z[-60]
           prediction7[ii,N]=Z[-72]
           prediction8[ii,N]=Z[-84]
           prediction9[ii,N]=Z[-96] 

           ig = IntegratedGradients(model)
           constant=0
           num_steps=150
           baseline_sequence = torch.full(x_test.size(), constant, device=x_test.device)
           attributions = ig.attribute(x_test, baselines=baseline_sequence, n_steps=num_steps,target=torch.tensor([0])) 
           
           attributions=attributions.reshape(216,108,6)
           IG[0,ii,N]=attributions[-1,-1,0]
           IG[1,ii,N]=attributions[-1,-1,1]
           IG[2,ii,N]=attributions[-1,-1,2]
           IG[3,ii,N]=attributions[-1,-1,3]
           IG[4,ii,N]=attributions[-1,-1,4]
           IG[5,ii,N]=attributions[-1,-1,5]

           IG2[0,ii,N]=attributions[-84,-84,0]
           IG2[1,ii,N]=attributions[-84,-84,1]
           IG2[2,ii,N]=attributions[-84,-84,2]
           IG2[3,ii,N]=attributions[-84,-84,3]
           IG2[4,ii,N]=attributions[-84,-84,4]
           IG2[5,ii,N]=attributions[-84,-84,5]

           attributions2=[]
           # Iterate through the train_dataloader to compute attributions for each batch
           
           # Compute GradientShap attributions for the batch
           constant=0
           baseline = torch.full(x_test.size(), constant, device=x_test.device)
           n_samples = 96
           attributions = gs.attribute(x_test, baseline, n_samples=n_samples, target=torch.tensor([0])) 

           attributions=attributions.reshape(216,108,6)
           GB1[0,ii,N]=attributions[-1,-1,0]
           GB1[1,ii,N]=attributions[-1,-1,1]
           GB1[2,ii,N]=attributions[-1,-1,2]
           GB1[3,ii,N]=attributions[-1,-1,3]
           GB1[4,ii,N]=attributions[-1,-1,4]
           GB1[5,ii,N]=attributions[-1,-1,5]

           GB2[0,ii,N]=attributions[-84,-84,0]
           GB2[1,ii,N]=attributions[-84,-84,1]
           GB2[2,ii,N]=attributions[-84,-84,2]
           GB2[3,ii,N]=attributions[-84,-84,3]
           GB2[4,ii,N]=attributions[-84,-84,4]
           GB2[5,ii,N]=attributions[-84,-84,5]
           
           

        if N>0:
           X_test, Y_test= X_ss[-test_len-N:-N],Y_mm[-test_len-N:-N]
           x_test = torch.tensor(data = X_test).float()
           y_test = torch.tensor(data = Y_test).float()
           x_test = x_test.transpose(1, 2)
           

           best_model = TCNN(**model_params)
           best_model.eval()
           best_model.load_state_dict(best_params)

           tcn_prediction = best_model(x_test)

           A=24
           test_len=input2[0]+output2[0]  #150/3
           
           Z=ts_int(
            tcn_prediction[-1,:].tolist(),
            target_series[-108-N:-N,],
            start = target_series[-108-N-1,]
            )
           Q=["NO3 anomaly","PO4 anomaly","first pc of SST"]
           ci = 0.1 * np.std(Z[input2[0]:]) / np.mean(Z[input2[0]:])
           #95% confidence interval
           #plt.figure()
           #plt.fill_between(years[-96-N:-N],(Z[20:]-ci), (Z[20:]+ci), color='green', alpha=0.5)
           #plt.plot(years[-108-N:-N],Z[input2[0]:],label = 'tcn',color='k',linewidth=2.5)
           #plt.plot(years[-108-N:-N],T2[-108-N:-N,], label = 'real',color='r',linewidth=2.5)
           #plt.ylabel(Q[ii],fontsize=13)
           #plt.xlabel("Years",fontsize=13)
           #plt.xticks(fontsize=13)
           #plt.yticks(fontsize=13)
           #plt.legend()
           #plt.show( 
              
           prediction1[ii,N]=Z[-1]
           prediction2[ii,N]=Z[-12]
           prediction3[ii,N]=Z[-24]
           prediction4[ii,N]=Z[-36]
           prediction5[ii,N]=Z[-48]
           prediction6[ii,N]=Z[-60]
           prediction7[ii,N]=Z[-72]
           prediction8[ii,N]=Z[-84]
           prediction9[ii,N]=Z[-96] 
           
           ig = IntegratedGradients(model)
           constant=0
           num_steps=150
           baseline_sequence = torch.full(x_test.size(), constant, device=x_test.device)
           attributions = ig.attribute(x_test, baselines=baseline_sequence, n_steps=num_steps,target=torch.tensor([0])) 

        
           attributions=attributions.reshape(216,108,6)
           IG[0,ii,N]=attributions[-1,-1,0]
           IG[1,ii,N]=attributions[-1,-1,1]
           IG[2,ii,N]=attributions[-1,-1,2]
           IG[3,ii,N]=attributions[-1,-1,3]
           IG[4,ii,N]=attributions[-1,-1,4]
           IG[5,ii,N]=attributions[-1,-1,5]

           IG2[0,ii,N]=attributions[-84,-84,0]
           IG2[1,ii,N]=attributions[-84,-84,1]
           IG2[2,ii,N]=attributions[-84,-84,2]
           IG2[3,ii,N]=attributions[-84,-84,3]
           IG2[4,ii,N]=attributions[-84,-84,4]
           IG2[5,ii,N]=attributions[-84,-84,5]
           
           attributions2=[]
           # Iterate through the train_dataloader to compute attributions for each batch
           
           # Compute GradientShap attributions for the batch
           constant=0
           baseline = torch.full(x_test.size(), constant, device=x_test.device)
           n_samples = 96
           attributions = gs.attribute(x_test, baseline, n_samples=n_samples, target=torch.tensor([0])) 

           attributions=attributions.reshape(216,108,6)
           GB1[0,ii,N]=attributions[-1,-1,0]
           GB1[1,ii,N]=attributions[-1,-1,1]
           GB1[2,ii,N]=attributions[-1,-1,2]
           GB1[3,ii,N]=attributions[-1,-1,3]
           GB1[4,ii,N]=attributions[-1,-1,4]
           GB1[5,ii,N]=attributions[-1,-1,5]

           GB2[0,ii,N]=attributions[-84,-84,0]
           GB2[1,ii,N]=attributions[-84,-84,1]
           GB2[2,ii,N]=attributions[-84,-84,2]
           GB2[3,ii,N]=attributions[-84,-84,3]
           GB2[4,ii,N]=attributions[-84,-84,4]
           GB2[5,ii,N]=attributions[-84,-84,5]

           
barWidth = 0.5
vector = np.nanmean(np.nanmean(IG[:, :, :], axis=1), axis=1).reshape(6,)
IG1 = vector.tolist()
br1 = np.arange(len(IG1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]


vector = np.nanmean(np.nanmean(IG2[:, :, :], axis=1), axis=1).reshape(6,)
IG3 = vector.tolist()
br1 = np.arange(len(IG3))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]




plt.figure()
plt.bar(feature,np.abs(IG1), color ='b', width = barWidth,
        edgecolor ='grey', label ='IG')
#shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
plt.xlabel("Input Feature",fontsize=16, fontweight='bold')
plt.ylabel("IG",fontsize=16, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.figure()
plt.bar(feature,np.abs(IG3), color ='b', width = barWidth,
        edgecolor ='grey', label ='IG')
#shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
plt.xlabel("Input Feature",fontsize=16, fontweight='bold')
plt.ylabel("IG",fontsize=16, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

    

barWidth = 0.5
vector = np.nanmean(np.nanmean(GB1[:, :, :], axis=1), axis=1).reshape(6,)
GB1 = vector.tolist()
br1 = np.arange(len(GB1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]


vector = np.nanmean(np.nanmean(GB2[:, :, :], axis=1), axis=1).reshape(6,)
GB3 = vector.tolist()
br1 = np.arange(len(GB3))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]




plt.figure()
plt.bar(feature,np.abs(GB1), color ='b', width = barWidth,
        edgecolor ='grey', label ='IG')
#shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
plt.xlabel("Input Feature",fontsize=16, fontweight='bold')
plt.ylabel("IG",fontsize=16, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.figure()
plt.bar(feature,np.abs(GB3), color ='b', width = barWidth,
        edgecolor ='grey', label ='IG')
#shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
plt.xlabel("Input Feature",fontsize=16, fontweight='bold')
plt.ylabel("IG",fontsize=16, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()
 
    

    

