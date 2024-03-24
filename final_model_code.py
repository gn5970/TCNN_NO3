
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from matplotlib import ticker
from cmcrameri import cm
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import intake

import numpy as np
import cartopy
import numpy.polynomial.polynomial as poly
from netCDF4 import Dataset
import shap
import torch
from torch.autograd import grad
from eofs.xarray import Eof
import torch.nn as nn
from torch.nn.utils import weight_norm
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
#import shap
import statsmodels.api as sm
from torch.autograd import Variable
#torch.cuda.is_available()

seed =4 #2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def remove_time_mean(x):
    return x - x.mean(dim='time', skipna=True)

def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)

def calculate_integrated_gradients(model, input_sequence, num_steps=60):
      # Generate a random baseline sequence
      baseline_sequence = torch.randn_like(input_sequence)

      # Compute the difference sequence
      diff_sequence = input_sequence - baseline_sequence

      # Calculate the step size for interpolation
      alpha = torch.linspace(0, 1, num_steps)

      # Initialize the integrated gradients
      integrated_gradients = torch.zeros_like(input_sequence)

      # Compute the integrated gradients
      for step in alpha:
        # Interpolate between the baseline and input sequences
        interpolated_sequence = baseline_sequence + step * diff_sequence

        # Enable gradients calculation
        interpolated_sequence.requires_grad_(True)

        # Forward pass through the model
        output = model(interpolated_sequence)

        # Backward pass to accumulate gradients
        grads = grad(output.sum(), interpolated_sequence)[0]

        # Scale the gradients and accumulate
        integrated_gradients += grads * diff_sequence

      # Average the accumulated gradients
      integrated_gradients /= num_steps
      return integrated_gradients

def remove_time_mean(x):
    return x - x.mean(dim='time')


def wgt_areaave(indat, latS, latN, lonW, lonE):
  lat=indat.lat
  lon=indat.lon

  if ( ((lonW < 0) or (lonE < 0 )) and (lon.values.min() > -1) ):
     anm=indat.assign_coords(lon=( (lon + 180) % 360 - 180) )
     lon=( (lon + 180) % 360 - 180)
  else:
     anm=indat

  iplat = lat.where( (lat >= latS ) & (lat <= latN), drop=True)
  iplon = lon.where( (lon >= lonW ) & (lon <= lonE), drop=True)

#  print(iplon)
  wgt = np.cos(np.deg2rad(lat))
  odat=anm.sel(lat=iplat,lon=iplon).weighted(wgt).mean(("lon", "lat"), skipna=True)
  return(odat)


# TCNN forecast




def sliding_window(ts, features, target_len ):
    X = []
    Y = []

    for i in range(features + target_len, len(ts) + 1):
        X.append(ts[i - (features + target_len):i - target_len])
        Y.append(ts[i - target_len:i])

    return np.array(X),np.array(Y)

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


def ts_diff(ts):
    diff_ts = [0] * len(ts)
    for i in range(1, len(ts)):
        diff_ts[i] = ts[i] - ts[i - 1]
    return diff_ts

class Crop(nn.Module):

    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[:, :, :-self.crop_size].contiguous()

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, input_size),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

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

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout = 0.1):
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
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.conv3 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop3 = Crop(padding)
        self.relu3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 =  weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop4 = Crop(padding)
        self.relu4 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(dropout)


        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1,
                                 self.conv2, self.crop2, self.relu2, self.dropout2)
        self.residual = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
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


class TemporalConvolutionNetwork(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size, dropout ):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        tcl_param = {
            'kernel_size': kernel_size,
            'stride':      1,
            'dropout':     dropout
        }
        for i in range(num_levels):
            dilation = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            tcl_param['dilation'] = dilation
            tcl = TemporalCasualLayer(in_ch, out_ch, **tcl_param)
            layers.append(tcl)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout,l1_factor,l2_factor):
        super(TCNN, self).__init__()
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.attention = MultiHeadAttention(num_channels[-1], attention_hidden_size, num_heads)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.dropout = nn.Dropout(dropout)
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def fourier_feature_engineering(self, x):
        # Perform Fourier Transform on the input data
        # x: input data tensor of shape (batch_size, seq_len, input_size)

        # Apply FFT along the sequence dimension
        x_fft = torch.fft.fft(x, dim=2)

        # Compute magnitude spectrum
        x_mag = torch.abs(x_fft)

        return x_mag

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
        out = self.linear(y[:, :, -1])

        if self.l1_factor > 0:
            out += self.l1_factor * l1_reg

        if self.l2_factor > 0:
            out += 0.5 * self.l2_factor * l2_reg

        return out


def training_loop(n_epochs, model, optimiser, loss_fn, X_train, y_train,X_val,y_val):
            for epoch in range(n_epochs):
                model.train()
                outputs = model.forward(X_train) # forward pass
                optimiser.zero_grad() # calculate the gradient, manually setting to 0
                # obtain the loss function
                loss = loss_fn(outputs, y_train)
                loss.backward() # calculates the loss of the loss function
                optimiser.step() # improve from loss, i.e backprop
                losses.append(loss.item())

                train_predict = model(X_train[-1].unsqueeze(0)) # get the last sample
                train_predict = train_predict.detach().numpy()
                train_predict = mm.inverse_transform(train_predict)
                train_predict = train_predict[0].tolist()

                train_target = y_train_tensors[-1].detach().numpy() # last sample again
                train_target = mm.inverse_transform(train_target.reshape(1, -1))
                train_target = train_target[0].tolist()

                true2=np.array(train_target)
                preds2=np.array(train_predict)

                acc=np.corrcoef(true2,preds2)
                running_accuracy.append(acc[1][0])
                with torch.no_grad():
                     output=model.forward(X_val)
                     val_loss = loss_fn(output, y_val)
                     val_losses.append(val_loss.item())
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("We are at epoch:", epoch)
                    break
                val_predict = model(X_val[-1].unsqueeze(0)) # get the last sample
                val_predict = val_predict.detach().numpy()
                val_predict = mm.inverse_transform(val_predict)
                val_predict = val_predict[0].tolist()

                val_target = y_val_tensors[-1].detach().numpy() # last sample again
                val_target = mm.inverse_transform(val_target.reshape(1, -1))
                val_target = val_target[0].tolist()

                true2=np.array(val_target)
                preds2=np.array(val_predict)
                
                acc2=np.corrcoef(true2,preds2)
                running_accuracy_val.append(acc2[1][0])

                if epoch % 200==0:
                   print("Epoch: %d, train loss: %1.5f, val loss: %1.5f, train_acc: %1.5f" % (epoch,loss.item(),val_loss.item(),acc[1][0]))

def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

def calculate_integrated_gradients_uniformbaseline(model, input_sequence, num_steps):
      # Generate a random baseline sequence
      mean = x_test.mean()  # Mean of the distribution
      stddev = x_test.std()  # Standard deviation of the distribution

      # Define the size of the tensor (similar to x_test)
      tensor_size = input_sequence.size()

      # Generate random samples from a Gaussian distribution
      baseline_sequence = torch.randn(tensor_size) * stddev + mean

                
      # Create the uniform baseline by sampling from the uniform distribution
      #baseline_sequence=torch.empty_like(x_test).uniform_(lower_bound, upper_bound)
      #baseline_sequence = torch.randn_like(input_sequence)

      # Compute the difference sequence
      diff_sequence = input_sequence - baseline_sequence

      # Calculate the step size for interpolation
      alpha = torch.linspace(0, 1, num_steps)

      # Initialize the integrated gradients
      integrated_gradients = torch.zeros_like(input_sequence)

      # Compute the integrated gradients
      for step in alpha:
        # Interpolate between the baseline and input sequences
        interpolated_sequence = baseline_sequence + step * diff_sequence

        # Enable gradients calculation
        interpolated_sequence.requires_grad_(True)

        # Forward pass through the model
        output = model(interpolated_sequence)

        # Backward pass to accumulate gradients
        grads = grad(output.sum(), interpolated_sequence)[0]

        # Scale the gradients and accumulate
        integrated_gradients += grads * diff_sequence

      # Average the accumulated gradients
      integrated_gradients /= num_steps
      return integrated_gradients

def calculate_integrated_gradients_constbaseline(model, input_sequence, num_steps):
      # Generate a constant baseline sequence
      constant=0
      baseline_sequence = torch.full(input_sequence.size(),constant)
      # Compute the difference sequence
      input_sequence=input_sequence.unsqueeze(0).expand(2,-1, -1, -1)
      baseline_sequence=baseline_sequence.unsqueeze(0).expand(2,-1, -1, -1)
      diff_sequence = input_sequence - baseline_sequence

      # Calculate the step size for interpolation
      alpha = torch.linspace(0, 1, num_steps)

      # Initialize the integrated gradients
      integrated_gradients = torch.zeros_like(input_sequence)
      #integrated_gradients=integrated_gradients.unsqueeze(0).expand(2,-1, -1, -1)

      # Compute the integrated gradients

      for target_idx in range(2):
        for step in alpha:
          # Interpolate between the baseline and input sequences
          interpolated_sequence = baseline_sequence[target_idx,:, :,:] + step * diff_sequence[target_idx,:, :,:]

          # Enable gradients calculation
          interpolated_sequence.requires_grad_(True)

          # Forward pass through the model
          output = model(interpolated_sequence)

          # Backward pass to accumulate gradients
          grads = grad(output[target_idx,:].sum(), interpolated_sequence,allow_unused=True)[0]

          # Scale the gradients and accumulate
          integrated_gradients += grads * diff_sequence

        # Average the accumulated gradients
      integrated_gradients /= num_steps
      return integrated_gradients

def calculate_integrated_gradients_constbaseline(model, input_sequence, num_steps):
      # Generate a constant baseline sequence
      constant=1
      baseline_sequence = torch.full(input_sequence.size(),constant)

      # Compute the difference sequence
      diff_sequence = input_sequence - baseline_sequence

      # Calculate the step size for interpolation
      alpha = torch.linspace(0, 1, num_steps)

      # Initialize the integrated gradients
      integrated_gradients = torch.zeros_like(input_sequence)

      # Compute the integrated gradients
      for step in alpha:
        # Interpolate between the baseline and input sequences
        interpolated_sequence = baseline_sequence + step * diff_sequence

        # Enable gradients calculation
        interpolated_sequence.requires_grad_(True)

        # Forward pass through the model
        output = model(interpolated_sequence)

        # Backward pass to accumulate gradients
        grads = grad(output.sum(), interpolated_sequence)[0]

        # Scale the gradients and accumulate
        integrated_gradients += grads * diff_sequence

      # Average the accumulated gradients
      integrated_gradients /= num_steps
      return integrated_gradients



def lrp(model, x, baseline,epsilon):

    # Forward pass to obtain output
    x.requires_grad = True
    output = model(x)

    # Initialize relevance scores
    R = output.clone()

    # Backward pass with LRP
    output.backward(gradient=R, retain_graph=True)

    # Compute feature importances
    relevance_scores = x.grad * (x - baseline)

    # Normalize the relevance scores to consider both positive and negative contributions
    # You may need to adjust this part based on the specific LRP rule you want to apply
    relevance_scores /= (x.grad.sum(dim=1, keepdim=True) + epsilon)

    # Clear gradients for future computations
    model.zero_grad()
    x.grad.data.zero_()

    return relevance_scores

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



# time series input
features =20
# training epochs
epochs =300 #1000
# synthetic time series dataset
ts_len = 1980
# test dataset size
test_len =106
# temporal casual layer channels
channel_sizes = [4] * 3
# convolution kernel size
kernel_size =3 #5
dropout = 0.1

train_ratio=0.7

l1_factor=3*10^(-10)
l2_factor=3*10^(-10)

C=[0,1,2]


input2=[96]
output2=[96]
D=np.zeros((190,21))
skill_so=np.zeros((9))

N=264

prediction1=np.zeros((264))
prediction2=np.zeros((264))
prediction3=np.zeros((264))
prediction4=np.zeros((264))
prediction5=np.zeros((264))
prediction6=np.zeros((264))
prediction7=np.zeros((264))
prediction8=np.zeros((264))
prediction9=np.zeros((264))
prediction10=np.zeros((264))

shap3=np.zeros((10, 264, 18))
shap4=np.zeros((10, 264, 18))
shap5=np.zeros((10, 264, 18))

integrated_gradients3=np.zeros((264, 18))
integrated_gradients4=np.zeros((264, 18))
integrated_gradients5=np.zeros((264, 18))

relevance_scores3=np.zeros((264,18))
relevance_scores4=np.zeros((264,18))
relevance_scores5=np.zeros((264,18))


A_lrp0_24=np.zeros((264))
A_lrp1_24=np.zeros((264))
A_lrp2_24=np.zeros((264))
A_lrp3_24=np.zeros((264))
A_lrp4_24=np.zeros((264))
A_lrp5_24=np.zeros((264))
A_lrp6_24=np.zeros((264))
A_lrp7_24=np.zeros((264))
A_lrp8_24=np.zeros((264))
A_lrp9_24=np.zeros((264))
A_lrp10_24=np.zeros((264))

A_lrp0_96=np.zeros((264))
A_lrp1_96=np.zeros((264))
A_lrp2_96=np.zeros((264))
A_lrp3_96=np.zeros((264))
A_lrp4_96=np.zeros((264))
A_lrp5_96=np.zeros((264))
A_lrp6_96=np.zeros((264))
A_lrp7_96=np.zeros((264))
A_lrp8_96=np.zeros((264))
A_lrp9_96=np.zeros((264))
A_lrp10_96=np.zeros((264))



Z0_shap_24=np.zeros((264))
Z1_shap_24=np.zeros((264))
Z2_shap_24=np.zeros((264))
Z3_shap_24=np.zeros((264))
Z4_shap_24=np.zeros((264))
Z5_shap_24=np.zeros((264))
Z6_shap_24=np.zeros((264))
Z7_shap_24=np.zeros((264))
Z8_shap_24=np.zeros((264))
Z9_shap_24=np.zeros((264))
Z10_shap_24=np.zeros((264))

Z0_shap_96=np.zeros((264))
Z1_shap_96=np.zeros((264))
Z2_shap_96=np.zeros((264))
Z3_shap_96=np.zeros((264))
Z4_shap_96=np.zeros((264))
Z5_shap_96=np.zeros((264))
Z6_shap_96=np.zeros((264))
Z7_shap_96=np.zeros((264))
Z8_shap_96=np.zeros((264))
Z9_shap_96=np.zeros((264))
Z10_shap_96=np.zeros((264))

Z0_ig_24=np.zeros((264))
Z1_ig_24=np.zeros((264))
Z2_ig_24=np.zeros((264))
Z3_ig_24=np.zeros((264))
Z4_ig_24=np.zeros((264))
Z5_ig_24=np.zeros((264))
Z6_ig_24=np.zeros((264))
Z7_ig_24=np.zeros((264))
Z8_ig_24=np.zeros((264))
Z9_ig_24=np.zeros((264))
Z10_ig_24=np.zeros((264))
Z11_ig_24=np.zeros((264))
Z12_ig_24=np.zeros((264))
Z13_ig_24=np.zeros((264))
Z14_ig_24=np.zeros((264))
Z12_ig_24=np.zeros((264))
Z13_ig_24=np.zeros((264))
Z14_ig_24=np.zeros((264))
Z15_ig_24=np.zeros((264))
Z16_ig_24=np.zeros((264))
Z17_ig_24=np.zeros((264))
Z18_ig_24=np.zeros((264))
Z19_ig_24=np.zeros((264))
Z20_ig_24=np.zeros((264))
Z21_ig_24=np.zeros((264))

Z0_ig_96=np.zeros((264))
Z1_ig_96=np.zeros((264))
Z2_ig_96=np.zeros((264))
Z3_ig_96=np.zeros((264))
Z4_ig_96=np.zeros((264))
Z5_ig_96=np.zeros((264))
Z6_ig_96=np.zeros((264))
Z7_ig_96=np.zeros((264))
Z8_ig_96=np.zeros((264))
Z9_ig_96=np.zeros((264))
Z10_ig_96=np.zeros((264))
Z11_ig_96=np.zeros((264))


A_mean_baseline1_24=np.zeros((10,264,8))
A_mean_baseline1_96=np.zeros((10,264,8))
A_mean_baseline_ig1_24=np.zeros((10,264,8))
A_mean_baseline_ig1_96=np.zeros((10,264,8))
A_mean_lrp_baseline_24=np.zeros((10,264,8))
A_mean_lrp_baseline_96=np.zeros((10,264,8))

tos_index=np.array(tos_index)
so_index=np.array(so_index)
zos_index=np.array(zos_index)
mld_index=np.array(mld_index)
b_index=np.array(b_index)
npp_index=np.array(npp_index)


# persistence
lags = range(97)
persistence=np.zeros((1,97))
import statsmodels.api as sm
acorr = sm.tsa.acf(no3_index, nlags = len(lags)-1)
#auto2[im,:]=acorr
#acorr = sm.tsa.acf(savitzky_golay(A2[:,im],12,3), nlags = len(lags)-1)
persistence[0,:]=acorr


plt.figure()
plt.plot(persistence[0,::12])
plt.ylabel('NO3 persistence',fontsize=13)
plt.xlabel('Time (years)',fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.savefig('skill_persistence_NO3.png')



years=np.arange(1850,2015,1/12)
plt.figure()
plt.plot(years,tos_index)
plt.xticks(fontsize=15,weight='bold')
plt.yticks(fontsize=15,weight='bold')
plt.savefig('tos_index.pdf')

plt.figure()
plt.plot(years,zos_index)
plt.xticks(fontsize=15,weight='bold')
plt.yticks(fontsize=15,weight='bold')
plt.savefig('zos_index.pdf')

plt.figure()
plt.plot(years,so_index)
plt.xticks(fontsize=15,weight='bold')
plt.yticks(fontsize=15,weight='bold')
plt.savefig('so_index.pdf')

plt.figure()
plt.plot(years,ekman_pumping_index)
plt.xticks(fontsize=15,weight='bold')
plt.yticks(fontsize=15,weight='bold')
plt.savefig('ekman_pumping_index.pdf')

plt.figure()
plt.plot(years,mld_index)
plt.xticks(fontsize=15,weight='bold')
plt.yticks(fontsize=15,weight='bold')
plt.savefig('mld_index.pdf')

plt.figure()
plt.plot(years,b_index)
plt.xticks(fontsize=15,weight='bold')
plt.yticks(fontsize=15,weight='bold')
plt.savefig('b_index.pdf')

plt.figure()
plt.plot(years,pd_index)
plt.xticks(fontsize=15,weight='bold')
plt.yticks(fontsize=15,weight='bold')
plt.savefig('pd_index.pdf')

plt.figure()
plt.plot(years,fe_index)
plt.xticks(fontsize=15,weight='bold')
plt.yticks(fontsize=15,weight='bold')
plt.savefig('fe_index.pdf')

plt.figure()
plt.plot(years,ekman_pumping_index)
plt.xticks(fontsize=15,weight='bold')
plt.yticks(fontsize=15,weight='bold')
plt.savefig('ekman_pumping_index.pdf')

plt.figure()
plt.plot(years,no3_index)
plt.xticks(fontsize=15,weight='bold')
plt.yticks(fontsize=15,weight='bold')
plt.savefig('no3_index.pdf')


for ii in range(1):
    w=0
    i=0
    Z=[0,12,24,36,48]
    #D=np.concatenate((pc1_chl[:,:5],pc1_tos[:,:10],pc1_so[:,:5],pc1_zos[:,:5]),axis=1)
    D=np.concatenate((tos_index,so_index,zos_index,ekman_pumping_index,mld_index,b_index,pd_index,fe_index),axis=0)
    D=D.reshape(1980,8) 
    # Use the mask to exclude NaN values
    print("D",D.shape)
    D2=np.zeros((1980,8)) 
     
    for i in range(D2.shape[1]):
        D2[:,i]=ts_diff(D[:,i])
    T2=no3_index
    X_ss, Y_mm =  split_sequences(D2,T2, input2[0],output2[0])
    print("X_ss",X_ss.shape)
    print("y_mm",Y_mm.shape)
    train_ratio=0.75
    train_len = round(len(X_ss[:-(input2[0]+output2[0]+264)]) * train_ratio)
    test_len=input2[0]+output2[0] #150/3
    X_train, Y_train= X_ss[:-(input2[0]+output2[0]+264)],\
                                   Y_mm[:-(input2[0]+output2[0]+264)]
                                       

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
    'output_size':  1,
    'num_channels': channel_sizes,
    'kernel_size':  kernel_size,
    'dropout':      dropout,
    'l1_factor':l1_factor,
    'l2_factor':l2_factor
    }
    early_stopping = EarlyStopping(patience=30, verbose=True)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNN(**model_params)

    optimizer = torch.optim.Adam(params = model.parameters(),weight_decay=0.0001, lr =0.001 )#0.00005
    mse_loss = torch.nn.MSELoss()

    best_params = None
    min_val_loss = sys.maxsize

    training_loss = []
    validation_loss = []

    #model = model.to(device)
    model.train()

    for t in range(epochs):

        prediction = model(x_train)
        loss = mse_loss(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_prediction = model(x_val)
        val_loss = mse_loss(val_prediction, y_val)

        training_loss.append(loss.item())
        validation_loss.append(val_loss.item())

        early_stopping(val_loss, model)
        if val_loss.item() < min_val_loss:
           best_params = copy.deepcopy(model.state_dict())
           min_val_loss = val_loss.item()

        if early_stopping.early_stop:
          print("Early stopping at epoch", t)
          break
          if t % 100 == 0:
             diff = (y_train - prediction).view(-1).abs_().tolist()
             print(f'epoch {t}. train: {round(loss.item(), 4)}, '
             f'val: {round(val_loss.item(), 4)}') 
    
    plt.figure()
    plt.title('Training Progress')
    plt.yscale("log")
    plt.plot(training_loss, label = 'train')
    plt.plot(validation_loss, label = 'validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    plt.savefig("loss_"+str(w)+".png")


    def ts_int(ts_diff, ts_base, start=0):
        ts_diff = np.asarray(ts_diff)  # Convert to NumPy array for vectorized operations
        ts_base = np.asarray(ts_base)
    
        ts = np.empty_like(ts_diff)  # Create an empty array to store the integrated time series
        ts[0] = start + ts_diff[0]  # Set the initial value
    
        # Perform vectorized addition to calculate the integrated series
        ts[1:] = ts_diff[1:] + ts_base[:-1]
    
        return ts.tolist()

    for N in range(264):
        if N==0:
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
           

           Z=ts_int(tcn_prediction.view(-1).tolist(),
           T2[-test_len:,],
           start = T2[-test_len - 1, ]
           )

           
           prediction1[0]=Z[-1]
           prediction2[0]=Z[-12]
           prediction3[0]=Z[-24]
           prediction4[0]=Z[-36]
           prediction5[0]=Z[-48]
           prediction6[0]=Z[-60]
           prediction7[0]=Z[-72]
           prediction8[0]=Z[-84]
             
           
           P=np.arange(40,140,10)
           P1=np.arange(0.9,1.9,1/10)/1150
           for i in range(len(P)):
                  
                  integrated_gradients=calculate_integrated_gradients_uniformbaseline(best_model, x_test, num_steps=P[i])
                  integrated_gradients=np.array(integrated_gradients.detach().numpy())
                
                  
                  integrated_gradients=integrated_gradients.transpose(2,0,1)
                  integrated_gradients=np.mean(integrated_gradients,axis=0)

                  Z0_ig_24[0]=integrated_gradients[-72,0]
                  Z1_ig_24[0]=integrated_gradients[-72,1]
                  Z2_ig_24[0]=integrated_gradients[-72,2]
                  Z3_ig_24[0]=integrated_gradients[-72,3]
                  Z4_ig_24[0]=integrated_gradients[-72,4]
                  Z5_ig_24[0]=integrated_gradients[-72,5]
                  Z6_ig_24[0]=integrated_gradients[-72,6]
                  Z7_ig_24[0]=integrated_gradients[-72,7]
                  #print("A[:Q,:,N,np.newaxis]",np.mean(Z0[:Q,:,np.newaxis],axis=1).shape)
                  A_mean_baseline_ig1_24[i,0,:]=np.concatenate((Z0_ig_24[0,np.newaxis],Z1_ig_24[0,np.newaxis],Z2_ig_24[0,np.newaxis],Z3_ig_24[0,np.newaxis],Z4_ig_24[0,np.newaxis],Z5_ig_24[0,np.newaxis],Z6_ig_24[0,np.newaxis],Z7_ig_24[0,np.newaxis]),axis=0)
                  Z0_ig_96[0]=integrated_gradients[-1,0]
                  Z1_ig_96[0]=integrated_gradients[-1,1]
                  Z2_ig_96[0]=integrated_gradients[-1,2]
                  Z3_ig_96[0]=integrated_gradients[-1,3]
                  Z4_ig_96[0]=integrated_gradients[-1,4]
                  Z5_ig_96[0]=integrated_gradients[-1,5]
                  Z6_ig_96[0]=integrated_gradients[-1,6]
                  Z7_ig_96[0]=integrated_gradients[-1,7]

                  #print("A[:Q,:,N,np.newaxis]",np.mean(Z0[:Q,:,np.newaxis],axis=1).shape)
                  A_mean_baseline_ig1_96[i,0,:]=np.concatenate((Z0_ig_96[0,np.newaxis],Z1_ig_96[0,np.newaxis],Z2_ig_96[0,np.newaxis],Z3_ig_96[0,np.newaxis],Z4_ig_96[0,np.newaxis],Z5_ig_96[0,np.newaxis],Z6_ig_96[0,np.newaxis],Z7_ig_96[0,np.newaxis]),axis=0)
                  
                  mean = x_test.mean()  # Mean of the distribution
                  stddev =  x_test.std()  # Standard deviation of the distribution
                  tensor_size = x_test.size()
                  baseline_data = torch.randn(tensor_size) * stddev + mean
                  
                  # Create an instance of the LSTM model
                  best_model = TCNN(**model_params)
                  best_model.eval()
                  best_model.load_state_dict(best_params)
                  # Perform LRP on the model
                  relevance_scores = lrp(best_model, x_test, baseline_data, epsilon=P1[i])
                  relevance_scores=np.array(relevance_scores.detach().numpy())
                  print('relevance',relevance_scores.shape)
                  relevance_scores=relevance_scores.transpose(2,0,1)
                  print('relevance',relevance_scores.shape)
                  relevance_scores=np.mean(relevance_scores,axis=0)
                  #plt.figure()
                  #plt.plot(np.mean(np.array(relevance_scores[:,input2[0]:]),axis=1))

                  new_x_ticks = [0,1, 2, 3, 4, 5, 6,7]  # Your new x-axis tick values
                  new_x_labels=['SST','Salinity','SSH','WSC','MLD','buoyancy','PD','Fe']
                  #plt.xticks(new_x_ticks, new_x_labels)
                  #plt.ylabel("relevance")
                  #plt.xticks(rotation=45)
                  #plt.savefig("lrp_SO_baseline2_"+str(C1[k])+"_"+str(C2[i])+".pdf")
                  #Q=relevance_scores[:,input2[0]:,:].shape[0]
                  A_lrp0_24[0]=relevance_scores[-72,0]
                  A_lrp1_24[0]=relevance_scores[-72,1]
                  A_lrp2_24[0]=relevance_scores[-72,2]
                  A_lrp3_24[0]=relevance_scores[-72,3]
                  A_lrp4_24[0]=relevance_scores[-72,4]
                  A_lrp5_24[0]=relevance_scores[-72,5]
                  A_lrp6_24[0]=relevance_scores[-72,6]
                  A_lrp7_24[0]=relevance_scores[-72,7]

                  A_mean_lrp_baseline_24[i,0,:]=np.concatenate((A_lrp0_24[0,np.newaxis],A_lrp1_24[0,np.newaxis],A_lrp2_24[0,np.newaxis],A_lrp3_24[0,np.newaxis],A_lrp4_24[0,np.newaxis],A_lrp5_24[0,np.newaxis],A_lrp6_24[0,np.newaxis],A_lrp7_24[0,np.newaxis]),axis=0)
                  A_lrp0_96[0]=relevance_scores[-1,0]
                  A_lrp1_96[0]=relevance_scores[-1,1]
                  A_lrp2_96[0]=relevance_scores[-1,2]
                  A_lrp3_96[0]=relevance_scores[-1,3]
                  A_lrp4_96[0]=relevance_scores[-1,4]
                  A_lrp5_96[0]=relevance_scores[-1,5]
                  A_lrp6_96[0]=relevance_scores[-1,6]
                  A_lrp7_96[0]=relevance_scores[-1,7]

                  A_mean_lrp_baseline_96[i,0,:]=np.concatenate((A_lrp0_96[0,np.newaxis],A_lrp1_96[0,np.newaxis],A_lrp2_96[0,np.newaxis],A_lrp3_96[0,np.newaxis],A_lrp4_96[0,np.newaxis],A_lrp5_96[0,np.newaxis],A_lrp6_96[0,np.newaxis],A_lrp7_96[0,np.newaxis]),axis=0)
                  print('A_lrp5_96[k,0,np.newaxis]',A_lrp5_96[0,np.newaxis].shape)
        if N>0:
           X_test, Y_test= X_ss[-test_len-N:-N],Y_mm[-test_len-N:-N]
           x_test = torch.tensor(data = X_test).float()
           y_test = torch.tensor(data = Y_test).float()
           x_test = x_test.transpose(1, 2)
           

           best_model = TCNN(**model_params)
           best_model.eval()
           best_model.load_state_dict(best_params)

           tcn_prediction = best_model(x_test)

           A=264
           years=np.arange(1996-int(A/12),2015,1/12)
           test_len=input2[0]+output2[0]  #150/3
           
           Z=ts_int(tcn_prediction.view(-1).tolist(),
           T2[-test_len-N:-N,],
           start = T2[-test_len-N - 1, ]
           )
           #print("T",T.shape)
           #print("Z",len(Z))
           
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
           #plt.show()

           
              
           prediction1[N]=Z[-1]
           prediction2[N]=Z[-12]
           prediction3[N]=Z[-24]
           prediction4[N]=Z[-36]
           prediction5[N]=Z[-48]
           prediction6[N]=Z[-60]
           prediction7[N]=Z[-72]
           prediction8[N]=Z[-84]

           M=np.arange(1,156,2)
           plt.close('all')   
           #baseline_data = torch.zeros_like(x_test)
           #mean = x_test.mean()  # Mean of the distribution
           #stddev = x_test.std()  # Standard deviation of the distribution
           print('N',N)
           
           P=np.arange(40,140,10)
           P1=np.arange(0.9,1.9,1/10)/1150 #1150
           for i in range(len(P)):
                  integrated_gradients=calculate_integrated_gradients_uniformbaseline(best_model, x_test, num_steps=P[i])
                  integrated_gradients=np.array(integrated_gradients.detach().numpy())
                  integrated_gradients=integrated_gradients.transpose(2,0,1)
                  integrated_gradients=np.mean(integrated_gradients,axis=0)
                  #print("alpha",alpha.shape)
                  
                  Z0_ig_24[N]=integrated_gradients[-72,0]
                  Z1_ig_24[N]=integrated_gradients[-72,1]
                  Z2_ig_24[N]=integrated_gradients[-72,2]
                  Z3_ig_24[N]=integrated_gradients[-72,3]
                  Z4_ig_24[N]=integrated_gradients[-72,4]
                  Z5_ig_24[N]=integrated_gradients[-72,5]
                  Z6_ig_24[N]=integrated_gradients[-72,6]
                  Z7_ig_24[N]=integrated_gradients[-72,7]
                  #print("A[:Q,:,N,np.newaxis]",np.mean(Z0[:Q,:,np.newaxis],axis=1).shape)
                  A_mean_baseline_ig1_24[i,N,:]=np.concatenate((Z0_ig_24[N,np.newaxis],Z1_ig_24[N,np.newaxis],Z2_ig_24[N,np.newaxis],Z3_ig_24[N,np.newaxis],Z4_ig_24[N,np.newaxis],Z5_ig_24[N,np.newaxis],Z6_ig_24[N,np.newaxis],Z7_ig_24[N,np.newaxis]),axis=0) 
                  Z0_ig_96[N]=integrated_gradients[-1,0]
                  Z1_ig_96[N]=integrated_gradients[-1,1]
                  Z2_ig_96[N]=integrated_gradients[-1,2]
                  Z3_ig_96[N]=integrated_gradients[-1,3]
                  Z4_ig_96[N]=integrated_gradients[-1,4]
                  Z5_ig_96[N]=integrated_gradients[-1,5]
                  Z6_ig_96[N]=integrated_gradients[-1,6]
                  Z7_ig_96[N]=integrated_gradients[-1,7]

                  #print("A[:Q,:,N,np.newaxis]",np.mean(Z0[:Q,:,np.newaxis],axis=1).shape)
                  A_mean_baseline_ig1_96[i,N,:]=np.concatenate((Z0_ig_96[N,np.newaxis],Z1_ig_96[N,np.newaxis],Z2_ig_96[N,np.newaxis],Z3_ig_96[N,np.newaxis],Z4_ig_96[N,np.newaxis],Z5_ig_96[N,np.newaxis],Z6_ig_96[N,np.newaxis],Z7_ig_96[N,np.newaxis]),axis=0)
                  #constant_value = 1
                  # Create the tensor with constant values
                  #baseline = torch.full(x_test.size(), constant_value)
                  # Create an instance of the LSTM model
                  mean = x_test.mean() # Mean of the distribution
                  stddev = x_test.std() # Standard deviation of the distribution

                  # Define the size of the tensor (similar to x_test)
                  #tensor_size = x_test.size()

                  # Generate random samples from a Gaussian distribution
                  baseline_data = torch.randn(tensor_size) * stddev + mean
                  #constant_value=1
                  #baseline_data = torch.full(x_test.shape, constant_value)
                  #lower_bound = -2.0
                  #upper_bound = 2.0
                  #print('Z0_ig_96[k,:Q,0,np.newaxis]',Z0_ig_96[0,np.newaxis].shape)
                  # Create the uniform baseline by sampling from the uniform distribution
                  #baseline_data=torch.empty_like(x_test).uniform_(lower_bound, upper_bound)
                  best_model = TCNN(**model_params)
                  best_model.eval()
                  best_model.load_state_dict(best_params)
                  # Perform LRP on the model
                  relevance_scores = lrp(model, x_test, baseline_data,epsilon=P1[i])

                  relevance_scores=np.array(relevance_scores.detach().numpy())
                  relevance_scores=relevance_scores.transpose(2,0,1)
                  print('relevance',relevance_scores.shape)
                  relevance_scores=np.mean(relevance_scores,axis=0)
                  #Q=relevance_scores[:,input2[0]:,:].shape[0]
                  A_lrp0_24[N]=relevance_scores[-72,0]
                  A_lrp1_24[N]=relevance_scores[-72,1]
                  A_lrp2_24[N]=relevance_scores[-72,2]
                  A_lrp3_24[N]=relevance_scores[-72,3]
                  A_lrp4_24[N]=relevance_scores[-72,4]
                  A_lrp5_24[N]=relevance_scores[-72,5]
                  A_lrp6_24[N]=relevance_scores[-72,6]
                  A_lrp7_24[N]=relevance_scores[-72,7] 

                  A_mean_lrp_baseline_24[i,N,:]=np.concatenate((A_lrp0_24[N,np.newaxis],A_lrp1_24[N,np.newaxis],A_lrp2_24[N,np.newaxis],A_lrp3_24[N,np.newaxis],A_lrp4_24[N,np.newaxis],A_lrp5_24[N,np.newaxis],A_lrp6_24[N,np.newaxis],A_lrp7_24[N,np.newaxis],),axis=0)
                  A_lrp0_96[N]=relevance_scores[-1,0]
                  A_lrp1_96[N]=relevance_scores[-1,1]
                  A_lrp2_96[N]=relevance_scores[-1,2]
                  A_lrp3_96[N]=relevance_scores[-1,3]
                  A_lrp4_96[N]=relevance_scores[-1,4]
                  A_lrp5_96[N]=relevance_scores[-1,5]
                  A_lrp6_96[N]=relevance_scores[-1,6]
                  A_lrp7_96[N]=relevance_scores[-1,7]

                  A_mean_lrp_baseline_96[i,N,:]=np.concatenate((A_lrp0_96[N,np.newaxis],A_lrp1_96[N,np.newaxis],A_lrp2_96[N,np.newaxis],A_lrp3_96[N,np.newaxis],A_lrp4_96[N,np.newaxis],A_lrp5_96[N,np.newaxis],A_lrp6_96[N,np.newaxis],A_lrp7_96[N,np.newaxis]),axis=0)              
    #Q=[12,24,36,48,60,72,84,96]
    
    Q=12
    A=np.corrcoef(prediction1[::-1],T2[-264:,])
    skill_so[8]=A[1][0]
    A=np.corrcoef(prediction2[::-1],T2[-264-Q:-12,])
    skill_so[7]=A[1][0]
    A=np.corrcoef(prediction3[::-1],T2[-264-Q*2:-24,])
    skill_so[6]=A[1][0]
    A=np.corrcoef(prediction4[::-1],T2[-264-Q*3:-36,])
    skill_so[5]=A[1][0]
    A=np.corrcoef(prediction5[::-1],T2[-264-Q*4:-48,])
    skill_so[4]=A[1][0]
    A=np.corrcoef(prediction6[::-1],T2[-264-Q*5:-60,])
    skill_so[3]=A[1][0]
    A=np.corrcoef(prediction7[::-1],T2[-264-Q*6:-72,])
    skill_so[2]=A[1][0]
    A=np.corrcoef(prediction8[::-1],T2[-264-Q*7:-84,])
    skill_so[1]=A[1][0]
    skill_so[0]=1
    
    plt.figure(figsize=(10, 10),dpi=1200) 
    plt.plot(skill_so,'r')
    plt.ylabel("skill (correlation values)",fontsize=13)
    plt.xlabel("Time (years)",fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig('skill_NO3_2.pdf') 
    
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.plot(np.sort(skill_so,axis=0))
    plt.ylabel("skill (correlation values)",fontsize=20)
    plt.xlabel("Time (years)",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('skill_NO3_sort_2.pdf')

    plt.figure(figsize=(10, 10),dpi=1200)
    plt.plot(skill_so,'r', label='Skill TCNN')
    plt.plot(skill_reg,'b',label='Skill Ridge Reg')
    plt.plot(skill_linreg,'g',label='Skill Multilinear  Reg')
    plt.plot(persistence[0,::12],'k',label='Persistence')
    plt.ylabel("skill (correlation values)",fontsize=20)
    plt.xlabel("Time (years)",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize='large', loc='lower left')
    plt.savefig('skill_NO3_all.pdf')


    plt.figure(figsize=(10, 10),dpi=1200)
    plt.fill_between(years[-264:],(prediction1[::-1]-ci), (prediction1[::-1]+ci), color='green', alpha=0.5)
    plt.plot(years[-264:],prediction1[::-1],label = 'tcn',color='k',linewidth=1.75)
    plt.plot(years[-264:],T2[-264:,], label = 'real',color='r',linewidth=1.75)
    plt.ylabel("NO3 anomaly",fontsize=20)
    plt.xlabel("Years",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend()
    plt.savefig('forecast_composite_SO_8_2.pdf')

    plt.figure(figsize=(10, 10),dpi=1200)
    plt.fill_between(years[-264-Q*4:-Q*4],(prediction5[::-1]-ci), (prediction5[::-1]+ci), color='green', alpha=0.5)
    plt.plot(years[-264-Q*4:-Q*4],prediction5[::-1],label = 'tcn',color='k',linewidth=1.75)
    plt.plot(years[-264-Q*4:-Q*4],T2[-264-Q*4:-Q*4,], label = 'real',color='r',linewidth=1.75)
    plt.ylabel("NO3 anomaly",fontsize=20)
    plt.xlabel("Years",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend()
    plt.savefig('forecast_composite_SO_4_2.pdf')
    

    plt.figure(figsize=(10, 10),dpi=1200)
    plt.fill_between(years[-264-Q*6:-Q*6],(prediction7[::-1]-ci), (prediction7[::-1]+ci), color='green', alpha=0.5)
    plt.plot(years[-264-Q*6:-Q*6],prediction7[::-1],label = 'tcn',color='k',linewidth=1.75)
    plt.plot(years[-264-Q*6:-Q*6],T2[-264-Q*6:-Q*6,], label = 'real',color='r',linewidth=1.75)
    plt.ylabel("NO3 anomaly",fontsize=20)
    plt.xlabel("Years",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend()
    plt.savefig('forecast_composite_SO_2_2.pdf')
    plt.close("all")
    print("skill",skill_so)
    lags = range(98)
    


for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction6[::-1],i), A_tos[-264-Q*6:-Q*6,:,:]),np.var(np.roll(prediction6[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.6, 0.6, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg.squeeze(),v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_tos_2.pdf')

for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction6[::-1],i), A_zos[-264-Q*6:-Q*6,:,:]),np.var(np.roll(prediction6[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.6, 0.6, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg.squeeze(),v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_zos_2.pdf')


for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction6[::-1],i), A_so2[-264-Q*6:-Q*6,:,:]),np.var(np.roll(prediction6[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.5, 0.5, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg,v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_so_2.png')

for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction6[::-1],i), A_mld[-264-Q*6:-Q*6,:,:]),np.var(np.roll(prediction6[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.35, 0.35, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg.squeeze(),v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_mld_2.pdf')

for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction6[::-1],i), A_b[-264-Q*6:-Q*6,:,:]),np.var(np.roll(prediction6[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.5, 0.5, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg.squeeze(),v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_b_2.pdf')

for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction6[::-1],i), A_pd[-264-Q*6:-Q*6,:,:]),np.var(np.roll(prediction6[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.6, 0.6, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg,v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_pd_2.pdf')

for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction6[::-1],i), A_fe[-264-Q*6:-Q*6,:,:]),np.var(np.roll(prediction6[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.1, 0.1, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg,v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_fe_2.pdf')


A_ekman_pumping=np.array(A_ekman_pumping)
print('A_ekman_pumping',A_ekman_pumping.shape)
for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction6[::-1],i), A_ekman_pumping[-264-Q*6:-Q*6,:,:]),np.var(np.roll(prediction6[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.25, 0.25, 40, endpoint=True)
    fill=ax.contourf(lon_ep-180,lat_ep,T_reg,v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
plt.savefig('regression_no3_ek_2.pdf')
#for i in range(1):
#    T_reg=np.divide(covariance_map(np.roll(prediction8[::-1],i), A_wsc[-264-84:-84,:,:]),np.var(np.roll(prediction8[::-1],i)))
#    plt.figure(figsize=(13,6.2))
#    plt.subplot(211)
#    ax=fig.add_axes([0.1,0.1,0.8,0.8])
#    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
#    ax.coastlines(resolution='110m')
#    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
#    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
#    plt.title("Regression of WSC anomalies into the 2 year prediction")
#    lat_formatter = cticker.LatitudeFormatter()
#    ax.yaxis.set_major_formatter(lat_formatter)
#    v = np.linspace(-0.7, 0.7, 40, endpoint=True)
#    fill=ax.contourf(lon,lat,T_reg.squeeze(),v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
#    ax.set_aspect('auto')
#    cb = plt.colorbar(fill, orientation='vertical',shrink=1.1)
#    font_size = 20 # Adjust as appropriate.
#    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
#    polarCentral_set_latlim([-90,-30], ax)
#    plt.show()
#plt.savefig('regression_no3_wsc_2.png')



for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction1[::-1],i), A_tos[-264:,:,:]),np.var(np.roll(prediction1[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.6, 0.6, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg.squeeze(),v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_tos_9.pdf')

for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction1[::-1],i), A_zos[-264:,:,:]),np.var(np.roll(prediction1[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.6, 0.6, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg.squeeze(),v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_zos_9.pdf')


for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction1[::-1],i), A_so2[-264:,:,:]),np.var(np.roll(prediction1[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.5, 0.5, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg,v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_so_9.png')

for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction1[::-1],i), A_b[-264:,:,:]),np.var(np.roll(prediction1[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.5, 0.5, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg,v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_b_9.png')


for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction1[::-1],i), A_mld[-264:,:,:]),np.var(np.roll(prediction1[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.35, 0.35, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg,v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_mld_9.pdf')

for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction1[::-1],i), A_fe[-264:,:,:]),np.var(np.roll(prediction1[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.1, 0.1, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg,v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_fe_9.png')


for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction1[::-1],i), A_pd[-264:,:,:]),np.var(np.roll(prediction1[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.6, 0.6, 40, endpoint=True)
    fill=ax.contourf(lon,lat,T_reg,v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_pd_9.png')


ikman_pumping_anom = ekman_pumping_weighted.groupby('time.month') - ekman_pumping_clim
ekman_pumping_anom=ekman_pumping_anom/ekman_pumping_anom.std()
A_ekman_pumping=np.array(ekman_pumping_anom)
#A_ekman_pumping=A_ekman_pumping.reshape(1980,61,360)

print('lon_ep',lon_ep)
print('lat_ep',lat_ep)
for i in range(1):
    T_reg=np.divide(covariance_map(np.roll(prediction1[::-1],i), A_ekman_pumping[-264:,:,:]),np.var(np.roll(prediction1[::-1],i)))
    plt.figure(figsize=(10, 10),dpi=1200)
    plt.subplot(211)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.coastlines(resolution='110m')
    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    v = np.linspace(-0.1, 0.1, 40, endpoint=True)
    fill=ax.contourf(lon_ep-180,lat_ep,T_reg,v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_aspect('equal')
    cb = plt.colorbar(fill, orientation='vertical',shrink=0.8)
    font_size = 20 # Adjust as appropriate.
    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())
    plt.show()
plt.savefig('regression_no3_ek_9.png')

#for i in range(1):
#    T_reg=np.divide(covariance_map(np.roll(prediction1[::-1],i), A_wsc[-264:,:,:]),np.var(np.roll(prediction1[::-1],i)))
#    plt.figure(figsize=(13,6.2))
#    plt.subplot(211)
#    ax=fig.add_axes([0.1,0.1,0.8,0.8])
#    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
#    ax.coastlines(resolution='110m')
#    ax.gridlines()
    #ax.set_xticks(np.arange(-180,180,20), crs=ccrs.PlateCarree())
#    lon_formatter = cticker.LongitudeFormatter()
    #ax.xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
#    plt.title("Regression of WSC anomalies into the 8 year prediction")
#    lat_formatter = cticker.LatitudeFormatter()
#    ax.yaxis.set_major_formatter(lat_formatter)
#    v = np.linspace(-0.7, 0.7, 40, endpoint=True)
#    fill=ax.contourf(lon_wsc,lat_wsc,T_reg.squeeze(),v,cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
#    ax.set_aspect('auto')
#    cb = plt.colorbar(fill, orientation='vertical',shrink=1.1)
#    font_size = 20 # Adjust as appropriate.
#    cb.ax.tick_params(labelsize=font_size)
    #ax.contour(lon,lat, T_reg.squeeze(),color='k',cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
#    polarCentral_set_latlim([-90,-30], ax)
#    plt.show()
#plt.savefig('regression_no3_wsc_9.png')


feature = ['SST','Salinity','SSH','EP','MLD','buoyancy','PD','Fe']

new_x_ticks = [0,1, 2, 3, 4, 5, 6,7]  # Your new x-axis tick values
new_x_labels=['SST','Salinity','SSH','EP','MLD','buoyancy','PD','Fe']


# u undo the canceling
print("shap",np.nanmean(A_mean_baseline1_24,axis=0))

#plt.figure()
#plt.bar(feature,np.nanmean(np.nanmean(A_mean_baseline1_24[:Q,:,:],axis=0),axis=0))
#shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
#plt.ylabel('SHAP',fontsize=13)
#plt.xlabel('Input feature',fontsize=13)
#plt.xticks(rotation=20)
#plt.ylabel('feature',fontsize=13)
#plt.xlabel('composite_shap',fontsize=13)
#plt.savefig("shap_composite_2.png")

#PD=pd.DataFrame(A_mean_baseline_timeline1)
#print('np.nanmean(np.nanmean(np.nanmean(A_mean_lrp_baseline_24[:Q,:,:,:],axis=0),axis=0),axis=1)',np.nanmean(np.nanmean(np.nanmean(A_mean_lrp_baseline_24[:Q,:,:,:],axis=0),axis=0),axis=1))
print('lrp',np.nanmean(np.nanmean(A_mean_lrp_baseline_24,axis=0),axis=0))

plt.figure(figsize=(10, 10),dpi=1200)
plt.bar(feature,np.abs(np.nanmean(np.nanmean(A_mean_lrp_baseline_24,axis=0),axis=0)))
#plt.ylabel('Relevance',fontsize=10)
#plt.xlabel('Input features',fontsize=10)
plt.xticks(rotation=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

#plt.yscale('log')
plt.savefig("lrp_composite_2_2.pdf")

plt.figure(figsize=(10, 10),dpi=1200)
plt.bar(feature,np.abs(np.nanmean(np.nanmean(A_mean_baseline_ig1_24,axis=0),axis=0)))
#plt.ylabel('Gradients',fontsize=10)
#plt.xlabel('Input feature',fontsize=10)
plt.xticks(rotation=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.yscale('log')
plt.savefig("ig_composite_2_2.pdf")




#plt.figure()
#plt.bar(feature,np.nanmean(np.nanmean(A_mean_baseline1_96[:Q,:,:],axis=0),axis=0))
#shap.summary_plot(A_mean_baseline1,feature_names=feature,plot_type="auto")
#plt.ylabel('SHAP',fontsize=13)
#plt.xlabel('Input feature',fontsize=13)
#plt.xticks(rotation=20)
#plt.ylabel('feature',fontsize=13)
#plt.xlabel('composite_shap',fontsize=13)
#plt.savefig("shap_composite_8.png")


print('lrp',np.abs(np.nanmean(np.nanmean(A_mean_lrp_baseline_96,axis=0),axis=0)))
#PD=pd.DataFrame(A_mean_baseline_timeline1)
plt.figure(figsize=(10, 10),dpi=1200)
plt.bar(feature,np.abs(np.nanmean(np.nanmean(A_mean_lrp_baseline_96,axis=0),axis=0)))
#plt.ylabel('Relevance',fontsize=10)
#plt.xlabel('Input features',fontsize=10)
plt.xticks(rotation=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.yscale('log')
plt.savefig("lrp_composite_8_2.pdf")


plt.figure(figsize=(10, 10),dpi=1200)
plt.bar(feature,np.abs(np.nanmean(np.nanmean(A_mean_baseline_ig1_96,axis=0),axis=0)))
#plt.ylabel('Gradients',fontsize=5)
#plt.xlabel('Input feature',fontsize=10)
plt.xticks(rotation=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.yscale('log')
plt.savefig("ig_composite_8_2.pdf")


print('A_so',A_so2)
