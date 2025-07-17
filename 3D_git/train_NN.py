
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import os
root_data = 'data/'
root_output = 'output/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_dim=3
sde_dt = 0.2
diff_scale = np.array([1.5,3.0, 20]) 

# Convert to tensors
xTrain = np.load(os.path.join(root_data, 'xTrain.npy'))
yTrain = np.load(os.path.join(root_data, 'yTrain.npy'))
print("xTrain shape:", xTrain.shape)
print("yTrain shape:", yTrain.shape)

#------------------------------------------------------------------------------
#            Define the architecture of the  neural network
#------------------------------------------------------------------------------

class FN_Net(nn.Module):
    
    def __init__(self, input_dim, output_dim, hid_size):
        super(FN_Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_size = hid_size
        
        self.input = nn.Linear(self.input_dim,self.hid_size)
        self.fc1 = nn.Linear(self.hid_size,self.hid_size)
        self.output = nn.Linear(self.hid_size,self.output_dim)

        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)
    
    def forward(self,x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = self.output(x)
        return x

    def update_best(self):

        self.best_input_weight = torch.clone(self.input.weight.data)
        self.best_input_bias = torch.clone(self.input.bias.data)
        self.best_fc1_weight = torch.clone(self.fc1.weight.data)
        self.best_fc1_bias = torch.clone(self.fc1.bias.data)
        self.best_output_weight = torch.clone(self.output.weight.data)
        self.best_output_bias = torch.clone(self.output.bias.data)

    def final_update(self):

        self.input.weight.data = self.best_input_weight 
        self.input.bias.data = self.best_input_bias
        self.fc1.weight.data = self.best_fc1_weight
        self.fc1.bias.data = self.best_fc1_bias
        self.output.weight.data = self.best_output_weight
        self.output.bias.data = self.best_output_bias

#========================================================================================================
#   train F(x0,z)=x1_hat-x0 using labled data (x0,z, x1_hat-x0)
#========================================================================================================
# Check for infinite values in yTrain
is_finite_yTrain = np.isfinite(yTrain) & ~np.isnan(yTrain)
# Exclude rows with infinite values
xTrain_filtered = xTrain[is_finite_yTrain.all(axis=1)]
yTrain_filtered = yTrain[is_finite_yTrain.all(axis=1)]


train_size_new = xTrain_filtered.shape[0]

print('trainning data size:', train_size_new)
# Generate random indices for shuffling
indices = np.random.permutation(train_size_new)
# Shuffle xTrain and yTrain using the generated indices
xTrain_shuffled = xTrain_filtered[indices]
yTrain_shuffled = yTrain_filtered[indices]

xTrain_mean = np.mean(xTrain_filtered, axis=0, keepdims=True)
xTrain_std = np.std(xTrain_filtered, axis=0, keepdims=True)
yTrain_mean = np.mean(yTrain_filtered, axis=0, keepdims=True)
yTrain_std = np.std(yTrain_filtered, axis=0, keepdims=True)


xTrain_new = (xTrain_shuffled - xTrain_mean) / xTrain_std
yTrain_new = (yTrain_shuffled - yTrain_mean) / yTrain_std

# Convert data to a tensor
xTrain_new = torch.tensor(xTrain_new, dtype=torch.float32).to(device)
yTrain_new = torch.tensor(yTrain_new, dtype=torch.float32).to(device)
xTrain_mean = torch.tensor(xTrain_mean, dtype=torch.float32).to(device)
xTrain_std = torch.tensor(xTrain_std , dtype=torch.float32).to(device)
yTrain_mean = torch.tensor(yTrain_mean, dtype=torch.float32).to(device)
yTrain_std  = torch.tensor(yTrain_std , dtype=torch.float32).to(device)

dataname2 = os.path.join(root_data, 'data_inf.pt')
torch.save({'xTrain_mean': xTrain_mean,'xTrain_std': xTrain_std, 'yTrain_mean': yTrain_mean, 'yTrain_std': yTrain_std, 'diff_scale': diff_scale}, dataname2)

print('xTrain_mean:', xTrain_mean)
print('xTrain_std:', xTrain_std)
print( 'yTrain_mean:', yTrain_mean)
print( 'yTrain_std:', yTrain_std)


NTrain = int(train_size_new* 0.9)
NValid = int(train_size_new * 0.1)

xValid_normal = xTrain_new [NTrain:,:]
yValid_normal = yTrain_new [NTrain:,:]

xTrain_normal = xTrain_new [:NTrain,:]
yTrain_normal = yTrain_new [:NTrain,:]

learning_rate = 0.001
FN = FN_Net(x_dim*2, x_dim, 128).to(device)
FN.zero_grad()
optimizer = optim.Adam(FN.parameters(), lr = learning_rate, weight_decay=1e-6)
criterion = nn.MSELoss()

best_valid_err = 5.0
n_iter = 20000
for j in range(n_iter):
    optimizer.zero_grad()
    pred = FN(xTrain_normal)
    loss = criterion(pred,yTrain_normal)
    loss.backward()
    optimizer.step()

    pred1 = FN(xValid_normal)
    valid_loss = criterion(pred1,yValid_normal)
    if valid_loss < best_valid_err:
        FN.update_best()
        best_valid_err = valid_loss

    if j%100==0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr}")
        print(j,loss,valid_loss)

FN.final_update()

FN_path = os.path.join(root_data, 'FN.pth')
torch.save(FN.state_dict(), FN_path)
