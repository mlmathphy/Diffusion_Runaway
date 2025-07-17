import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.io import loadmat
from scipy.stats import norm

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_output = 'output/'   # where to save trained models
x_dim = 3
sde_dt = 0.2
datadir = os.path.join(f'data_{int(sde_dt * 100):02d}')
figdir = os.path.join(
    f'fig_{int(sde_dt*100):02d}')

# 2. Load normalization info & diff_scale
data_inf = torch.load(os.path.join(datadir, 'data_inf.pt'), weights_only=False)
xTrain_mean = data_inf['xTrain_mean'].to(device)
xTrain_std = data_inf['xTrain_std'].to(device)
yTrain_mean = data_inf['yTrain_mean'].to(device)
yTrain_std = data_inf['yTrain_std'].to(device)
diff_scale = data_inf['diff_scale']

# 3. Define your NN class again (same as training)
class FN_Net(nn.Module):
    def __init__(self, input_dim, output_dim, hid_size):
        super(FN_Net, self).__init__()
        self.input = nn.Linear(input_dim, hid_size)
        self.fc1 = nn.Linear(hid_size, hid_size)
        self.output = nn.Linear(hid_size, output_dim)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        return self.output(x)

# 4. Load trained model
FN = FN_Net(x_dim*2, x_dim, 50).to(device)
FN.load_state_dict(torch.load(os.path.join(datadir, 'FN.pth'), map_location=device, weights_only=True))
FN.eval()



# 4. Load trained model
FN = FN_Net(x_dim*2, x_dim, 50).to(device)
FN.load_state_dict(torch.load(os.path.join(datadir, 'FN.pth'), map_location=device, weights_only=True))
FN.eval()

def load_weight_checkpoint(filepath):
    """
    Load checkpoint
    """
    checkpoint = torch.load(filepath, weights_only=True)
    model = EscapeModel().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def load_weight_model():
    """
    Load model
    """
    savedir = root_output
    filename = 'NN_time_weight'

    # print(savedir + filename + '.pt')
    device = torch.device("cuda")
    model = load_weight_checkpoint(savedir + filename + '.pt')
    model.to(device)
    model.eval()
    return model
# Define deeper model with dropout
class EscapeModel(nn.Module):
    def __init__(self):
        super(EscapeModel, self).__init__()
        self.dim = 4
        self.hid_size = 256
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.hid_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(self.hid_size, self.hid_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(self.hid_size, self.hid_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(self.hid_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    


def compute_production_rates(prediction, y_true_transform, p_star=1.25):
    """
    Compute the production rates (PR) for predicted and test data.
    
    Parameters:
        y_pred (np.ndarray): Predicted samples, shape (N, 2), columns are [p_par, p_per].
        y_test (np.ndarray): Reference samples, shape (N, 2), columns are [p_par, p_per].
        p_star (float): Momentum threshold for production rate calculation.
        
    Returns:
        PR_pred (float): Production rate for predicted data.
        PR_test (float): Production rate for test data.
    """
    # --- Transform to (theta, p) ---
    Npath_true = len(y_true_transform)
    Npath_pred = len(prediction)
    # print(Npath_pred)
    print(Npath_true)
    y_true = np.zeros((Npath_true, x_dim))
    y_true[:, 1] = np.arctan2(y_true_transform[:, 1], y_true_transform[:, 0])  # eta
    y_true[:, 0] = np.sqrt(y_true_transform[:, 0]**2 + y_true_transform[:, 1]**2)  # p
    y_true[:,2] = y_true_transform[:, 2]

    y_pred = np.zeros((Npath_pred, x_dim))
    y_pred[:, 1] = np.arctan2(prediction[:, 1], prediction[:, 0])  # eta
    y_pred[:, 0] = np.sqrt(prediction[:, 0]**2 + prediction[:, 1]**2)  # p
    y_pred[:,2] = prediction[:, 2]


    # period boundary condition for y_pred
    p_min = 0.5
    y_pred[y_pred[:,0] <= p_min,0] = p_min

    def _compute_pr(p_vals, p_star):
        p_max = np.max(p_vals)
        p_min = np.min(p_vals)

        if p_star > p_max:
            return 0.0
        
        pkde = gaussian_kde(p_vals)
        ind_total = np.linspace(p_min, p_max, 201)
        kde_total = pkde(ind_total)
        PR_total = (p_max - p_min) / 200 * 0.5 * (np.sum(kde_total[:-1]) + np.sum(kde_total[1:]))

        ind_partial = np.linspace(p_star, p_max, 201)
        kde_partial = pkde(ind_partial)
        PR_partial = (p_max - p_star) / 200 * 0.5 * (np.sum(kde_partial[:-1]) + np.sum(kde_partial[1:]))

        return PR_partial / PR_total if PR_total > 0 else 0.0

    p_pred = y_pred[:, 0]
    p_test = y_true[:, 0]
    
    PR_pred = _compute_pr(p_pred, p_star)
    PR_test = _compute_pr(p_test, p_star)
    print(PR_test)
    return PR_pred, PR_test


def get_production_rate_vs_time(t_plot_array, sde_dt, true_init_transform, ode_path_true,
                                FN, xTrain_mean, xTrain_std, yTrain_mean, yTrain_std, diff_scale,
                                p_star=1.25):
    """
    Evaluate PR_pred and PR_test for multiple time steps.

    Args:
        t_plot_array (np.ndarray): Array of times (in float) at which to evaluate PR.
        sde_dt (float): Time step size.
        Npath (int): Number of sample paths.
        x_dim (int): Dimension of the state space.
        true_init_transform (torch.Tensor): Transformed initial condition (shape: [x_dim]).
        ode_path_true (np.ndarray): Ground truth path data from RE_2dfun (shape: [Npath, T, x_dim]).
        FN (torch.nn.Module): Trained neural network model.
        xTrain_mean, xTrain_std, yTrain_mean, yTrain_std (torch.Tensor): Normalization stats.
        diff_scale (float): Diffusion scale used in post-processing.
        device (torch.device): 'cpu' or 'cuda'.
        p_star (float): Threshold momentum for PR calculation.

    Returns:
        t_plot_array (np.ndarray): The same input time array.
        PR_pred_list (list): List of predicted PR values for each time.
        PR_test_list (list): List of test PR values for each time.
    """
    PR_pred_list = []
    PR_test_list = []

    for t_plot in t_plot_array:
        ode_time_steps = int(np.floor(t_plot / sde_dt))
        print(ode_time_steps)
        y_true_transform = ode_path_true[ode_time_steps,:,:]
        y_true_transform = y_true_transform[y_true_transform[:,2] < 1,:]

        # Predict over time steps
        x_pred_new = torch.tensor(true_init_transform, dtype=torch.float32, device=device)
        Npath_pred0=true_init_transform.shape[0]
        for _ in range(ode_time_steps):
            noise = torch.randn(Npath_pred0, x_dim).to(device)
            input_tensor = (torch.hstack((x_pred_new, noise)) - xTrain_mean) / xTrain_std
            prediction = FN(input_tensor) * yTrain_std + yTrain_mean
            prediction = (prediction.to('cpu').detach().numpy() / diff_scale + x_pred_new.to('cpu').detach().numpy())
            x_pred_new = torch.tensor(prediction).to(device, dtype=torch.float32)

        with open('data/weight_mean_std.npy', 'rb') as f:
            mu_weight = np.load(f)
            s_weight = np.load(f)

        Escape = load_weight_model()
        t_column = np.full((true_init.shape[0], 1), t_plot)
        true_init_with_time = np.concatenate([true_init, t_column], axis=1)
        test0 = (true_init_with_time-mu_weight)/s_weight
        test0 = torch.tensor(test0, dtype=torch.float32).to(device)
        weight_pred = Escape(test0).to('cpu').detach().numpy().flatten()
        random_values = np.random.uniform(0, 1, size=len(weight_pred))
        mask_to_keep = weight_pred > random_values.flatten()
        prediction = prediction[mask_to_keep,:]


        PR_pred, PR_test = compute_production_rates(prediction, y_true_transform, p_star=p_star)
        PR_pred_list.append(PR_pred)
        PR_test_list.append(PR_test)

    return t_plot_array, np.array(PR_pred_list), np.array(PR_test_list)



#----------------------------------------------------------------------------------------------------
#  Main
#-----------------------------------------------------------------------------------------------------
# Set parameters

sde_T = 20.0
p_star=1.75

ode_time_steps = int(np.floor(sde_T/sde_dt))
print('ode_time_steps is:', ode_time_steps)
ode_mean_pred = np.zeros((ode_time_steps,x_dim))
ode_std_pred = np.zeros((ode_time_steps,x_dim))
ode_mean_true = np.zeros((ode_time_steps,x_dim))
ode_std_true = np.zeros((ode_time_steps,x_dim))


# Define path to saved file
idx_maxwell = 10
ode_path_file = os.path.join(datadir, f'ode_path_max{idx_maxwell}.npy')

loaded = np.load(ode_path_file, allow_pickle=True).item()
true_init = loaded['true_init']
true_init_transform = loaded['true_init_transform']
ode_path_true = loaded['ode_path_true']

t_plot_array = np.arange(2,21,2)

_,PR_pred_list,PR_test_list = get_production_rate_vs_time(t_plot_array, sde_dt,true_init_transform, ode_path_true,
                                FN, xTrain_mean, xTrain_std, yTrain_mean, yTrain_std, diff_scale,
                                p_star)


fig, axes = plt.subplots(figsize=(4, 4), ncols=1, nrows=1)
axes.plot(t_plot_array,PR_pred_list, label='Diffusion', color="r",linewidth=3)
axes.plot(t_plot_array,PR_test_list, label='MC', color="b",linestyle='dashed',linewidth=3)
axes.set_xlabel('time t',fontsize=12)
axes.set_ylabel('$n_{RE}$',fontsize=12)  
axes.set_xlim(2,20)
# axes.set_ylim(0.3,0.6)
axes.legend()
plt.tight_layout()
plt.grid()
plt.show()
plt.savefig('fig_20/production_time.png', dpi='figure')
 
fig, axes = plt.subplots(figsize=(4, 4), ncols=1, nrows=1)
axes.plot(t_plot_array,np.abs(PR_pred_list - PR_test_list)/PR_test_list, label='relative error', color="b",linestyle='dashed',linewidth=3)
axes.set_xlabel('time t',fontsize=12)
axes.set_ylabel(r'$\Delta n_{RE}$/$n_{RE}$',fontsize=12)  
axes.set_xlim(2,20)
# axes.set_ylim(0.3,0.6)
axes.legend()
plt.tight_layout()
plt.grid()
plt.show()
plt.savefig('fig_20/production_time_ratio.png', dpi='figure')
 



# # plot for early 
# t_plot = 26
# ode_time_steps_plot = int(np.floor(t_plot/sde_dt))
# y_true_transform = ode_path_true[:,ode_time_steps_plot,:]
# x_pred_new = true_init_transform.to(device,dtype=torch.float32)

# for jj in range(ode_time_steps_plot):
    
#     prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
#     prediction = ( prediction.to('cpu').detach().numpy() /diff_scale +x_pred_new.to('cpu').detach().numpy()  ) 
#     ode_mean_pred[jj,:] = np.mean(prediction,axis=0)
#     ode_std_pred[jj,:] = np.std(prediction,axis=0)
#     x_pred_new= torch.tensor( prediction  ).to(device,dtype= torch.float32)

# PR_pred, PR_test = compute_production_rates(prediction, y_true_transform, p_star=1.25)

# print(PR_pred)
# print(PR_test)



# # plot for early 
# t_plot = 10
# ode_time_steps_plot = int(np.floor(t_plot/sde_dt))
# y_true_transform = ode_path_true[:,ode_time_steps_plot,:]
# x_pred_new = true_init_transform.to(device,dtype=torch.float32)
# for jj in range(ode_time_steps_plot):
    
#     prediction = FN( (torch.hstack((x_pred_new,torch.randn(Npath,x_dim).to(device,dtype= torch.float32)))-xTrain_mean)/xTrain_std  ) * yTrain_std + yTrain_mean 
#     prediction = ( prediction.to('cpu').detach().numpy() /diff_scale +x_pred_new.to('cpu').detach().numpy()  )   
#     ode_mean_pred[jj,:] = np.mean(prediction,axis=0)
#     ode_std_pred[jj,:] = np.std(prediction,axis=0)    
#     x_pred_new= torch.tensor( prediction  ).to(device,dtype= torch.float32)

# PR_pred, PR_test = compute_production_rates(prediction, y_true_transform, p_star=1.25)

# print(PR_pred)
# print(PR_test)
