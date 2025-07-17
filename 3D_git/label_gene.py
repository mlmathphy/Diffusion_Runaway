
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import expm
import os
from tqdm import tqdm  # For progress bars
import faiss



if torch.cuda.is_available():
    cuda_device_number = 0  # Set this to the desired GPU number (0, 1, etc.)
    device = torch.device(f'cuda:{cuda_device_number}')  # Use the specified GPU
else:
    device = torch.device('cpu')  # Fallback to CPU if CUDA is not available
    cuda_device_number = None
print('device',device)
torch.set_default_dtype(torch.float32)


torch.manual_seed(12345678)
np.random.seed(12312414)

def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

#-----------------------------------------------------
#            Setup the generative diffusion model
#-----------------------------------------------------
def cond_alpha(t, dt):
    # conditional information
    # alpha_t(0) = 1
    # alpha_t(1) = 0
    return 1 - t + dt


def cond_sigma2(t, dt):
    # conditional sigma^2
    # sigma2_t(0) = 0
    # sigma2_t(1) = 1
    # return (1-scalar) + t*scalar
    return t + dt

def f(t, dt):
    # f=d_(log_alpha)/dt
    alpha_t = cond_alpha(t,dt)
    f_t = -1.0/(alpha_t)
    return f_t


def g2(t, dt):
    # g = d(sigma_t^2)/dt -2f sigma_t^2
    dsigma2_dt = 1.0
    f_t = f(t,dt)
    sigma2_t = cond_sigma2(t,dt)
    g2 = dsigma2_dt - 2*f_t*sigma2_t
    return g2

def g(t,dt):
    return (g2(t,dt))**0.5


odeslover_time_steps = 5000
t_vec = torch.linspace(1.0,0.0,odeslover_time_steps +1)
def ODE_solver(zt,x_sample,z_sample,x0_test,time_steps):
    # log_weight_likelihood = -1.0* torch.sum( (x0_test[:,None,:]-x_sample)**2/2 , axis = 2, keepdims= False)
    # weight_likelihood =torch.exp(log_weight_likelihood)
    for j in range(time_steps): 
        t = t_vec[j+1]
        dt = t_vec[j] - t_vec[j+1]

        score_gauss = -1.0*(zt[:,None,:]-cond_alpha(t,dt)*z_sample)/cond_sigma2(t,dt)

        log_weight_gauss= -1.0* torch.sum( (zt[:,None,:]-cond_alpha(t,dt)*z_sample)**2/(2*cond_sigma2(t,dt)) , axis =2, keepdims= False)
        weight_temp = torch.exp( log_weight_gauss )
        # weight_temp = weight_temp*weight_likelihood
        weight = weight_temp/ torch.sum(weight_temp,axis=1, keepdims=True)
        score = torch.sum(score_gauss*weight[:,:,None],axis=1, keepdims= False)  
        
        zt= zt - (f(t,dt)*zt-0.5*g2(t, dt)*score) *dt
    return zt



#----------------------------------------------------------------------------------------------------
#  Find indices of first short_size(=2000) elements of x_samples closest to each element of x0_train
#  this is used in score estimation
#-----------------------------------------------------------------------------------------------------

def process_chunk(it_n_index, it_size_x0train, short_size,x_sample, x0_train, train_size):
    x0_train_index_initial = np.empty((train_size, short_size ), dtype=int)
    gpu = faiss.StandardGpuResources()  # Initialize GPU resources each time
    index = faiss.IndexFlatL2(x_dim)  # Create a FAISS index for exact searches
    gpu_index = faiss.index_cpu_to_gpu(gpu, 0, index)
    gpu_index.add(x_sample)  # Add the chunk of x_sample to the index
    for jj in range(it_n_index):
        start_idx = jj * it_size_x0train
        end_idx = min((jj + 1) * it_size_x0train, train_size)
        x0_train_chunk = x0_train[start_idx:end_idx]

        # Perform the search
        _, index_initial = gpu_index.search(x0_train_chunk, short_size)
        x0_train_index_initial[start_idx:end_idx,:] = index_initial 

        if jj % 500 == 0:
            print('find indx iteration:', jj, it_size_x0train)
    # Cleanup resources
    del gpu_index
    del index
    del gpu
    return x0_train_index_initial

#=========================================================================
#    3D parameter
#=========================================================================
x_dim = 3  # Dimensions: (p, xi, r)

######################## some paramters you can choose######################## 
##############################################################################
sde_dt = 0.2   # Time step for training data, mc_dt=0.005 for MC sampling

diff_scale = np.array([1.5,3.0, 20])  # make sure the standard deviation of xy_diff cannot be too small, should be big like (0.5 to 1), you can test different version
N_sample = 100000000  # define your training sample size, this should be large since the data domain is large, should be at least millions 
############################################################################   




root_data = 'data/'
root_output = 'output/'


if os.path.exists(root_data):
    print("Directory exists.")
    xx_sample = np.load(os.path.join(root_data,'x_sample.npy'))
    yy_sample = np.load(os.path.join(root_data,'y_sample.npy'))

else:
    print("Directory does not exist")
# Set the model in evaluation mode (if needed)



x_sample = xx_sample.copy()
y_sample = yy_sample.copy()
x_sample[:,0] = xx_sample[:,0]*xx_sample[:,1]
x_sample[:,1] = xx_sample[:,0]*np.sqrt(1-xx_sample[:,1]**2)
y_sample[:,0] = yy_sample[:,0]*yy_sample[:,1]
y_sample[:,1] = yy_sample[:,0]*np.sqrt(1-yy_sample[:,1]**2)



np.random.seed(42)

# Assuming x_sample is already loaded
# Get the total number of samples
total_size = len(x_sample)


# Randomly select N_sample indices without replacement
random_indices = np.random.choice(total_size, size=min(total_size,N_sample) , replace=False)

# Get the selected samples
x_sample = x_sample[random_indices]
y_sample = y_sample[random_indices]



xy_diff = (y_sample-x_sample)*diff_scale
# print('difference of x_sample and y_sample is: ', xy_diff)
print('mean and std of xy_diff are: ',  np.mean(xy_diff, axis = 0),np.std(xy_diff, axis=0))

del y_sample 

sample_size = x_sample.shape[0]
print('sample size:', sample_size )



#========================================================================================================
#   choose 50,000 x0 from sample data to generate labled data (x0,x1_hat-x0,z)
#========================================================================================================
train_size = 50000
selected_row_indices =  np.random.permutation(sample_size)[:train_size]
x0_train = x_sample[selected_row_indices]
print('size of x0_train is: ', x0_train.shape )


short_size = 2047

it_size_x0train = 500
it_n_index = train_size // it_size_x0train

x_short_indx = process_chunk(it_n_index, it_size_x0train, short_size,x_sample, x0_train, train_size)
print('finish finding indx',short_size, it_size_x0train) 




x_short = x_sample[x_short_indx]
z_short = xy_diff[x_short_indx]

np.save(os.path.join(root_data, f"data_training_x_short.npy"), x_short)
np.save(os.path.join(root_data, f"data_training_z_short.npy"), z_short)
np.save(os.path.join(root_data, f"data_training_x0train.npy"), x0_train)

del x_sample, xy_diff,x_short_indx

zT = np.random.randn(train_size,x_dim)
yTrain = np.zeros((train_size,x_dim))
it_size = min(60000,train_size)
it_n = int(train_size/it_size)
for jj in range(it_n):

    start_idx = jj * it_size
    end_idx = min((jj + 1) * it_size, train_size)
    it_zt = torch.tensor(zT[start_idx: end_idx]).to(device, dtype=torch.float32)
    it_x0 =  torch.tensor(x0_train[start_idx: end_idx]).to(device, dtype=torch.float32)

    x_mini_batch =  torch.tensor(x_short[start_idx:end_idx]).to(device, dtype=torch.float32)
    z_mini_batch =  torch.tensor(z_short[start_idx: end_idx]).to(device, dtype=torch.float32)
    
    y_temp = ODE_solver( it_zt , x_mini_batch, z_mini_batch,  it_x0, odeslover_time_steps)
    yTrain[start_idx: end_idx, :x_dim] = y_temp.to('cpu').detach().numpy()

    if jj%5==0:
        print(jj)
   
xTrain = np.hstack((x0_train,zT))

np.save(os.path.join(root_data, 'xTrain.npy'), xTrain)
np.save(os.path.join(root_data, 'yTrain.npy'), yTrain)

