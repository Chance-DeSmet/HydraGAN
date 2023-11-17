
import torch
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random
import glob
import os
print("Device is", device, torch.cuda.is_available())


def import_data_spec(mode=0, path='./yes_folder/', normalize=False):
    if(mode==0):
        #Get a single spec CSV, don't shuffle
        data = pd.read_csv(path)
        data = torch.tensor(data.to_numpy())
        out = data
    if(mode==1):
        #get a single csv, shuffle
        data = pd.read_csv(path)
        data = torch.tensor(data.to_numpy())
        data=data[torch.randperm(data.size()[0])]
        out = data
    if(mode==2):
        #get all csvs in folder, shuffle
        data_list=[]
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        random.shuffle(csv_files)
        i = 0
        for csv in csv_files:
            data = pd.read_csv(csv)
            data = torch.tensor(data.to_numpy())
            if(i == 0):
                data_list = data[None,:,:]
            else:
                data_list = torch.cat((data_list,data[None,:,:]))
            i += 1
        out = data_list
    if(mode==3):
        #Get all csvs in folder, average them
        data_list=[]
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        i = 0
        for csv in csv_files:
            data = pd.read_csv(csv)
            data = torch.tensor(data.to_numpy())
            if(i == 0):
                data_list = data[None,:,:]
            else:
                data_list = torch.cat((data_list,data[None,:,:]))
            i += 1
        data_list = torch.mean(data_list,dim=0)
        out = data_list
    if(normalize):
        out -= out.min()
        out /=out.max()
    return(out)

def create_noise(sample_size, nz=5):
    return(torch.randn(sample_size, nz)[:,None, None, :])

def create_noise_shaped(num_batches, batch_len, features, nz):
    noise = torch.randn(num_batches, 1, batch_len+nz, features+nz)
    return(torch.randn(num_batches, nz)[:,None, None, :].to(device))
    return(noise)


def import_data_non_normed(dat_len, dat_ref):
    df = pd.read_csv(dat_ref)
    out = df.sample(n=dat_len, replace=True).to_numpy()
    return(out)

def diff_priv(data):
    return(0)

def import_data_all(dat_ref, norm=1):
    df = pd.read_csv(dat_ref)
    df = df.apply(pd.to_numeric)
    if(norm == 1):
        df = (df-df.min())/(df.max()-df.min() + 0.0000001)
    out = torch.Tensor(df.to_numpy())
    print(out)
    return(out)



def import_data(dat_len, dat_ref, norm=1):
    df = pd.read_csv(dat_ref)
    df = df.apply(pd.to_numeric)
    if(norm==1):
        df = (df-df.min())/((df.max()-df.min()) + 0.0000001)
        out = df.sample(n=dat_len, replace=True).to_numpy()
    else:        
        out = df.to_numpy()
    return(out)

def import_data_batch(num_batches, batch_len, dat_ref):
    i = 0
    dat_list = []
    while i < num_batches:
        temp_dat = import_data(batch_len, dat_ref)
        dat_list.append(temp_dat)
        i += 1
    dat_list = np.array(dat_list)
    dat_list = torch.Tensor(dat_list)
    return(dat_list[:,None,:,:])

def denorm_data(data, dat_ref):
    df = pd.read_csv(dat_ref)
    new_df = pd.DataFrame(data)
    new_df.columns = df.columns
    new_df = (new_df * (df.max() - df.min())) + df.min()
    return(new_df)
    
def generate(gen,num):
    noise = create_noise(num)
    gen_out = gen(noise)
    return(gen_out)

def wass_from_dat(dat_1, dat_2):
    i = 0
    dist_list = []
    while i < dat_1.shape[1]:
        temp_dist = wasserstein_distance(dat_1[:,i], dat_2[:,i])
        dist_list.append(temp_dist)
        i += 1
    out = np.array(dist_list)
    return(np.mean(out))

def multi_wass(dat_1, dat_2):
    i = 0
    dist_list = []
    while i < dat_1.shape[1]:
        temp_dist = wasserstein_distance(dat_1[:,i], dat_2[:,i])
        temp_dist = np.abs(temp_dist)
        dist_list.append(temp_dist)
        i += 1
    return(dist_list)

def calc_wass_dist(gen,dat_loc,num=50):
    dat = generate(gen,num)[0,0,:,:].detach().numpy()
    real_dat = import_data(50,dat_loc)
    dist = multi_wass(dat,real_dat)
    mean_dist = np.mean(np.array(dist))
    return(mean_dist)

def calc_diversity(dat, orig_dat_ref, index):
    dat = denorm_data(dat, orig_dat_ref).to_numpy()
    row = dat[:,index]
    row = np.round(row, 0)    
    values, counts = np.unique(row, return_counts=True)
    N = np.sum(counts) + 0.0
    div = 0
    i = 0
    for items in values:
        if(counts[i] == 0):
            div+= 0
        else:
            div += (counts[i]/N)*np.log2(counts[i]/N)
        i += 1
    max_div = np.log2(values.shape[0])
    return(-1*div, max_div)
def save_generation(gen,i,data):
    df = pd.read_csv(data)
    noise = create_noise(1)
    out = gen(noise).detach().numpy()[0,0,:,:]
    out = pd.DataFrame(out)
    out.columns = df.columns
    out.to_csv("../saved_gens/epoch_"+str(i)+".csv")
    
def clc_normal_wass(dat_loc):
    i = 0 
    dist = []
    while i < 100:
        dat1 = import_data(50,dat_loc)
        dat2 = import_data(50,dat_loc)
        dist.append(multi_wass(dat1,dat2))        
        i += 1
    mean_dist = np.mean(np.array(dist))
    return(mean_dist)

def score(dat, index):
    dat = np.squeeze(dat)
    dat_len = dat.shape[0]
    midpoint = int(0.8*dat_len)
    y = dat[:,index]
    dat[:,index] = 0
    X = dat
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X[0:midpoint,:],y[0:midpoint])
    return(regressor.score(X[midpoint:],y[midpoint:]))
    
def calc_quick(dat, index):
    row = dat[:,index]
    row = np.around(row.cpu(), 1)
    values, counts = np.unique(row, return_counts=True)
    N = np.sum(counts) + 0.0
    div = 0
    i = 0
    for items in values:
        div += counts[i]/N*np.log(counts[i]/N)
        i += 1
    return(div)    

def calc_even(dat, index):
    row = dat[:,index]
    row = np.around(row, 1)
    values, counts = np.unique(row, return_counts=True)
    N = values.shape[0] + 0.0
    div = 0
    i = 0
    for items in values:
        div += 1/N*np.log(1/N)
        i += 1
    return(div)  

def plot_data(loc):
    dat = pd.read_csv(loc)
    dat_nump = dat.to_csv()
    plt.pcolormesh( dat_nump, cmap = 'coolwarm')
    plt.title( "Heat Map of Data" )
    plt.savefig('./data_visualized.png')
    
def batch_from_dat_tens(dat,batches,batch_len, non_shuffle=0):
    d = dat
    i = 0
    if(non_shuffle == 0):
        while (i < batches):
            if(i == 0):
                indices = torch.Tensor(random.sample(range(0,d.shape[0]),batch_len))
                indices = indices.long()
                indices = d[None,None,indices,:]
            else:
                ind = torch.Tensor(random.sample(range(0,d.shape[0]),batch_len))[None,None,:]
                ind = ind.long()
                ind = d[ind,:]
                indices = torch.cat((indices,ind))
            i += 1
    else:
        while(i < batches):
            if(i == 0):
                indices = d[None,None,:,:]
            else:
                indices = torch.cat((indices,d[None,None,:,:]))
            i += 1
    return(indices)

def batch_from_dat_tens_of_specs(dat,batches,batch_len):
    d = dat
    i = 0
    while (i < batches):
        if(i == 0):
            indices = torch.Tensor(random.sample(range(0,d.shape[0]),batch_len))
            indices = indices.long()
            indices = d[indices,None,:,:]
        else:
            ind = torch.Tensor(random.sample(range(0,d.shape[0]),batch_len))
            ind = ind.long()
            ind = d[ind,None,:,:]
            indices = torch.cat((indices,ind))
        i += 1
        
