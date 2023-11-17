import torch 
import torch.nn as nn
from torch_wass import torch_wasserstein_loss, torch_reid_loss, torch_single_point_loss
from networks import Generator, Standard_Discriminator, Diversity_Discriminator, Reidentification_Discriminator
from hydra_utilities import import_data, create_noise, generate, calc_wass_dist,save_generation
from hydra_utilities import calc_even, import_data_batch, calc_quick, score, create_noise_shaped
from hydra_utilities import import_data_all, batch_from_dat_tens, import_data_spec
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NORMALIZE_INPUT = 1 #0 to not normalize, 1 to normalize
SHUFFLE = 0  #0 to shuffle, 1 to not shuffle
MULTI_CSV = False  #Do we pull from multiple CSVS?


class AccuracyDiscriminatorClass:
    '''
    Standard discriminator, analyzing a group of data
    '''
    def __init__(self, num_batches, batch_length, features, data, nz,lr=0.0001):
        self.name = "_Distribution_Realism"
        self.disc = Standard_Discriminator(batch_length,features).to(device)
        self.optim = torch.optim.SGD(self.disc.parameters(), lr=lr)
        self.disc_scheduler = ReduceLROnPlateau(self.optim, 'min', factor=0.05)
        self.loss = torch.nn.MSELoss()
        self.own_loss_chart = []
        self.gen_loss_chart = []
        self.batch_length = batch_length
        self.num_batches = num_batches
        self.features = features
        self.data = data
        self.last_disc_loss = 0
        self.last_gen_loss = 0
        self.nz = nz
        if(MULTI_CSV):
            self.data_stored = import_data_spec(mode=2,path=self.data,normalize=NORMALIZE_INPUT)
        else:
            self.data_stored = import_data_all(data, norm=NORMALIZE_INPUT)
    def train(self, gen, g_optim, g_loss, not_collapsed=1,gen_collapsed=0,scalar=1, reps=1):
        '''
        train gen here
        '''
        noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
        gen_out = gen(noise)
        gen_labs = torch.ones(self.num_batches)[:,None].to(device)
        gen_disc_out = self.disc(gen_out)
        gen_loss = g_loss(gen_disc_out,gen_labs)
        self.last_gen_loss = gen_loss.detach().cpu().numpy()
        self.gen_loss_chart.append(gen_loss.detach().cpu().numpy())
        gen_loss.backward()
        i = 0
        while(i < reps and not_collapsed):
            self.optim.zero_grad()
            noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
            gen_out = gen(noise)
            new_noise = (torch.rand(gen_out.shape))
            if(MULTI_CSV):
                real_data = self.data_stored[torch.randint(self.data_stored.shape[0], (self.num_batches,))] 
            else:
                real_data = batch_from_dat_tens(self.data_stored,self.num_batches, self.batch_length, non_shuffle=SHUFFLE)
                real_data = torch.Tensor(real_data).to(device)

            
            new_noise = (torch.rand(gen_out.shape))*0.01 - 0.005
            real_data = real_data.to(device) + new_noise.to(device)
            real_data = torch.clamp(real_data, min=0.0)
            disc_labels_pos = torch.ones(self.num_batches).to(device) 
            disc_labels_neg = torch.zeros(self.num_batches).to(device) 
            
            disc_labs_all = torch.cat((disc_labels_pos, disc_labels_neg))
            
            real_data = real_data[:,0,:,:]
            real_data = real_data[:,None,:,:] 
            data_in_all = torch.cat((real_data.float(),gen_out.float()))
            disc_out_all = self.disc(data_in_all)
            disc_loss_all = self.loss(disc_out_all.flatten(), disc_labs_all.flatten())
            
            
            disc_loss_all += 0.001
            disc_loss_all.backward()

            self.optim.step()
            self.own_loss_chart.append(disc_loss_all.detach().cpu().numpy())
            self.last_disc_loss = disc_loss_all.detach().cpu().numpy()


            i += 1
        return(self.last_gen_loss)
    def test_loss(self):
        return(self.own_loss_chart)
    def get_disc(self):
        return(self.disc)
    
    
class PrivacyDiscriminatorClass:
    '''
    Try to reidentify sensitive attribute(s) from other
    info
    '''
    def __init__(self, num_batches, batch_length, features, data, nz, sensitive_attribute,scalar=1, lr=0.0001):
        self.name = "_Reidentification"
        self.disc = Reidentification_Discriminator(batch_length, features).to(device)
        self.optim = torch.optim.SGD(self.disc.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.own_loss_chart = []
        self.gen_loss_chart = []
        self.batch_length = batch_length
        self.num_batches = num_batches
        self.features = features
        self.data = data
        self.last_disc_loss = 0
        self.last_gen_loss = 0
        self.sensitive_attribute = sensitive_attribute
        self.nz = nz
        self.set_g_loss = torch_reid_loss()
        self.data_stored = import_data_all(data, norm=NORMALIZE_INPUT)
    def train(self, gen, g_optim, g_loss,not_collapsed=1, gen_collapsed=0,scalar=1, reps=1):
        noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz).to(device)
        
        gen_out = gen(noise)
        gen_labs = gen_out[:,:,:, self.sensitive_attribute].clone().detach()
        gen_out[:,:,:, self.sensitive_attribute] = 0
        gen_labs = torch.squeeze(gen_labs)
        gen_disc_out = self.disc(gen_out)
        gen_loss = g_loss(gen_disc_out.flatten(), gen_labs.flatten())
        gen_loss = -1*gen_loss*scalar


        self.last_gen_loss = gen_loss.detach().cpu().numpy()
        self.gen_loss_chart.append(gen_loss.detach().cpu().numpy())
        
        
        gen_loss.backward()
        i = 0
        while i < reps:
            self.optim.zero_grad()
            
            noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
            real_data = batch_from_dat_tens(self.data_stored,self.num_batches, self.batch_length, non_shuffle=SHUFFLE)
            real_data = torch.Tensor(real_data).to(device)
            disc_dat = real_data
            
            
            disc_labs = disc_dat[:,:,:,self.sensitive_attribute].clone().detach()
            disc_labs = torch.squeeze(disc_labs)
            disc_dat[:,:,:,self.sensitive_attribute] = 0
            disc_out = self.disc(disc_dat)
            disc_loss = self.loss(disc_out.flatten(), disc_labs.flatten())
            disc_loss.backward()
            self.optim.step()
            self.own_loss_chart.append(disc_loss.detach().cpu().numpy())
            self.last_disc_loss = disc_loss.detach().cpu().numpy()
            i += 1
        return(np.abs(self.last_gen_loss))
    def test_loss(self):
        return(self.own_loss_chart)
    def get_disc(self):
        return(self.disc)
    
class DiversityDiscriminatorClass:
    def __init__(self, num_batches, batch_length, features, data, nz, diverse_element, lr=0.0001):
        self.name = "_Diversity"
        self.disc = Standard_Discriminator(batch_length, features).to(device)
        self.optim = torch.optim.SGD(self.disc.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.own_loss_chart = []
        self.gen_loss_chart = []
        self.batch_length = batch_length
        self.features = features
        self.data = data
        self.last_disc_loss = 0
        self.last_gen_loss = 0
        self.diverse_element = diverse_element
        self.nz = nz
        self.num_batches = num_batches
        self.data_sample = import_data(200, data)
        self.data_stored = import_data_all(data, norm=NORMALIZE_INPUT)
        self.desired_div = calc_even(self.data_sample, self.diverse_element)
        
    def train(self, gen, g_optim, g_loss,not_collapsed=1, gen_collapsed=0,scalar=1, reps=1):
        noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
        
        gen_out = gen(noise)
        gen_labs = torch.ones(self.num_batches)*self.desired_div
        gen_labs = gen_labs.to(device)
        gen_disc_out = self.disc(gen_out)
        gen_loss = g_loss(gen_labs.flatten(), gen_disc_out.flatten())*scalar
        

        self.last_gen_loss = gen_loss.detach().cpu().numpy()
        self.gen_loss_chart.append(gen_loss.detach().cpu().numpy())

        gen_loss.backward()
        i = 0
        while i < reps:
            self.optim.zero_grad()
            noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
            gen_out = gen(noise)
            real_data = batch_from_dat_tens(self.data_stored,self.num_batches, self.batch_length, non_shuffle=SHUFFLE)
            real_data = torch.Tensor(real_data).to(device)
            real_data = torch.cat((gen_out.detach(),real_data))
            div_list = []
            j = 0
            for items in real_data:
                divers = calc_quick(real_data[j,:,:], self.diverse_element)
                div_list.append(divers)
                j += 1
            disc_labs = torch.Tensor(div_list).to(device)
            disc_dat = real_data
            disc_out = self.disc(disc_dat)
            disc_loss = self.loss(disc_out.flatten(), disc_labs.flatten())
            disc_loss.backward()
            self.optim.step()
            self.own_loss_chart.append(disc_loss.detach().cpu().numpy())
            self.last_disc_loss = disc_loss.detach().cpu().numpy()
            i += 1
        return(self.last_gen_loss)
    def test_loss(self):
        return(self.own_loss_chart)
    def get_disc(self):
        return(self.disc)
    
class PredictionDiscriminatorClass:
    def __init__(self, num_batches, batch_length, features, data, nz, classifying_attribute, lr=0.0001):
        self.name = "_Accuracy"
        self.disc = Reidentification_Discriminator(batch_length, features).to(device)
        self.optim = torch.optim.SGD(self.disc.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.own_loss_chart = []
        self.gen_loss_chart = []
        self.batch_length = batch_length
        self.features = features
        self.data = data
        self.last_disc_loss = 0
        self.last_gen_loss = 0
        self.classifying_attribute = classifying_attribute
        self.num_batches = num_batches
        self.nz = nz
        
        self.data_sample = import_data(200, data)
        self.score = score(self.data_sample,self.classifying_attribute)
        self.data_stored = import_data_all(data, norm=NORMALIZE_INPUT)
    def train(self, gen, g_optim, g_loss,not_collapsed=1,  gen_collapsed=0, scalar=1,reps=1):
        noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz).to(device)
        
        gen_out = gen(noise)
        gen_labs = gen_out[:,:,:, self.classifying_attribute].clone().detach()
        gen_out[:,:,:, self.classifying_attribute] = 0
        gen_labs = torch.squeeze(gen_labs)
        
        gen_disc_out = self.disc(gen_out)
        gen_loss = g_loss(gen_disc_out.flatten(), gen_labs.flatten())*scalar
        self.last_gen_loss = gen_loss.detach().cpu().numpy()
        self.gen_loss_chart.append(gen_loss.detach().cpu().numpy())
        


        gen_loss.backward()
        
        i = 0
        while i < reps:
            self.optim.zero_grad()
            
            noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
            real_data = batch_from_dat_tens(self.data_stored,self.num_batches, self.batch_length, non_shuffle=SHUFFLE)
            real_data = torch.Tensor(real_data).to(device)
            disc_dat = real_data
            
            
            disc_labs = disc_dat[:,:,:,self.classifying_attribute].clone().detach()
            disc_labs = torch.squeeze(disc_labs)
            disc_dat[:,:,:,self.classifying_attribute] = 0
            disc_out = self.disc(disc_dat)
            disc_loss = self.loss(disc_out.flatten(), disc_labs.flatten())
            disc_loss.backward()
            self.optim.step()
            self.own_loss_chart.append(disc_loss.detach().cpu().numpy())
            self.last_disc_loss = disc_loss.detach().cpu().numpy()
            i += 1
        return(self.last_gen_loss)
    def test_loss(self):
        return(self.own_loss_chart)
    def get_disc(self):
        return(self.disc)
        
    
    
class SinglePointAccuracyDiscriminatorClass:
    '''
    Standard discriminator, analyzing a group of data
    '''
    def __init__(self, num_batches, batch_length, features, data, nz, lr=0.0001):
        self.name = "_Point-Wise_Realism"
        self.disc = Reidentification_Discriminator(batch_length,features).to(device)
        self.optim = torch.optim.SGD(self.disc.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.own_loss_chart = []
        self.gen_loss_chart = []
        self.batch_length = batch_length
        self.num_batches = num_batches
        self.features = features
        self.data = data
        self.last_disc_loss = 0
        self.last_gen_loss = 0
        self.nz = nz
        self.single_g_loss = torch_single_point_loss()
        if(MULTI_CSV):
            self.data_stored = import_data_spec(mode=2,path=self.data,normalize=NORMALIZE_INPUT)
        else:
            self.data_stored = import_data_all(data, norm=NORMALIZE_INPUT)
    def train(self, gen, g_optim, g_loss,not_collapsed=1, gen_collapsed=0, scalar=1,reps=1):
        g_optim.zero_grad()
        noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
        gen_out = gen(noise)

        new_noise = (torch.rand(gen_out.shape)*0.05) - 0.025
        gen_out = gen_out + new_noise.to(device)

        gen_labs = torch.ones(self.num_batches)
        gen_disc_out = self.disc(gen_out)
        gen_loss = self.single_g_loss(gen_labs.flatten(), gen_disc_out.flatten())*scalar
        
        self.last_gen_loss = gen_loss.detach().cpu().numpy()
        self.gen_loss_chart.append(gen_loss.detach().cpu().numpy())

        gen_loss.backward()
        i = 0
        while i < reps:
            self.optim.zero_grad()
            noise = create_noise_shaped(self.num_batches,self.batch_length, self.features, self.nz)
            
            gen_out = gen(noise)
            new_noise = (torch.rand(gen_out.shape)*0.05) - 0.025
            gen_out = gen_out + new_noise.to(device)
            if(MULTI_CSV):
                real_data = self.data_stored[torch.randint(len(self.data_stored), (self.num_batches,))] 
            else:
                real_data = batch_from_dat_tens(self.data_stored,self.num_batches, self.batch_length, non_shuffle=SHUFFLE)
                real_data = torch.Tensor(real_data).to(device)

            new_noise = (torch.rand(real_data.shape)*0.025) - 0.0125
            real_data = real_data.to(device) + new_noise.to(device)

            real_data = real_data[:,0,:,:]
            real_data = real_data[:,None,:,:]  #FIX THIS

            disc_labels_pos = torch.ones(self.num_batches, self.batch_length) 
            disc_labels_neg = torch.zeros(self.num_batches, self.batch_length) 
            disc_labs = torch.cat((disc_labels_pos, disc_labels_neg)).to(device)
            disc_dat = torch.cat((real_data.float(), gen_out.float()))
            disc_out = self.disc(disc_dat)
            disc_loss = self.loss(disc_out.flatten(), disc_labs.flatten())
            disc_loss.backward()
            self.optim.step()
            self.own_loss_chart.append(disc_loss.detach().cpu().numpy())
            self.last_disc_loss = disc_loss.detach().cpu().numpy()
            i += 1
        return(self.last_gen_loss)
    def test_loss(self):
        return(self.own_loss_chart)
    def get_disc(self):
        return(self.disc)    


if(__name__ == '__main__'):
    print(__name__)