import torch
import torch.nn as nn
import os
from hydra_utilities import import_data, create_noise, create_noise_shaped
from discriminators import AccuracyDiscriminatorClass, PrivacyDiscriminatorClass
from discriminators import DiversityDiscriminatorClass, PredictionDiscriminatorClass
from discriminators import SinglePointAccuracyDiscriminatorClass, DifferentialDiscriminatorClass
from torch_wass import torch_wasserstein_loss
from networks import Generator
import pandas as pd
import matplotlib.pyplot as plt
from process_generations import generate_report, generate_report_EM
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import datetime
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device, torch.cuda.is_available())

def attribute_dict():
    '''
    dictionary index:
        0 -> sensitive attribute -- int
        1 -> classification attribute -- int
        2 -> diversity attribute -- int
        3 -> diversity weights used for comparisons -- list 
    '''
    att_dict = {"heart":[0,1,13,[1,1]], "cervical_cancer":[0,1,32,[1,1]], "health_insurance":[1,14,19,[1,1]], "electric_grid":[4,0,13,[1,1]],
                "smarthome_wsu":[9,12,1,[1,1]], "iris":[0,1,4,[1,1]], "yes_spec":[0,1,4,[1,1]], "yes_1":[0,1,4,[1,1]], "yes_w":[0,1,4,[1,1]]}
    
    return(att_dict)

def make_selections(num_els):
    all_lists = []
    i = 0 
    full = []
    while i < num_els:
        full.append(i)
        i += 1
    all_lists.append(full)
    j = 1
    while j < num_els:
        new_missing = full.copy()
        new_missing.remove(j)
        all_lists.append(new_missing)
        j += 1
    k = 1
    while k < num_els:
        new_duo = [0, k]
        all_lists.append(new_duo)
        k += 1
    return(all_lists)

def run_through(data_list, run_name, num_epochs, gen_lr, disc_lr,sample_interval, load=0, dat_path=None, weight_path=None):
    att_dict = attribute_dict()
    i = 0
    epochs = num_epochs
    nz = 64
    batch_len = 33
    #batch_width = 14
    num_batches = 50
    lr = gen_lr
    disc_lr = disc_lr
    correcting = 0
    for items in data_list:
        batch_width = pd.read_csv(dat_path).shape[1]
        data_name = data_list[i][2:-4]
        data_list[i] = dat_path 
        if(load == 0):
            gen = Generator(nz, batch_len, batch_width).to(device)
        else:
            gen = Generator(nz, batch_len, batch_width).to(device)
            load_path = weight_path + 'gen_' + data_name
            gen.load_state_dict(torch.load(load_path))
        gen_optim = torch.optim.SGD(gen.parameters(), lr=lr)
        gen_scheduler = 0
        gen_loss = torch.nn.MSELoss()
        list_of_nets = init_nets(num_batches, batch_len, batch_width, data_list[i], disc_lr, nz, att_dict, data_name)
        if(load == 1):
            list_of_nets[0] = AccuracyDiscriminatorClass(num_batches, batch_len, batch_width, data_list[i], nz, disc_lr)
            load_path = weight_path + 'st_' + data_name
            list_of_nets[0].disc.load_state_dict(torch.load(load_path))
        elif(load == 2):
            load_path = weight_path + 'st_' + data_name
            list_of_nets[0] = AccuracyDiscriminatorClass(num_batches, batch_len, batch_width, data_list[i], nz, disc_lr)
            list_of_nets[0].disc.load_state_dict(torch.load(load_path))
            list_of_nets[1] = AccuracyDiscriminatorClass(num_batches, batch_len, batch_width, data_list[i], nz, disc_lr)
            list_of_nets[1].disc.load_state_dict(torch.load(load_path))
        elif(load == 3):
            load_path_re = weight_path + 're_' + data_name
            load_path_st = load_path = weight_path + 'st_' + data_name
            list_of_nets[0] = AccuracyDiscriminatorClass(num_batches, batch_len, batch_width, data_list[i], nz, disc_lr)
            list_of_nets[0].disc.load_state_dict(torch.load(load_path_st))
            list_of_nets[1] = SinglePointAccuracyDiscriminatorClass(num_batches, batch_len, batch_width, data_list[i], nz, disc_lr)
            list_of_nets[1].disc.load_state_dict(torch.load(load_path_re))
        data_sample = train_nets(epochs, gen, gen_optim, gen_loss, list_of_nets, nz, data_list[i], gen_scheduler,batch_len,batch_width,sample_interval,att_dict['yes_w'],load,data_name)
        process_output(data_sample, data_list[i], run_name)
        
        i += 1
        with open("./saved_weights/gen_"+data_name+"_"+str(epochs)+"_perf_"+ str(round(data_sample,4))+".pkl", 'wb') as f:
            pickle.dump(gen, f)
        l = 0
        for items in list_of_nets:
            with open("./saved_weights/disc_"+list_of_nets[l].name+"_"+data_name+"_"+str(epochs)+"_perf_"+ str(round(data_sample,4))+".pkl", 'wb') as f:
                pickle.dump(list_of_nets[l], f)
            l += 1
    
def init_nets(num_batches, batch_len, features, data, disc_lr, nz, att_dic, dat_name):
    ret_list = []
    
    Acc_disc = AccuracyDiscriminatorClass(num_batches, batch_len, features, data, nz, disc_lr)
    ret_list.append(Acc_disc)
    
    Priv_disc = PrivacyDiscriminatorClass(num_batches, batch_len, features, data, nz, att_dic[dat_name][0], disc_lr)
    ret_list.append(Priv_disc)
    
    Div_disc = DiversityDiscriminatorClass(num_batches, batch_len, features, data, nz, att_dic[dat_name][2], disc_lr)
    ret_list.append(Div_disc)
    
    Class_disc = PredictionDiscriminatorClass(num_batches, batch_len, features, data, nz, att_dic[dat_name][1], disc_lr)
    ret_list.append(Class_disc)

    Single_Acc_disc = SinglePointAccuracyDiscriminatorClass(num_batches, batch_len, features, data, nz, disc_lr)
    ret_list.append(Single_Acc_disc)
    
    return(ret_list)
    

def train_single(disc_element, number,gen, gen_opt, gen_loss,not_collapsed,gen_collapsed):
    i = 0
    while i < number:
        disc_element.train(gen, gen_opt, gen_loss,not_collapsed,gen_collapsed,reps=1)
        i += 1
    

def train_nets(epochs, gen, gen_opt, gen_loss, list_of_nets, nz, dat_source, gen_scheduler,batch_len,features,interval,att_dict,load, dat_name):
    generative_combined_loss = []
    i = 0
    not_collapsed=1
    gen_collapsed=0
    correcting=0
    EM_chart_list = []
    curr_EM = 1
    gen_learning_rate_chart = []
    disc_learning_rate_chart = []
    mean_dist_chart = []
    all_disc_to_gen_losses = []
    all_disc_names = []
    name_tracker = 0
    post_train = 0
    for names in list_of_nets:
        new_list = []
        all_disc_to_gen_losses.append(new_list)
        all_disc_names.append(list_of_nets[name_tracker].name[1:])
        name_tracker += 1
    mean_losses = [0]*len(list_of_nets)
    
    while i < (epochs + post_train):
        j = 0        
        gen_opt.zero_grad()
        comb_losses = 0
        thresh = 50 #allows filtering the saving of bad results
        for items in list_of_nets:
            loss_tracking_list = []
            loss_tracking_list_sizes = []
            if(curr_EM < thresh or j == 0):
                if(i < epochs):
                    curr_loss = list_of_nets[j].train(gen, gen_opt, gen_loss,not_collapsed,gen_collapsed,mean_losses[j],reps=1)
                else:
                    curr_loss = list_of_nets[j].train(gen, gen_opt, gen_loss,not_collapsed,gen_collapsed,mean_losses[j],reps=1)
                all_disc_to_gen_losses[j].append(curr_loss)
                
                comb_losses += curr_loss
        
            j += 1
        k = 0
        gen_opt.step()
        generative_combined_loss.append(comb_losses)
        i += 1
        if(i % interval == 0 or i == 1):
            noise = create_noise_shaped(10,400, features, nz)
            nout = gen(noise)
            
            
            nout = torch.flatten(nout, end_dim=2)
            nout = pd.DataFrame(nout.detach().cpu().numpy())             


            MYDIR = ("./saved_output/" + str(dat_name))
            CHECK_FOLDER = os.path.isdir(MYDIR)


            if not CHECK_FOLDER:
                os.makedirs(MYDIR)
                print("created folder : ", MYDIR)

            else:
                print(MYDIR, "folder already exists.")

            MYDIR = ("./reports/" + str(dat_name))
            CHECK_FOLDER = os.path.isdir(MYDIR)


            if not CHECK_FOLDER:
                os.makedirs(MYDIR)
                print("created folder : ", MYDIR)

            else:
                print(MYDIR, "folder already exists.")



            dat_loc = "./saved_output/"+str(dat_name)+"/generator_out_TEMP.csv" 
            png_loc = "./reports/"+str(dat_name)+"/generator_out_"+str(dat_name)+'_'+str(i)
            nout.to_csv(dat_loc, index=False)
            curr_EM, mean_dist = generate_report(dat_loc, dat_source, [], [], png_loc, att_dict[0], att_dict[1], att_dict[2])
            if(curr_EM < 100.25): 
                dat_loc = "./saved_output/"+str(dat_name)+"/generator_out_"+str(i)+"_dist_"+str(curr_EM)+'_load_status_'+str(load)+".csv"
                nout.to_csv(dat_loc, index=False)      
            print("########################")
            print("########################")
            print("########################")
            print("EM distance at " +str(i)+ " epochs:" + str(curr_EM))
            print("########################")
            print("########################")
            print("########################")
            EM_chart_list.append(curr_EM)
            mean_dist_chart.append(mean_dist)
            curr_g_loss = comb_losses
            curr_real_d_loss = list_of_nets[0].last_disc_loss
            print("Generative Loss is:", curr_g_loss)
            
            
            print("Discriminative Loss is:", curr_real_d_loss)
            plt.clf()
            plt.plot(EM_chart_list)
            plt.title("Observed EM Distance through training")
            plt.savefig("./plots/updating/EM_dist"+dat_name+'_load_status_'+str(load)+"_tracker_updating.png")

            plt.clf()
            plt.plot(mean_dist_chart)
            plt.title("Mean distance across all metrics")
            plt.savefig("./plots/updating/Mean_dist"+dat_name+'_load_status_'+str(load)+"_tracker_updating.png")

            plt.clf()
            plt.plot(generative_combined_loss)
            plt.title("Total loss to generator")
            plt.ylim(0,7)
            plt.savefig("./plots/updating/gen_loss_"+dat_name+'_load_status_'+str(load)+"_tracker_updating.png")

            plt.clf()
            plt.plot(list_of_nets[0].own_loss_chart)
            plt.title("First disc (usually acc, 0) loss")
            plt.savefig("./plots/updating/disc_0"+dat_name+'_load_status_'+str(load)+"_tracker.png")

            plt.clf()
            plt.plot(list_of_nets[1].own_loss_chart)
            plt.title("Second disc (usually indiv, 1) loss")
            plt.savefig("./plots/updating/disc_1"+dat_name+'_load_status_'+str(load)+"_tracker.png")

            plt.clf()
            plot_it = 0
            for lines in all_disc_to_gen_losses:
                plt.plot(all_disc_to_gen_losses[plot_it], label=all_disc_names[plot_it])
                plt.legend()
                plot_it += 1            
            plt.title("Individual losses seen by generator")
            plt.ylim(-0.1,1.75)
            plt.savefig("./plots/updating/indiv_gen_loss_"+dat_name+'_load_status_'+str(load)+"_tracker_updating.png")

    
    return(curr_EM)

if(__name__ == '__main__'):
    run_through(['./data_folder/heart.csv'], None,250,0.00001,0.00001,500,0)
    
    
            
        
    
    

