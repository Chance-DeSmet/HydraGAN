import numpy as np
import pandas as pd
from hydra_utilities import import_data, calc_wass_dist, calc_diversity, wass_from_dat
from eval_methods import util_score, priv_score, util_score_normal, priv_score_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from new_radar import make_radar_chart
from sklearn.metrics import matthews_corrcoef
import torch
import torch.nn as nn


def pull_scores(gen_csv,orig_csv, sens,acc,div):
    num_import = 1000
    
    gen_data = import_data(num_import, gen_csv)
    orig_data = import_data(num_import, orig_csv)
    max_diversity = 0
    EM_scores = 0
    GEN_accuracy = 0
    SENS_accuracy = 0
    SPECIFIC_accuracy = 0
    DIV_scores = 0
    EM_scores = 1-wass_from_dat(gen_data, orig_data)
    GEN_accuracy = 1-util_score_normal(orig_data, gen_data, orig_csv, sens)
    SENS_accuracy = priv_score_normal(orig_data, gen_data, orig_csv, sens)
    SPECIFIC_accuracy = 1 -priv_score(orig_data, gen_data, acc)
    diver, new_max = calc_diversity(gen_data, orig_csv, div)
    DIV_scores = diver 
    max_diversity = max(max_diversity, new_max)
    return(EM_scores, GEN_accuracy, SENS_accuracy, SPECIFIC_accuracy, DIV_scores)


def generate_report_abs(gen_csv,orig_csv,comp_csvs,comp_csv_names,disc_list_name, sens,acc,div, k):
    num_import = 1000
    gen_data = import_data(num_import, gen_csv)
    orig_data = import_data(num_import, orig_csv)
    i = 0
    comp_list = []
    for items in comp_csvs:
        comp_list.append(import_data(num_import, comp_csvs[i]))
        i += 1
    max_diversity = 0
    EM_scores = []
    GEN_accuracy = []
    SENS_accuracy = []
    SPECIFIC_accuracy = []
    DIV_scores = []
    EM_scores.append(1-wass_from_dat(gen_data, orig_data))
    GEN_accuracy.append(1-util_score_normal(orig_data, gen_data, orig_csv, sens))
    SENS_accuracy.append(priv_score_normal(orig_data, gen_data, orig_csv, sens))
    SPECIFIC_accuracy.append(1 -priv_score(orig_data, gen_data, acc))
    diver, new_max = calc_diversity(gen_data, orig_csv, div)
    DIV_scores.append(diver)
    max_diversity = max(max_diversity, new_max)
    i = 0
    for items in comp_list:
        EM_scores.append(1-wass_from_dat(comp_list[i], orig_data))
        GEN_accuracy.append(1-util_score_normal(orig_data, comp_list[i], orig_csv, sens))
        SENS_accuracy.append(priv_score_normal(orig_data, comp_list[i], orig_csv, sens))
        SPECIFIC_accuracy.append(1 - priv_score(orig_data, comp_list[i], acc))
        diver, new_max = calc_diversity(comp_list[i], orig_csv, div)
        DIV_scores.append(diver)
        max_diversity = max(max_diversity, new_max)
        i += 1
    DIV_scores = DIV_scores/max_diversity
    name_list = ['HydraGAN '] + comp_csv_names
    metrics_list = ['Mean Distance', 'L1 Loss', 'MSE Loss', 'KL Divergence Loss']
    l1_list = []
    mean_dist_list = []
    kl_dist_list = []
    mse_loss_list = []
    i = 0
    for items in name_list:
        
        curr_name = name_list[i]
        print(curr_name)
        MCC_true = [1.0]*5
        MCC_pred = (EM_scores[i], GEN_accuracy[i], SENS_accuracy[i], SPECIFIC_accuracy[i], DIV_scores[i])
        mean_metric = 1 - np.mean(MCC_pred)
        L1_metric = nn.L1Loss()
        MSE_metric = nn.MSELoss()
        KL_metric = nn.KLDivLoss()
        best = torch.ones(len(MCC_pred))
        
        l1_dist = L1_metric(best,torch.tensor(MCC_pred))
        
        MSE_dist = MSE_metric(best,torch.tensor(MCC_pred))
        KL_dist = KL_metric(best,torch.tensor(MCC_pred))
        auc = 0
        j = 0
        for items in MCC_pred:
            if(j == 0):
                auc += 0.5 * MCC_pred[0]*MCC_pred[-1]*np.sin(360/5)
            else:
                auc += 0.5 * MCC_pred[j]*MCC_pred[j-1]*np.sin(360/5)
            j += 1
            
        name_list[i] = curr_name + str(round(auc,2))
        
        l1_list.append(np.around(l1_dist.numpy(),4))
        mean_dist_list.append(np.around(mean_metric,4))
        kl_dist_list.append(np.around(KL_dist.numpy(),4))
        mse_loss_list.append(np.around(MSE_dist.numpy(), 4))
        i +=1 
    
    give_DF = pd.DataFrame({'Methods': name_list,
                   '$EM$': EM_scores,
                   '$RA$': GEN_accuracy,
                   '$RE$': SENS_accuracy,
                   '$TA$': SPECIFIC_accuracy,
                   '$DI$': DIV_scores
                   })
    
    
    overall_DF = pd.DataFrame({'Methods': name_list,
                   '$EM$': EM_scores,
                   '$RA$': GEN_accuracy,
                   '$RE$': SENS_accuracy,
                   '$TA$': SPECIFIC_accuracy,
                   '$DI$': DIV_scores,
                   'Mean Distance': mean_dist_list,
                   'L1Loss': l1_list,
                   'MSELoss': mse_loss_list,
                   'KLLoss': kl_dist_list
        
        })
  
    overall_DF = overall_DF.round(4)
    give_DF.iloc[:,1:] = (give_DF.iloc[:,1:].abs())
    give_DF.to_csv("./"+disc_list_name+".csv", index=False)
    
    overall_DF.iloc[:,1:] = (overall_DF.iloc[:,1:].abs())
    overall_DF.to_csv("./"+disc_list_name+"_overall.csv", index=False)
    
    make_radar_chart(give_DF, "./"+disc_list_name, disc_list_name, k)
    return(1-EM_scores[0])

def generate_report(gen_csv,orig_csv,comp_csvs,comp_csv_names,disc_list_name, sens,acc,div, avg=0):
    num_import = 1000
    
    gen_data = import_data(num_import, gen_csv)
    orig_data = import_data(num_import, orig_csv)
    
    i = 0
    comp_list = []
    for items in comp_csvs:
        comp_list.append(import_data(num_import, comp_csvs[i]))
        i += 1
    max_diversity = 0
    EM_scores = []
    GEN_accuracy = []
    SENS_accuracy = []
    SPECIFIC_accuracy = []
    DIV_scores = []
    
    EM_scores.append(1-wass_from_dat(gen_data, orig_data))
    EM_scores.append(1-wass_from_dat(orig_data, orig_data))
    GEN_accuracy.append(1-util_score_normal(orig_data, gen_data, orig_csv, sens))
    GEN_accuracy.append(1-util_score_normal(orig_data, orig_data, orig_csv, sens))
    SENS_accuracy.append(priv_score_normal(orig_data, gen_data, orig_csv, sens))
    SENS_accuracy.append(priv_score_normal(orig_data, orig_data, orig_csv, sens))
    SPECIFIC_accuracy.append(1 -priv_score(orig_data, gen_data, acc))
    SPECIFIC_accuracy.append(1- priv_score(orig_data, orig_data, acc))
    diver, new_max = calc_diversity(gen_data, orig_csv, div)
    DIV_scores.append(diver)
    max_diversity = max(max_diversity, new_max)
                    
    diver, new_max = calc_diversity(orig_data, orig_csv, div)
    DIV_scores.append(diver)
    max_diversity = max(max_diversity, new_max)
    
    
    i = 0
    for items in comp_list:
        EM_scores.append(1-wass_from_dat(comp_list[i], orig_data))
        GEN_accuracy.append(1-util_score_normal(orig_data, comp_list[i], orig_csv, sens))
        SENS_accuracy.append(priv_score_normal(orig_data, comp_list[i], orig_csv, sens))
        SPECIFIC_accuracy.append(1 - priv_score(orig_data, comp_list[i], acc))
        diver, new_max = calc_diversity(comp_list[i], orig_csv, div)
        DIV_scores.append(diver)
        max_diversity = max(max_diversity, new_max)
        i += 1
        
    DIV_scores = DIV_scores/max_diversity
    name_list = ['HydraGAN', 'Original Data'] + comp_csv_names
    metrics_list = ['Mean Distance', 'L1 Loss', 'MSE Loss', 'KL Divergence Loss']
    l1_list = []
    mean_dist_list = []
    kl_dist_list = []
    mse_loss_list = []
    auc_list = []
    i = 0
    for items in name_list:
        
        curr_name = name_list[i]
        print(curr_name)
        MCC_true = [1.0]*5
        MCC_pred = (EM_scores[i], GEN_accuracy[i], SENS_accuracy[i], SPECIFIC_accuracy[i], DIV_scores[i])
        mean_metric = 1 - np.mean(MCC_pred)
        L1_metric = nn.L1Loss()
        MSE_metric = nn.MSELoss()
        KL_metric = nn.KLDivLoss()
        best = torch.ones(len(MCC_pred))
        
        l1_dist = L1_metric(best,torch.tensor(MCC_pred))
        
        MSE_dist = MSE_metric(best,torch.tensor(MCC_pred))
        KL_dist = KL_metric(best,torch.tensor(MCC_pred))
        auc = 0
        j = 0
        for items in MCC_pred:
            if(j == 0):
                auc += 0.5 * MCC_pred[0]*MCC_pred[-1]*np.sin(360/5)
            else:
                auc += 0.5 * MCC_pred[j]*MCC_pred[j-1]*np.sin(360/5)
            j += 1
        name_list[i] = curr_name + str(round(auc,2))
        auc_list.append((round(auc,2)))
        
        l1_list.append(np.around(l1_dist.numpy(),4))
        mean_dist_list.append(np.around(mean_metric,4))
        kl_dist_list.append(np.around(KL_dist.numpy(),4))
        mse_loss_list.append(np.around(MSE_dist.numpy(), 4))
        i +=1 
    
    
    
    give_DF = pd.DataFrame({'Methods': name_list,
                   '$EM$': EM_scores,
                   '$RA$': GEN_accuracy,
                   '$RE$': SENS_accuracy,
                   '$TA$': SPECIFIC_accuracy,
                   '$DI$': DIV_scores
                   })
    if(avg==0):
        overall_DF = pd.DataFrame({'Methods': name_list,
                       '$EM$': EM_scores,
                       '$RA$': GEN_accuracy,
                       '$RE$': SENS_accuracy,
                       '$TA$': SPECIFIC_accuracy,
                       '$DI$': DIV_scores,
                       'Mean Distance': mean_dist_list,
                       'L1Loss': l1_list,
                       'MSELoss': mse_loss_list,
                       'KLLoss': kl_dist_list,
                       'auc': auc_list           
        
                        })
    else:
        overall_DF = pd.DataFrame({
                       '$EM$': EM_scores,
                       '$RA$': GEN_accuracy,
                       '$RE$': SENS_accuracy,
                       '$TA$': SPECIFIC_accuracy,
                       '$DI$': DIV_scores,
                       'Mean Distance': mean_dist_list,
                       'L1Loss': l1_list,
                       'MSELoss': mse_loss_list,
                       'KLLoss': kl_dist_list,
                       'auc': auc_list           
        
                        })
    overall_DF = overall_DF.round(4)
    give_DF.iloc[:,1:] = (give_DF.iloc[:,1:].abs())
    give_DF.to_csv(disc_list_name+"_TIST.csv", index=False)
    
    overall_DF.iloc[:,1:] = (overall_DF.iloc[:,1:].abs())
    overall_DF.to_csv(disc_list_name+"_overall_TIST.csv", index=False)
    make_radar_chart(give_DF, disc_list_name, disc_list_name)
    if(avg == 0):
        return(1-EM_scores[0])
    else:
        return(overall_DF.to_numpy())

