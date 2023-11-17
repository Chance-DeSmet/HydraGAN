import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def gen_radar_chart(dat_loc, save_name='./new_radar.png'):
    df = pd.DataFrame({'Col A': ['home', 'other', 'used', 'new', 'service'],
                   'Col B': [1, 2, 3, 4, 5],
                   'Col C':[3, 3, 3, 3, 3],
                   })
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    theta = np.arange(len(df) + 1) / float(len(df)) * 2 * np.pi
    values = df['Col B'].values
    print("Values are", values)
    values = np.append(values, values[0])
    print("Values are", values)
    l1, = ax.plot(theta, values, color="C2", marker="o", label="Name of Col B")
    plt.xticks(theta[:-1], df['Col A'], color='grey', size=12)
    ax.fill(theta, values, 'green', alpha=0.1)
    values = df['Col C'].values
    values = np.append(values, values[0])
    l1, = ax.plot(theta, values, color="C2", marker="o", label="Name of Col C")
    ax.fill(theta, values, 'red', alpha=0.1)
    
    plt.legend()
    plt.title("Radar chart comparison")
    plt.savefig(save_name)
    
    
    
def make_radar_chart(df, save_name='./new_radar.png', name='put a data name here', k=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    theta = np.arange(df.shape[1]) / float(df.shape[1] - 1) * 2 * np.pi
    plt.xticks(theta[:-1], df.columns[1:], size=12)
    plt.yticks([ 0.40, 0.80])
    ax.tick_params(pad=10)
    i = 0
    while i < df.shape[0]:
        curr_col = df.iloc[i,0]
        values = df.iloc[i,1:]
        values = np.append(values, values[0])
        l1, = ax.plot(theta, values, label=curr_col)
        ax.fill(theta, values, alpha=0.1)
            
        i += 1
    plt.legend(loc='lower right', bbox_to_anchor=(0.025,0.25), fontsize='small')
    plt.title(name[:-6], x=0.5, y=1.07, fontdict = {'fontsize' : 12} )
    plt.savefig(save_name+'_'+str(k)+'_'+str(name[0])+'_'+'.png', dpi=1200)
    plt.close(fig)
