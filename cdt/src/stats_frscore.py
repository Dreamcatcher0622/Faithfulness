import argparse, os
import pickle

    

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import FancyBboxPatch


def normalize(datas):
    s = sum(datas)
    if s == 0:
        return datas
    return [k/s for k in datas]

def draw(datas, path):

    ret = []

    for score in [100 , 0]:
        samples_number = 0
        scores = [0,0,0,0]
        for data in datas:
            if data['score']!=score:continue
            samples_number += 1

            supporting_pos = data['evidence_pos']
            interference_pos = data['attack_pos']
            lowfrequence_pos = data['emoji_pos']

            supporting_length = sum(x[1]-x[0] for x in supporting_pos)
            interference_length = sum(x[1]-x[0] for x in interference_pos)
            irrelevant_length = data['irr_length'] 
            lowfrequence_length = sum(x[1]-x[0] for x in lowfrequence_pos)
            
            lengths = [supporting_length, interference_length, irrelevant_length, lowfrequence_length]

            for l in range(28):
                for i in range(len(scores)):
                    scores[i] += data['attention'][l]['weight']['score'][i] * lengths[i]
            
            for i in range(len(scores)):
                scores[i]/=8
        
        for i in range(len(scores)):
            scores[i]/=max(1,samples_number)

        ret += [scores]

    data = pd.DataFrame({
        'Context Type': ['System\nInstruction', 'Raw\nExperience', 'Condensed\nExperience', 'Current\nTrajectory'],
        'correct': normalize(ret[0]),
        'wrong': normalize(ret[1]),
    })



    data_melted = data.melt(id_vars='Context Type', var_name='Predicted Result', value_name='Mean Weight')

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Context Type', y='Mean Weight', hue='Predicted Result', data=data_melted, width = 0.6)


    plt.title('FR score', fontsize=28)
    plt.ylabel('score value', fontsize=28)



    for p in bar_plot.patches:
        x = p.get_height()
        if x==0:x=""
        else: x = f"{x:.2f}" 
        bar_plot.annotate(x, (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')

    plt.xticks(ticks = [0,1,2,3], labels = data['Context Type'] ,fontsize= 22)
    plt.xlabel('')

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True)) 
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))


    plt.legend(title = '', loc = 'upper left', prop={'size':28})
    plt.tight_layout()

    print("save:",path)
    plt.savefig(path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--context_length', type=int, default=7900)
    parser.add_argument('--result_dir', type=str, default='../results/expel/last_turn/expel/Qwen3-1.7B')
    parser.add_argument('--model_tag',type = str, default='Qwen3-1.7B')

    args = parser.parse_args()
    results_dir = args.result_dir
    results_dir = f"../results/expel/first_turn/expel_first/Qwen3-1.7B/layer_0"
    results_dir = "../results/expel/last_turn/expel_2/Qwen3-1.7B/layer_0"

    file_paths = [os.path.join(results_dir,k) for k in os.listdir(results_dir)]

    file_datas = [pickle.load(open(k,'rb')) for k in file_paths]

    file_name = '-'.join(results_dir.split('/')[2:])
    draw(file_datas, f"../results/figures/FR-{file_name}.pdf")



