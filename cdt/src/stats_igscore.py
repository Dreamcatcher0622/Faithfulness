import argparse, os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
    

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

def test(datas):
    print(datas[1])

def draw(datas, layer, path):
    ret = []
    for score in [100 , 0]:
        samples_number = 0
        scores = [0,0,0,0]
        for data in datas:
            # print(data['score'])
            if data['score'] != score:continue
            samples_number += 1
            
            for i in range(len(scores)):
                if not np.isnan(data['embedding'][layer]['grad']['score'][i]):
                    scores[i] += data['embedding'][layer]['grad']['score'][i]
        
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

    plt.title(f'IG score', fontsize=28)
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


    plt.legend(title = '', loc = 'upper right', prop={'size':28})
    plt.tight_layout()

    print("save:",path)
    plt.savefig(path)

def draw_all(datas, file_name, end = 200):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for num, data in enumerate(datas):
        if num > end:
            break
        x_data = [i for i in range(28)]
        y_data = [[], [], [], []]
        plt.figure(figsize=(9, 6))

        for layer in range(28):
            for i in range(4):
                if not np.isnan(data['embedding'][layer]['grad']['score'][i]):
                    y_data[i].append(data['embedding'][layer]['grad']['score'][i])      

        for idx, field in enumerate(FIELDS):
            plt.plot(x_data, y_data[idx], 
                    label=field, 
                    marker=markers[idx],
                    color=colors[idx],
                    linewidth=2,
                    markersize=6)

        plt.xlabel('Layers', fontsize=12, fontweight='bold')
        plt.ylabel('Normalized IG Score', fontsize=12, fontweight='bold')
        plt.title(f'IG Score - {num}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        file_path = f"../results/figures/IG-ALL/{file_name}-{num}.pdf"

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已保存: {file_path}")

def draw_all_average(datas, file_name, model, end = 200):
    colors = ['#f4b184', '#a9d08d', '#9dc3e7', '#b1a6ce']
    markers = ['o', 's', '^', 'D']
    layers = 28
    if 'Qwen3-4B' in model or '8B' in model:
        layers = 36
    elif '14B' in model:
        layers = 40
    elif '32B' in model:
        layers = 64

    total = 0
    y_data = [[], [], [], []]
    for num, data in enumerate(datas):
        total += 1
        if num > end:
            break
        for layer in range(layers):
            for i in range(4):
                if not np.isnan(data['embedding'][layer]['grad']['score'][i]):
                    if num == 0:
                        y_data[i].append(data['embedding'][layer]['grad']['score'][i])
                    else:
                        # print(layer)
                        y_data[i][layer] += data['embedding'][layer]['grad']['score'][i]

    print(f"Total number:{total}")
    for layer in range(layers):
        for i in range(4):
            y_data[i][layer] /= total

    # plt.figure(figsize=(9, 6))
    # for idx, field in enumerate(FIELDS):
    #     plt.plot(x_data, y_data[idx], 
    #             label=field, 
    #             marker=markers[idx],
    #             color=colors[idx],
    #             linewidth=2,
    #             markersize=6)

    # plt.xlabel('Layers', fontsize=12, fontweight='bold')
    # plt.ylabel('Normalized IG Score', fontsize=12, fontweight='bold')
    # plt.title(f'IG Score - AVG', fontsize=14, fontweight='bold')
    # plt.legend(loc='best', fontsize=10)
    # plt.grid(True, alpha=0.3, linestyle='--')
    # plt.tight_layout()

    mpl.rcParams.update({
        "font.size": 14,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    })

    FIELDS = ["System Instruction", "Raw Experience", "Condensed Experience", "Current Trajectory"]
    methods = {
        "System Instruction": (y_data[0],  "#f4b184", "o", "-"),
        "Raw Experience": (y_data[1], "#a9d08d", "v", "-"),
        "Condensed Experience": (y_data[2], "#9dc3e7", "d", "-"),
        "Current Trajectory": (y_data[3], "#b1a6ce", "s", "-"),
        # "RAG(Top-5)": ([7, 18, 49, 60, 63, 44, 55, 60, 30, 22], "#7ad1d2", "D", "-."),
        # "DPO": ([10, 15, 42, 70, 73, 25, 55, 53, 47, 50], "#c8b3f2", "P",  "-"),
        # "SFT": ([9, 14, 44, 65, 62, 25, 55, 60, 55, 45], "#98c4e8", "s", "-"),
        # "Qwen-RLPA": ([64, 68, 75, 72, 76, 70, 78, 79, 84, 78], "#2ca25f", "o", "-"),
    }
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # 显示所有四条边框（实线）
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    x_data = [i for i in range(len(y_data[0]))]
    for name, (y, color, marker, ls) in methods.items():
        marker_indices = list(range(0, len(y), 2))
        if len(y) % 2 == 0:
            marker_indices.append(len(y) - 1)
        plt.plot(
            x_data, y,
            label=name,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.3,
            markersize=6,
            alpha=0.9,
            markevery=marker_indices
        )
    leg = plt.legend(
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=True,
        fontsize=13
    )

    leg.get_frame().set_linewidth(2)   # 边框线宽
    # leg.get_frame().set_edgecolor('black')  # 边框颜色
    # leg.get_frame().set_alpha(1.0)  # 背景不透明

    # 设置边框的粗细
    spines = ax.spines
    # spines['top'].set_linewidth(2)    # 上边框粗细
    spines['right'].set_linewidth(2.5)  # 右边框粗细

    plt.xlabel("Layers", fontsize=16)
    plt.ylabel("Normalized IG Score", fontsize=16)

    # # 网格（更柔和的呈现）
    # plt.grid(alpha=0.25, linestyle="--")

    # Y 轴范围（可按需调整）
    plt.ylim(0, 0.8)

    plt.tight_layout()
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    file_path = f"../results/figures/IG-AVG-{file_name}.pdf"

    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    print(f"已保存: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--context_length', type=int, default=900)
    parser.add_argument('--result_dir', type=str, default='../results/expel/last_turn/expel/Qwen3-1.7B')
    parser.add_argument('--model_tag',type = str, default='Qwen3-1.7B')
    parser.add_argument('--layer',type = int, default=0)
    parser.add_argument('--all_layer',type = str, default='True')

    args = parser.parse_args()
    
    results_dir = args.result_dir
    models = ['Qwen3-32B', 'Qwen3-14B', 'Qwen3-8B', 'Qwen3-4B', 'Qwen3-1.7B']
    types = ['expel', 'expel_insights_empty', 'expel_insights_corrupted', 'expel_insights_irrelevant', 'expel_insights_filler_tokens', 'expel_irrelevant_trajectory', 'expel_shuffle_trajectory']
    # types = ['expel_insights_empty']
    for model in models:
        for type in types:
            results_dir = f"../results/expel/{model}/{type}/{model}/layer_0"
            if os.path.exists(results_dir):
                file_paths = [os.path.join(results_dir,k) for k in os.listdir(results_dir)]
                file_datas = [pickle.load(open(k,'rb')) for k in file_paths]
                file_name = '-'.join(results_dir.split('/')[2:-1])

                if args.all_layer == 'False':
                    draw(file_datas, layer = args.layer, path = f"../results/figures/IG-{file_name}.png")
                else:
                    # draw_all(file_datas, file_name = file_name)
                    draw_all_average(file_datas, file_name = file_name, model = model)

    model = 'Qwen3-1.7B'
    for turn in ['first', 'last']:
        for type in types:
            if turn == 'first':
                type = type + '_first'
                results_dir = f"../results/expel/first_turn/{type}/{model}/layer_0"
            else:
                results_dir = f"../results/expel/last_turn/{type}/{model}/layer_0"
            if os.path.exists(results_dir):
                file_paths = [os.path.join(results_dir,k) for k in os.listdir(results_dir)]
                file_datas = [pickle.load(open(k,'rb')) for k in file_paths]
                file_name = '-'.join(results_dir.split('/')[2:-1])

                if args.all_layer == 'False':
                    draw(file_datas, layer = args.layer, path = f"../results/figures/IG-{file_name}.png")
                else:
                    # draw_all(file_datas, file_name = file_name)
                    draw_all_average(file_datas, file_name = file_name, model = model)



