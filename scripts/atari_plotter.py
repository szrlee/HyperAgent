import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.pyplot import MultipleLocator

sns.set()
fontsize = 12
sns.set_style("whitegrid", {"grid.linestyle": "--"})


COLORS = {
    "Variational": "#8c564b",
    "LangevinMC": "#d62728",
    "Ensemble+": "#9467bd",
    "Rainbow": "#2ca02c",
    "HyperAgent": "#1f77b4",
}


def mean_confidence_interval(data, confidence=0.9):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def smooth(scalar, weight=0.6):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def gen_ydata(ys, min_len, weight):
    ymin_len = min([len(y) for y in ys])
    min_len = min(min_len, ymin_len)
    y_matrix = np.vstack([y[:min_len] for y in ys])
    y_mean, low_CI_bound, high_CI_bound = mean_confidence_interval(y_matrix)
    y_min = np.min(y_matrix, axis=0)
    y_max = np.max(y_matrix, axis=0)
    y_low = np.maximum(y_min, low_CI_bound)
    y_high = np.minimum(y_max, high_CI_bound)
    return smooth(y_mean, weight), smooth(y_low, weight), smooth(y_high, weight)


def plot_distribution(
    xs, ys, ax=None, set_x_label=False, set_y_label=False, weight=0.6
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_title(f"{game_name}", size=fontsize)
    if set_y_label:
        ax.set_ylabel("Episode Return", size=fontsize)
    if set_x_label:
        ax.set_xlabel("Num of Steps", size=fontsize)
    if len(xs) == 0:
        ax.set_xticks([])
        ax.set_yticks([])

    for label in COLORS.keys():
        if label not in xs.keys():
            continue
        x = xs[label]
        y = ys[label]
        min_len = min([len(x) for x in xs[label]])
        x = x[0][:min_len]
        y, y_low, y_high = gen_ydata(ys[label], min_len, weight)
        ax.plot(x, y, label=label, linewidth=2, c=COLORS[label])
        ax.fill_between(x, y_high, y_low, alpha=0.2 , color=COLORS[label])
    ax.set_xlim(0, 2e6)
    ax.set_xticks([0, 0.5e6, 1e6, 1.5e6, 2e6])
    ax.set_xticklabels(["0", "0.5M", "1.0M", "1.5M", "2M"])
    if game_name == "Alien":
        ax.set_ylim(top=2600)
    elif game_name == "Venture":
        ax.set_ylim(top=420)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")

data_path = "your_data_path"
LangevinAdam_data = pd.read_csv(os.path.join(data_path, "LangevinAdam_double.csv"), sep=",")
AdamLMCDQN_data = pd.read_csv(os.path.join(data_path, "AdamLMCDQN_double.csv"), sep=",")
Ensemble_data = pd.read_csv(os.path.join(data_path, "BootDQN_double.csv"), sep=",")
Rainbow_data = pd.read_csv(os.path.join(data_path, "rainbow.csv"), sep=",")
HyperAgent_data = pd.read_csv(os.path.join(data_path, "HyperAgent_8hard.csv"), sep=",")
Variational_data = pd.read_csv(os.path.join(data_path, "SANE_8hard.csv"), sep=",")

datas = {
    "Variational": Variational_data,
    "LangevinMC": AdamLMCDQN_data,
    "Ensemble+": Ensemble_data,
    "Rainbow": Rainbow_data,
    "HyperAgent": HyperAgent_data
}
game_names = [
    "Alien",
    "Freeway",
    "Gravitar",
    "Hero",
    "Pitfall",
    "Qbert",
    "Solaris",
    "Venture",
]
n_row, n_col = 2, 4
fig, axes = plt.subplots(n_row, n_col, figsize=(3 * n_col, 3 * n_row + 0.6))  # 3.6
fig.subplots_adjust(
    left=0.07, right=0.98, bottom=0.2, top=0.93, hspace=0.35, wspace=0.05
)
legend_id = -1
for i, game_name in enumerate(game_names): 
    row = i // 4
    col = i % 4
    sample_num, mean_reward = {}, {}

    for alg, data in datas.items():
        temp_data = data[data["environment_name"] == game_name.lower()]
        sample_num[alg], mean_reward[alg] = [], []
        temp_reward, temp_frame = [], []
        seeds = set(temp_data["seed"].to_numpy())
        for i in seeds:
            frame = temp_data[temp_data["seed"] == i]["frame"].to_numpy() // 4
            index = int(np.where(frame == 2e6)[0])
            frame = frame[:index+1]
            eval_return = temp_data[temp_data["seed"] == i][
                "eval_episode_return"
            ].to_numpy()[:index+1]
            temp_frame.append(frame)
            temp_reward.append(eval_return)
        sample_num[alg] = temp_frame
        mean_reward[alg] = temp_reward
    
    sample_num = dict(sorted(sample_num.items(), key=lambda x: x[0], reverse=True))
    mean_reward = dict(sorted(mean_reward.items(), key=lambda x: x[0], reverse=True))
    plot_distribution(
        sample_num,
        mean_reward,
        ax=axes[row][col],
        set_x_label=(row == 1),
        set_y_label=col == 0,
        weight=0.6,
    )

lines, labels = fig.axes[legend_id].get_legend_handles_labels()
fig.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.1),
    borderaxespad=0.0,
    ncol=5,
    fontsize=15,
)
plt.savefig("./atari_hard", bbox_inches="tight")
plt.savefig("./atari_hard.pdf", bbox_inches="tight")
