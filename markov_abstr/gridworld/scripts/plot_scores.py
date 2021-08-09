import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from visgrid.utils import load_experiment

def load_experiment(tag):
    logfiles = sorted(glob.glob(os.path.join('results/scores', tag + '*', 'scores-*.txt')))
    agents = [f.split('-')[-2] for f in logfiles]
    seeds = [int(f.split('-')[-1].split('.')[0]) for f in logfiles]
    logs = [open(f, 'r').read().splitlines() for f in logfiles]

    def read_log(log):
        results = [json.loads(item) for item in log]
        data = smooth(pd.DataFrame(results), 10)
        return data

    results = [read_log(log) for log in logs]
    keys = list(zip(agents, seeds))
    data = pd.concat(results, join='outer', keys=keys,
                     names=['agent',
                            'seed']).sort_values(by='seed',
                                                 kind='mergesort').reset_index(level=[0, 1])
    return data  #[data['episode']<=100]

def smooth(data, n):
    numeric_dtypes = data.dtypes.apply(pd.api.types.is_numeric_dtype)
    numeric_cols = numeric_dtypes.index[numeric_dtypes]
    data[numeric_cols] = data[numeric_cols].rolling(n).mean()
    return data

labels = ['tag']
experiments = [
    ('test-foo'),
]
data = pd.concat([load_experiment(e[0]) for e in experiments],
                 join='outer',
                 keys=experiments,
                 names=labels).reset_index(level=list(range(len(labels))))

# plt.rcParams.update({'font.size': 10})
p = sns.color_palette(n_colors=1)
p = sns.color_palette('Set1', n_colors=9, desat=0.5)
red, blue, green, purple, orange, yellow, brown, pink, gray = p
p = [blue]
g = sns.relplot(
    x='total_episodes',
    y='reward',
    kind='line',
    data=data,
    height=4,
    alpha=1,
    # hue='features',
    # style='features',
    # units='seed', estimator=None,
    # col='grid_type',  #col_wrap=2,
    # legend=False,
    facet_kws={
        'sharey': True,
        'sharex': False
    },
    palette=p,
)
# plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.15)
g.fig.suptitle('Reward vs. Time')
# plt.rcParams.update({'font.size': 22})
plt.savefig('results/foo.png')
plt.show()
