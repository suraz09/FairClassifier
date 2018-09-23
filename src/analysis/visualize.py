import numpy as np
np.random.seed(7)
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True, context="talk")

create_gif = False

"""
    Show the results in the grahph plot.
"""

def plot_distributions(y, Z, iteration=None, val_metrics=None, p_rules=None, fname=None):
    fig, axes = plt.subplots(1, 1, figsize=(10, 4), sharey=True)
    legend = {'race': ['African-American', 'Others']}
    for idx, attr in enumerate(Z.columns):
        for attr_val in [0, 1]:
            ax = sns.distplot(y[Z[attr] == attr_val], hist=False,
                              kde_kws={'shade': True, },
                              label='{}'.format(legend[attr][attr_val]))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 7)
        ax.set_yticks([])
        ax.set_title("sensitive attibute: {}".format(attr))
        if idx == 0:
            ax.set_ylabel('prediction distribution')
        ax.set_xlabel(r'$P({{risk>High}}|z_{{{}}})$'.format(attr))
    if iteration:
        fig.text(1.0, 0.9, f"Training iteration #{iteration}", fontsize='16')
    if val_metrics is not None:
        fig.text(1.0, 0.65, '\n'.join(["Prediction performance:",
                                       f"- ROC AUC: {val_metrics['ROC AUC']:.2f}",
                                       f"- Accuracy: {val_metrics['Accuracy']:.1f}"]),
                 fontsize='16')
    if p_rules is not None:
        fig.text(1.0, 0.4, '\n'.join(["Satisfied p%-rules:"] +
                                     [f"- {attr}: {p_rules[attr]:.0f}%-rule"
                                      for attr in p_rules.keys()]),
                 fontsize='16')
    fig.tight_layout()
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
    #     st.pyplot()
    return fig