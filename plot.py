import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

MODELS  = ['GCN', 'TGN', 'TGAT', 'TGCN']
PRIMARY = ['GCN', 'TGN']

POISON_ATTACKS  = ['random', 'heterophilic']
EVASION_ATTACKS = ['evasion_random', 'evasion_heterophilic', 'evasion_hub_cascade']

POISON_BAR_BUDGET  = 33_700   # ~10% of train (337,224 edges)
EVASION_BAR_BUDGET = 3_000    # ~5%  of test  (57,851 edges)

POISON_SWEEP_BUDGETS  = [16_800, 33_700, 50_000]
EVASION_SWEEP_BUDGETS = [1_000, 3_000, 5_800]

MODEL_COLORS = {
    'GCN':  '#4878CF',
    'TGN':  '#E24A33',
    'TGAT': '#56A64B',
    'TGCN': '#988ED5',
}

ATTACK_COLORS = {
    'none':                 '#888888',
    'random':               '#E24A33',
    'heterophilic':         '#4878CF',
    'evasion_random':       '#E24A33',
    'evasion_heterophilic': '#4878CF',
    'evasion_hub_cascade':  '#56A64B',
}

ATTACK_LABELS = {
    'none':                 'Clean',
    'random':               'Random',
    'heterophilic':         'Heterophilic',
    'evasion_random':       'Random',
    'evasion_heterophilic': 'Heterophilic',
    'evasion_hub_cascade':  'Hub Cascade',
}

ATTACK_LINE = {
    'none':                 ('solid',   None),
    'random':               ('dashed',  'o'),
    'heterophilic':         ('dotted',  's'),
    'evasion_random':       ('dashed',  'o'),
    'evasion_heterophilic': ('dotted',  's'),
    'evasion_hub_cascade':  ('dashdot', '^'),
}

def rkey(model, attack, budget):
    return f"{model}__{attack}__{budget}"


def load(path):
    if not os.path.exists(path):
        print(f"[warn] {path} not found — affected plots will be empty")
        return {}
    with open(path) as f:
        return json.load(f)


def get_ndcg(results, model, attack, budget):
    return results.get(rkey(model, attack, budget), {}).get('test_ndcg')


def get_val_curve(results, model, attack, budget):
    raw = results.get(rkey(model, attack, budget), {}).get('val_by_epoch', {})
    if not raw:
        return [], []
    epochs = sorted(int(k) for k in raw)
    vals   = [raw[str(e)] for e in epochs if str(e) in raw]
    return epochs[:len(vals)], vals


def style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)


def dedup_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, **kwargs)

def plot_bar(results, attacks, bar_budget, title, out_path):
    all_attacks = ['none'] + attacks
    n_atk = len(all_attacks)
    n_mod = len(MODELS)

    width   = 0.75 / n_atk
    offsets = np.linspace(-(n_atk - 1) * width / 2,
                           (n_atk - 1) * width / 2, n_atk)
    x = np.arange(n_mod)

    fig, ax = plt.subplots(figsize=(max(7, n_mod * 2.2), 5))

    for j, attack in enumerate(all_attacks):
        budget = 0 if attack == 'none' else bar_budget
        color  = ATTACK_COLORS.get(attack, '#aaaaaa')
        label  = ATTACK_LABELS.get(attack, attack)

        for i, model in enumerate(MODELS):
            val   = get_ndcg(results, model, attack, budget) or 0
            alpha = 1.0 if model in PRIMARY else 0.5
            bx    = x[i] + offsets[j]
            ax.bar(bx, val, width * 0.88,
                   color=color, alpha=alpha,
                   edgecolor='white', linewidth=0.5,
                   label=label if i == 0 else '_nolegend_')
            if val:
                ax.text(bx, val + 0.004, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=7, rotation=90)

    # Per-model clean baseline tick mark
    for i, model in enumerate(MODELS):
        baseline = get_ndcg(results, model, 'none', 0)
        if baseline:
            ax.plot([x[i] - 0.38, x[i] + 0.38], [baseline, baseline],
                    color=MODEL_COLORS[model], linestyle='--',
                    linewidth=1.2, alpha=0.55, zorder=5)

    # Lightly shade secondary model columns
    for xi in x[len(PRIMARY):]:
        ax.axvspan(xi - 0.42, xi + 0.42, color='gray', alpha=0.05, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=12)
    ax.set_ylabel('Test NDCG@10', fontsize=11)
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_ylim(0, (ax.get_ylim()[1] or 1.0) * 1.22)
    dedup_legend(ax, title='Attack', fontsize=9, title_fontsize=9,
                 loc='upper right', framealpha=0.9)
    style_ax(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[plot] {out_path}")


def plot_epoch_curves(poison_results, out_path):
    n   = len(MODELS)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    experiments = [('none', 0)] + [(a, POISON_BAR_BUDGET) for a in POISON_ATTACKS]

    for ax, model in zip(axes, MODELS):
        has_data = False
        for attack, budget in experiments:
            epochs, vals = get_val_curve(poison_results, model, attack, budget)
            if not epochs:
                continue
            has_data = True
            ls, mk = ATTACK_LINE.get(attack, ('solid', None))
            step = max(1, len(epochs) // 10)
            ax.plot(epochs, vals,
                    linestyle=ls,
                    marker=mk, markevery=step, markersize=4,
                    color=ATTACK_COLORS.get(attack, '#aaaaaa'),
                    linewidth=2.2 if attack == 'none' else 1.6,
                    alpha=1.0 if attack == 'none' else 0.85,
                    label=ATTACK_LABELS.get(attack, attack))

        if not has_data:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=10)

        ax.set_title(model, fontsize=13, fontweight='bold',
                     color=MODEL_COLORS[model])
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Validation NDCG@10', fontsize=10)
        ax.legend(fontsize=8, framealpha=0.9)
        style_ax(ax)

    fig.suptitle('Training Dynamics: Clean vs. Poisoning Attacks  (tgbn-trade)',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[plot] {out_path}")

def plot_budget_sweep(results, attacks, budgets, bar_budget,
                      xlabel, title, out_path):
    n   = len(MODELS)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, MODELS):
        # Horizontal clean baseline
        baseline = get_ndcg(results, model, 'none', 0)
        if baseline:
            ax.axhline(baseline,
                       color=ATTACK_COLORS['none'], linestyle='--',
                       linewidth=1.6, alpha=0.8,
                       label=ATTACK_LABELS['none'])

        for attack in attacks:
            xs, ys = [], []
            for b in budgets:
                val = get_ndcg(results, model, attack, b)
                if val is not None:
                    xs.append(b); ys.append(val)
            if not xs:
                continue
            ls, mk = ATTACK_LINE.get(attack, ('solid', 'o'))
            ax.plot(xs, ys,
                    linestyle=ls, marker=mk, markersize=5,
                    color=ATTACK_COLORS.get(attack, '#aaaaaa'),
                    linewidth=1.8,
                    label=ATTACK_LABELS.get(attack, attack))

        # Star marker at the bar-chart budget
        ax.axvline(bar_budget, color='gray', linestyle=':',
                   linewidth=0.9, alpha=0.5)

        ax.set_title(model, fontsize=13, fontweight='bold',
                     color=MODEL_COLORS[model])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Test NDCG@10', fontsize=10)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
        ax.tick_params(axis='x', labelrotation=25, labelsize=8)
        ax.legend(fontsize=8, framealpha=0.9)
        style_ax(ax)

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[plot] {out_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--poison-file',  default='results_poison.json')
    p.add_argument('--evasion-file', default='results_evasion.json')
    p.add_argument('--out-dir',      default='figures')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    poison  = load(args.poison_file)
    evasion = load(args.evasion_file)

    def out(name):
        return os.path.join(args.out_dir, name)

    plot_bar(
        poison, POISON_ATTACKS, POISON_BAR_BUDGET,
        title=(f'Poisoning Attack Effectiveness — tgbn-trade\n'
               f'budget = {POISON_BAR_BUDGET:,} ≈ 10% of training set'
               f'   |   dashed lines = per-model clean baseline'
               f'   |   shaded = secondary models'),
        out_path=out('poison_bar.png'),
    )

    plot_bar(
        evasion, EVASION_ATTACKS, EVASION_BAR_BUDGET,
        title=(f'Evasion Attack Effectiveness — tgbn-trade\n'
               f'budget = {EVASION_BAR_BUDGET:,} ≈ 5% of test set'
               f'   |   dashed lines = per-model clean baseline'
               f'   |   shaded = secondary models'),
        out_path=out('evasion_bar.png'),
    )

    plot_epoch_curves(
        poison,
        out_path=out('epoch_curves.png'),
    )

    plot_budget_sweep(
        poison, POISON_ATTACKS,
        POISON_SWEEP_BUDGETS, POISON_BAR_BUDGET,
        xlabel='Budget (edges into training set)',
        title='Poisoning — NDCG@10 vs. Attack Budget   (tgbn-trade)',
        out_path=out('poison_budget.png'),
    )

    plot_budget_sweep(
        evasion, EVASION_ATTACKS,
        EVASION_SWEEP_BUDGETS, EVASION_BAR_BUDGET,
        xlabel='Budget (edges into test set)',
        title='Evasion — NDCG@10 vs. Attack Budget   (tgbn-trade)',
        out_path=out('evasion_budget.png'),
    )

    print('\nAll figures written to:', args.out_dir)