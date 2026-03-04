import argparse
import json
import os
import re
import subprocess
import sys
import time

MODEL_SCRIPTS = {
    'GCN':  'gcn.py',
    'TGN':  'tgn.py',
    'TGAT': 'tgat.py',
    'TGCN': 'tgcn.py',
}

MODEL_EPOCHS = {
    'GCN':  20,
    'TGN':  20,
    'TGAT': 20,
    'TGCN': 20,
}

MODELS  = ['GCN', 'TGN', 'TGAT', 'TGCN']
ATTACKS = ['random', 'heterophilic']
SEEDS   = [1, 2, 3]

BUDGET_SWEEP_BUDGETS = [16_800, 33_700, 50_000]
BAR_BUDGET           = 33_700

BAR_EXPERIMENTS = (
    [(m, 'none', 0)       for m in MODELS]
    + [(m, a, BAR_BUDGET) for m in MODELS for a in ATTACKS]
)
SWEEP_EXPERIMENTS = [
    (m, a, b)
    for m in MODELS
    for a in ATTACKS
    for b in BUDGET_SWEEP_BUDGETS
]

def rkey(model, attack, budget, seed):
    return f"{model}__{attack}__{budget}__seed_{seed}"

def load_results(path):
    return json.load(open(path)) if os.path.exists(path) else {}

def save_results(results, path):
    json.dump(results, open(path, 'w'), indent=2)

def run_one(model, attack, budget, seed, device, epochs, script_dir):
    cmd = [
        sys.executable, os.path.join(script_dir, MODEL_SCRIPTS[model]),
        '--attack', attack,
        '--budget', str(budget),
        '--epochs', str(epochs),
        '--device', device,
        '--seed',   str(seed),
    ]
    print(f"\n{'─'*64}")
    print(f"  {model:5s} | {attack:20s} | budget={budget:>8,} | seed={seed} | epochs={epochs}")
    print(f"{'─'*64}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=7200, cwd=script_dir)
        out = proc.stdout + proc.stderr
        print(out[-3000:])
        m = parse_output(out)
        m['runtime_s'] = round(time.time() - t0)
        m['exit_code']  = proc.returncode
        return m
    except subprocess.TimeoutExpired:
        return {'error': 'timeout'}
    except Exception as e:
        return {'error': str(e)}


def parse_output(text):
    val_by_epoch = {}
    test_ndcg    = None
    for m in re.finditer(
        r'[Ee]poch[^\d]*(\d+)[^\n]*[Vv]alidation[^\n]*?(\d+\.\d+)', text
    ):
        val_by_epoch[m.group(1)] = float(m.group(2))
    for m in re.finditer(
        r'[Vv]alidation[^\n]*?(\d+\.\d+)[^\n]*[Ee]poch[^\d]*(\d+)', text
    ):
        val_by_epoch[m.group(2)] = float(m.group(1))
    for m in re.finditer(r'[Tt]est[^\d]*(\d+\.\d+)', text):
        v = float(m.group(1))
        if 0.0 < v < 2.0:
            test_ndcg = v
            break
    return {'test_ndcg': test_ndcg, 'val_by_epoch': val_by_epoch}

def all_experiments():
    seen, out = set(), []
    for exp in BAR_EXPERIMENTS + SWEEP_EXPERIMENTS:
        if exp not in seen:
            seen.add(exp)
            out.append(exp)
    return out

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-file', default='results_poison.json')
    p.add_argument('--device',       default='cpu')
    p.add_argument('--smoke-test',   action='store_true')
    args = p.parse_args()

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, args.results_file)
    results      = load_results(results_path)
    print(f"[cache] Loaded {len(results)} cached results from {results_path}")

    if args.smoke_test:
        experiments     = [('GCN', 'none', 0)]
        override_epochs = {'GCN': 2}
        seeds           = [1]
        print("[smoke-test] GCN clean, 2 epochs, seed 1.")
    else:
        experiments     = all_experiments()
        override_epochs = {}
        seeds           = SEEDS

    total = len(experiments) * len(seeds)
    idx   = 0
    for model, attack, budget in experiments:
        for seed in seeds:
            idx += 1
            key = rkey(model, attack, budget, seed)
            if key in results and results[key].get('test_ndcg') is not None:
                print(f"[{idx}/{total}] cached  — {key}")
                continue
            print(f"[{idx}/{total}] running — {key}")
            epochs  = override_epochs.get(model, MODEL_EPOCHS[model])
            metrics = run_one(model, attack, budget, seed, args.device, epochs, script_dir)
            results[key] = metrics
            save_results(results, results_path)
            print(f"  → ndcg={metrics.get('test_ndcg', '—')}  exit={metrics.get('exit_code', '—')}")

    # Summary table
    print(f"\nDone. Results saved to {results_path}")
    print(f"\n{'Model':<6} {'Attack':<22} {'Budget':>8}  "
          f"{'Mean NDCG':>10}  {'Std':>7}  {'Seeds done':>10}")
    print('─' * 68)
    for model, attack, budget in sorted(all_experiments()):
        vals = [results[rkey(model, attack, budget, s)]['test_ndcg']
                for s in SEEDS
                if rkey(model, attack, budget, s) in results
                and results[rkey(model, attack, budget, s)].get('test_ndcg')]
        if vals:
            import numpy as np
            mean_s = f"{float(np.mean(vals)):.4f}"
            std_s  = f"{float(np.std(vals)):.4f}"
        else:
            mean_s = std_s = '—'
        print(f"{model:<6} {attack:<22} {budget:>8,}  {mean_s:>10}  {std_s:>7}  {len(vals):>10}")