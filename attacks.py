import random

import torch

def _get_labels(data):
    """Return (labeled_nids, integer class labels)."""
    labeled_nids = data.node_y_nids
    labels = data.node_y
    if labels.dim() > 1:
        labels = labels.argmax(dim=1)
    else:
        labels = labels.long()
    return labeled_nids, labels


def _build_class_maps(labeled_nids, labels):
    """Return (node_to_class dict, unique_classes list, class_to_nodes dict)."""
    node_to_class = {labeled_nids[i].item(): labels[i].item()
                     for i in range(len(labeled_nids))}
    unique_classes = sorted(set(labels.tolist()))
    class_to_nodes = {c: labeled_nids[labels == c] for c in unique_classes}
    return node_to_class, unique_classes, class_to_nodes


def _compute_class_centroids(data, labeled_nids, labels, train_end_year=2009):
    """
    Compute mean edge feature vector per class.
    """
    if data.edge_x is None:
        return {}

    node_to_class, unique_classes, _ = _build_class_maps(labeled_nids, labels)
    feat_dim = data.edge_x.shape[1]

    edge_times = data.time[data.edge_mask]
    train_mask = edge_times < train_end_year
    train_src  = data.edge_index[train_mask, 0]
    train_feat = data.edge_x[train_mask]

    sums   = {c: torch.zeros(feat_dim) for c in unique_classes}
    counts = {c: 0 for c in unique_classes}
    for i in range(len(train_src)):
        src = train_src[i].item()
        if src in node_to_class:
            c = node_to_class[src]
            sums[c]   += train_feat[i]
            counts[c] += 1

    return {c: sums[c] / max(1, counts[c])
            for c in unique_classes if counts[c] > 0}

def _inject_edges(data, fake_src, fake_dst, fake_t, fake_feat=None):
    """
    Append edges to DGData's unified event log.
    """
    n_existing = len(data.time)
    n_inject   = len(fake_t)
    new_idx    = torch.arange(n_existing, n_existing + n_inject,
                              dtype=data.edge_mask.dtype)

    data.time       = torch.cat([data.time, fake_t.to(data.time.dtype)])
    data.edge_mask  = torch.cat([data.edge_mask, new_idx])
    new_rows        = torch.stack([fake_src.to(data.edge_index.dtype),
                                   fake_dst.to(data.edge_index.dtype)], dim=1)
    data.edge_index = torch.cat([data.edge_index, new_rows], dim=0)

    if data.edge_x is not None:
        if fake_feat is None:
            fake_feat = data.edge_x.mean(dim=0, keepdim=True).repeat(n_inject, 1)
        data.edge_x = torch.cat([data.edge_x, fake_feat], dim=0)

    return data


# POISONING ATTACKS

def random_edge_attack(data, budget: int, train_end_year: int = 2009):
    """
    Inject `budget` random edges uniformly across the training window.
    """
    t_min    = int(data.time.min().item())
    fake_src = torch.randint(0, data.num_nodes, (budget,))
    fake_dst = torch.randint(0, data.num_nodes, (budget,))
    fake_t   = torch.randint(t_min, train_end_year, (budget,))
    print(f"[random] Injected {budget} edges | t: {t_min}–{train_end_year}")
    return _inject_edges(data, fake_src, fake_dst, fake_t)


def heterophilic_attack(data, budget: int, train_end_year: int = 2009):
    """
    Inject cross-class edges concentrated in the final training years.
    """
    if data.node_y is None:
        print("[heterophilic] No labels — falling back to random")
        return random_edge_attack(data, budget, train_end_year)

    labeled_nids, labels = _get_labels(data)
    _, unique_classes, class_to_nodes = _build_class_maps(labeled_nids, labels)

    sl, dl, tl = [], [], []
    for _ in range(budget):
        c1, c2 = random.sample(unique_classes, 2)
        src = class_to_nodes[c1][torch.randint(len(class_to_nodes[c1]), (1,))].item()
        dst = class_to_nodes[c2][torch.randint(len(class_to_nodes[c2]), (1,))].item()
        t   = torch.randint(2008, train_end_year, (1,)).item()
        sl.append(src); dl.append(dst); tl.append(t)

    fake_src = torch.tensor(sl)
    fake_dst = torch.tensor(dl)
    fake_t   = torch.tensor(tl)
    print(f"[heterophilic] {budget} cross-class edges | t: 2008–{train_end_year}")
    return _inject_edges(data, fake_src, fake_dst, fake_t)

def hub_cascade_attack(data, budget: int, train_end_year: int = 2009,
                       top_k: int = 20):
    """
    Corrupt highest-degree hub nodes in the training window.
    """
    deg = torch.zeros(data.num_nodes, dtype=torch.long)
    for s in data.edge_index[:, 0]:
        deg[s] += 1
    _, hubs = deg.topk(min(top_k, data.num_nodes))

    t_min = int(data.time.min().item())
    eph   = budget // len(hubs)
    rem   = budget - eph * len(hubs)
    sl, dl, tl = [], [], []

    for i, hub in enumerate(hubs):
        n = eph + (1 if i < rem else 0)
        for _ in range(n):
            sl.append(hub.item())
            dl.append(torch.randint(0, data.num_nodes, (1,)).item())
            tl.append(torch.randint(t_min, train_end_year, (1,)).item())

    fake_src = torch.tensor(sl)
    fake_dst = torch.tensor(dl)
    fake_t   = torch.tensor(tl, dtype=data.time.dtype)
    print(f"[hub_cascade] {len(fake_src)} edges on {len(hubs)} hubs | "
          f"top-deg={deg[hubs[0]].item()}")
    return _inject_edges(data, fake_src, fake_dst, fake_t)

# EVASION ATTACKS

def _inject_edges_sorted(data, fake_src, fake_dst, fake_t, fake_feat=None):
    """
    Inject edges and re-sort chronologically.
    """
    data = _inject_edges(data, fake_src, fake_dst, fake_t, fake_feat)
    order           = data.time[data.edge_mask].argsort(stable=True)
    data.edge_mask  = data.edge_mask[order]
    data.edge_index = data.edge_index[order]
    if data.edge_x is not None:
        data.edge_x = data.edge_x[order]
    return data


def evasion_random_attack(test_data, budget: int, **_):
    """Inject random edges uniformly across the test window."""
    t0, t1   = float(test_data.time.min()), float(test_data.time.max())
    fake_src = torch.randint(0, test_data.num_nodes, (budget,))
    fake_dst = torch.randint(0, test_data.num_nodes, (budget,))
    fake_t   = torch.empty(budget).uniform_(t0, t1)
    print(f"[evasion_random] {budget} edges | t: {t0:.0f}–{t1:.0f}")
    return _inject_edges_sorted(test_data, fake_src, fake_dst, fake_t)


def evasion_hub_cascade_attack(test_data, budget: int, full_data=None,
                                top_k: int = 20, **_):
    """
    Corrupt highest-degree hub nodes at test time.
    """
    src_data = full_data if full_data is not None else test_data
    deg      = torch.zeros(src_data.num_nodes, dtype=torch.long)
    for s in src_data.edge_index[:, 0]:
        deg[s] += 1
    _, hubs = deg.topk(min(top_k, src_data.num_nodes))

    t0, t1 = float(test_data.time.min()), float(test_data.time.max())
    t_end  = t0 + (t1 - t0) * 0.10
    eph    = budget // len(hubs)
    rem    = budget - eph * len(hubs)
    sl, dl, tl = [], [], []

    for i, hub in enumerate(hubs):
        for _ in range(eph + (1 if i < rem else 0)):
            sl.append(hub.item())
            dl.append(torch.randint(0, test_data.num_nodes, (1,)).item())
            tl.append(t0 + torch.rand(1).item() * (t_end - t0))

    fake_src = torch.tensor(sl)
    fake_dst = torch.tensor(dl)
    fake_t   = torch.tensor(tl, dtype=test_data.time.dtype)
    print(f"[evasion_hub_cascade] {len(fake_src)} edges | {len(hubs)} hubs | "
          f"top-deg={deg[hubs[0]].item()}")
    return _inject_edges_sorted(test_data, fake_src, fake_dst, fake_t)


def evasion_historical_attack(test_data, budget: int, full_data=None,
                               val_end: float = 2014.0, **_):
    """
    Inject edges that existed in training but are absent from the test set.
    """
    if full_data is None:
        print("[evasion_historical] No full_data — falling back to random")
        return evasion_random_attack(test_data, budget)

    ft       = full_data.time[full_data.edge_mask]
    tr_edges = set(map(tuple, full_data.edge_index[ft < val_end].tolist()))
    te_edges = set(map(tuple, test_data.edge_index.tolist()))
    hist_neg = list(tr_edges - te_edges)

    if not hist_neg:
        print("[evasion_historical] No historical negatives — falling back to random")
        return evasion_random_attack(test_data, budget)

    samp = random.sample(hist_neg, min(budget, len(hist_neg)))
    if len(samp) < budget:
        samp += [hist_neg[random.randint(0, len(hist_neg) - 1)]
                 for _ in range(budget - len(samp))]

    t0, t1   = float(test_data.time.min()), float(test_data.time.max())
    fake_src = torch.tensor([e[0] for e in samp])
    fake_dst = torch.tensor([e[1] for e in samp])
    fake_t   = torch.empty(budget).uniform_(t0, t0 + (t1 - t0) * 0.5).to(test_data.time.dtype)
    print(f"[evasion_historical] {budget} historical negatives")
    return _inject_edges_sorted(test_data, fake_src, fake_dst, fake_t)


def evasion_heterophilic_attack(test_data, budget: int, full_data=None, **_):
    """Cross-class edge injection at test time."""
    src_data = full_data if full_data is not None else test_data
    if src_data.node_y is None:
        print("[evasion_heterophilic] No labels — falling back to random")
        return evasion_random_attack(test_data, budget)

    labeled_nids, labels = _get_labels(src_data)
    _, unique_classes, class_to_nodes = _build_class_maps(labeled_nids, labels)

    t0, t1 = float(test_data.time.min()), float(test_data.time.max())
    sl, dl, tl = [], [], []
    for _ in range(budget):
        c1, c2 = random.sample(unique_classes, 2)
        sl.append(class_to_nodes[c1][torch.randint(len(class_to_nodes[c1]), (1,))].item())
        dl.append(class_to_nodes[c2][torch.randint(len(class_to_nodes[c2]), (1,))].item())
        tl.append(t0 + torch.rand(1).item() * (t1 - t0))

    fake_src = torch.tensor(sl)
    fake_dst = torch.tensor(dl)
    fake_t   = torch.tensor(tl, dtype=test_data.time.dtype)
    print(f"[evasion_heterophilic] {budget} cross-class edges")
    return _inject_edges_sorted(test_data, fake_src, fake_dst, fake_t)

# =============================================================================
# UNIFIED DISPATCHER
# =============================================================================

POISON_ATTACKS = {
    'random':                random_edge_attack,
    'heterophilic':          heterophilic_attack,
    'hub_cascade':           hub_cascade_attack,
}

EVASION_ATTACKS = {
    'evasion_random':        evasion_random_attack,
    'evasion_hub_cascade':   evasion_hub_cascade_attack,
    'evasion_historical':    evasion_historical_attack,
    'evasion_heterophilic':  evasion_heterophilic_attack,
}


def apply_attack(data, attack_name: str, budget: int, full_data=None):
    """
    Apply a named attack to `data`.
    """
    if attack_name == 'none':
        return data

    if attack_name in POISON_ATTACKS:
        return POISON_ATTACKS[attack_name](data, budget)

    if attack_name in EVASION_ATTACKS:
        return EVASION_ATTACKS[attack_name](data, budget, full_data=full_data)

    raise ValueError(
        f"Unknown attack: '{attack_name}'.\n"
        f"  Poisoning: {sorted(POISON_ATTACKS)}\n"
        f"  Evasion:   {sorted(EVASION_ATTACKS)}"
    )