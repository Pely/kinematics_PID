import re, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

paths = [
    "pid_data_Kr86_E2.5_Sr88.npz",
    "pid_data_Kr86_E2.5_Sr89.npz",
    "pid_data_Kr86_E2.7_Sr88.npz",
    "pid_data_Kr86_E2.7_Sr89.npz",
    "pid_data_Kr86_E3.0_Sr88.npz",
    "pid_data_Kr86_E3.0_Sr89.npz",
]

def parse_meta(p):
    m = re.search(r"_E([0-9.]+)_Sr(\d+)\.npz$", os.path.basename(p))
    return (float(m.group(1)), int(m.group(2))) if m else (None, None)

loaded = {p: dict(np.load(p, allow_pickle=True)) for p in paths}

# group by energy in filename
groups = defaultdict(dict)
for p in paths:
    E, Sr = parse_meta(p)
    groups[E][Sr] = p

layers = [
    ("front",  "E_front",  "dE_front"),
    ("middle", "E_middle", "dE_middle"),
    ("back",   "E_back",   "dE_back"),
]

# isotope styling (different marker + color)
iso_style = {
    89: dict(marker="D", color="C1"),
    88: dict(marker="o", color="C0"),
}

for E in sorted(groups.keys()):
    fig = plt.figure(figsize=(7, 6), constrained_layout=False)
    ax = fig.add_subplot(111)
    ax.set_title(f"PID (dE vs E) at E={E} MeV/u  |  Kr86  |  Sr88 vs Sr89")

    for Sr in [89, 88]:
        p = groups[E].get(Sr, None)
        if p is None:
            continue
        d = loaded[p]

        for lname, Ek, dEk in layers:
            x = np.asarray(d[Ek])
            y = np.asarray(d[dEk])
            n = min(len(x), len(y))
            ax.scatter(
                x[:n], y[:n],
                s=12, alpha=0.65,
                label=f"Sr{Sr}",
                marker=iso_style[Sr]["marker"],
                color=iso_style[Sr]["color"],
            )

    ax.set_xlabel("Energy Si (MeV)")
    ax.set_ylabel("IC dE (MeV)")
    ax.set_xlim(50, 180)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    plt.show()
