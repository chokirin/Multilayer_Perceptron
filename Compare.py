
#!/usr/bin/env python3
"""Bonus: display the learning curves of several runs on the same graph.
 
Each run of train.py saves a history file (--history). Compare them with:
 
    python train.py --optimizer sgd  --history sgd.json  --name sgd
    python train.py --optimizer adam --history adam.json --name adam
    python compare.py sgd.json adam.json
"""
import argparse
import json
 
import matplotlib
import matplotlib.pyplot as plt
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("histories", nargs="+", help="history .json files from train.py")
    ap.add_argument("--out", default="comparison.png")
    args = ap.parse_args()
 
    panels = [("val_loss", "Validation loss"), ("val_acc", "Validation accuracy"),
              ("val_f1", "Validation F1")]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4))
    for path in args.histories:
        with open(path) as f:
            run = json.load(f)
        for ax, (key, title) in zip(axes, panels):
            ax.plot(run["history"][key], label=run["name"])
            ax.set(title=title, xlabel="epoch")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"> comparison saved to {args.out}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
 
 
if __name__ == "__main__":
    main()
