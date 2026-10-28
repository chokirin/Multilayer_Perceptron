#!/usr/bin/env python3
import argparse
import csv
 
import numpy as np
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ratio", type=float, default=0.8, help="training fraction")
    ap.add_argument("--seed", type=int, default=42, help="for a repeatable split")
    args = ap.parse_args()
 
    with open(args.dataset, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    if rows[0][1].strip() not in ("M", "B"):  # skip header if present
        rows = rows[1:]
 
    rng = np.random.default_rng(args.seed)
    train, valid = [], []
    for cls in ("M", "B"):  # stratified: keep the M/B proportion in both sets
        idx = [i for i, r in enumerate(rows) if r[1].strip() == cls]
        rng.shuffle(idx)
        cut = int(len(idx) * args.ratio)
        train += [rows[i] for i in idx[:cut]]
        valid += [rows[i] for i in idx[cut:]]
    rng.shuffle(train)
    rng.shuffle(valid)
 
    for name, data in (("data_training.csv", train), ("data_validation.csv", valid)):
        with open(name, "w", newline="") as f:
            csv.writer(f).writerows(data)
    print(f"{len(train)} training rows -> data_training.csv")
    print(f"{len(valid)} validation rows -> data_validation.csv")
 
 
if __name__ == "__main__":
    main()
