#!/usr/bin/env python3
"""Load a saved model, predict on a dataset and evaluate with binary cross-entropy.
 
Usage: python predict.py --model saved_model.npy --dataset data_validation.csv
"""
import argparse
 
import numpy as np
 
from train import MLP, accuracy, binary_crossentropy, one_hot, read_csv
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="./saved_model.npy")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
 
    m = np.load(args.model, allow_pickle=True).item()
    net = MLP(m["sizes"], m["activation"])
    net.W, net.b = m["W"], m["b"]
 
    labels, x = read_csv(args.dataset)
    x = (x - m["mean"]) / m["std"]  # same scaling as during training
    y = one_hot(labels)
    p = net.predict_proba(x)
 
    met = classification_metrics(p, y)

    print(f"samples              : {len(x)}")
    print(f"binary cross-entropy : {binary_crossentropy(p[:, 1], y[:, 1]):.4f}")
    print(f"accuracy             : {accuracy(p, y):.4f}")
    print(f"precision / recall : {met['precision']:.4f} / {met['recall']:.4f}")
    print(f"F1 score             : {met['f1']:.4f}")
    print(f"confusion (M=positive): TP={met['tp']} TN={met['tn']} FP={met['fp']} FN={met['fn']}")
 
 
if __name__ == "__main__":
    main()

