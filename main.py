import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import DataLoader

from data.dataset import TestbedDataset
from model.gnn_model import GCNPointNetGateTransformer


def save_aucs(aucs, filename):
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\t".join(map(str, aucs)) + "\n")


def train_one_epoch(model, device, drug1_loader_train, drug2_loader_train, optimizer, loss_fn, epoch, log_interval):
    print(f"Training on {len(drug1_loader_train.dataset)} samples...")
    model.train()
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1, data2 = data[0].to(device), data[1].to(device)
        y = data1.y.view(-1, 1).long().to(device).squeeze(1)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Train epoch: {epoch}\tLoss: {loss.item():.6f}")


def predict(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print(f"Make prediction for {len(drug1_loader_test.dataset)} samples...")
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1, data2 = data[0].to(device), data[1].to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).to("cpu").data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def parse_args():
    parser = argparse.ArgumentParser(description="Train GCNPointNetGateTransformer from processed datasets.")
    parser.add_argument("--datafile", default="new_labels_0_10")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--result-dir", default="data/result")
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    drug1_data = TestbedDataset(root=args.data_root, dataset=f"{args.datafile}_3d_new_drug1")
    drug2_data = TestbedDataset(root=args.data_root, dataset=f"{args.datafile}_3d_new_drug2")
    length = len(drug1_data)
    pot = int(length / 5)
    random_num = random.sample(range(0, length), length)
    os.makedirs(args.result_dir, exist_ok=True)

    for i in range(5):
        test_num = random_num[pot * i : pot * (i + 1)]
        train_num = random_num[: pot * i] + random_num[pot * (i + 1) :]

        drug1_train, drug1_test = drug1_data[train_num], drug1_data[test_num]
        drug2_train, drug2_test = drug2_data[train_num], drug2_data[test_num]

        loader1_train = DataLoader(drug1_train, batch_size=args.train_batch_size, shuffle=False)
        loader1_test = DataLoader(drug1_test, batch_size=args.test_batch_size, shuffle=False)
        loader2_train = DataLoader(drug2_train, batch_size=args.train_batch_size, shuffle=False)
        loader2_test = DataLoader(drug2_test, batch_size=args.test_batch_size, shuffle=False)

        model = GCNPointNetGateTransformer().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        model_file = os.path.join(args.result_dir, f"GCNPointNet_Gate_Transformer{i}--model_{args.datafile}.model")
        auc_file = os.path.join(args.result_dir, f"GCNPointNet_Gate_Transformer{i}--AUCs--{args.datafile}.txt")
        with open(auc_file, "w", encoding="utf-8") as f:
            f.write("Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL\n")

        best_auc = 0.0
        for epoch in range(args.epochs):
            train_one_epoch(
                model,
                device,
                loader1_train,
                loader2_train,
                optimizer,
                loss_fn,
                epoch + 1,
                args.log_interval,
            )
            T, S, Y = predict(model, device, loader1_test, loader2_test)
            auc = roc_auc_score(T, S)
            precision, recall, _ = metrics.precision_recall_curve(T, S)
            pr_auc = metrics.auc(recall, precision)
            bacc = balanced_accuracy_score(T, Y)
            tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec = precision_score(T, Y, zero_division=0)
            acc = accuracy_score(T, Y)
            kappa = cohen_kappa_score(T, Y)
            rec = recall_score(T, Y, zero_division=0)

            if best_auc < auc:
                best_auc = auc
                save_aucs([epoch, auc, pr_auc, acc, bacc, prec, tpr, kappa, rec], auc_file)
                torch.save(model.state_dict(), model_file)
            print("best_auc", best_auc)


if __name__ == "__main__":
    main()
