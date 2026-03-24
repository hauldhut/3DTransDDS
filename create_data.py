import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.dataset import TestbedDataset
from data.build_graph import smile_to_3d_graph


def create_data(datafile: str, cellfile: str, data_dir: str = "data", work_root: str = "data"):
    data_dir_path = Path(data_dir)
    work_root_path = Path(work_root)
    work_root_path.mkdir(parents=True, exist_ok=True)

    cell_features = pd.read_csv(cellfile, header=None).values
    smiles_df = pd.read_csv(data_dir_path / "smiles.csv")
    compound_iso_smiles = set(smiles_df["smile"].tolist())

    smile_graph = {}
    print("Start generating 3D graphs from SMILES...")
    for smile in tqdm(compound_iso_smiles):
        g3d = smile_to_3d_graph(smile)
        if g3d is not None:
            smile_graph[smile] = g3d
    print(f"Built {len(smile_graph)} graphs / {len(compound_iso_smiles)} SMILES")

    df = pd.read_csv(data_dir_path / f"{datafile}.csv")
    drug1 = np.asarray(df["drug1"])
    drug2 = np.asarray(df["drug2"])
    cell = np.asarray(df["cell"])
    label = np.asarray(df["label"])

    print("Start creating dataset files...")
    TestbedDataset(
        root=str(work_root_path),
        dataset=f"{datafile}_3d_new_drug1",
        xd=drug1,
        xt=cell,
        xt_feature=cell_features,
        y=label,
        smile_graph=smile_graph,
    )
    TestbedDataset(
        root=str(work_root_path),
        dataset=f"{datafile}_3d_new_drug2",
        xd=drug2,
        xt=cell,
        xt_feature=cell_features,
        y=label,
        smile_graph=smile_graph,
    )
    print("Dataset creation successful.")


def parse_args():
    parser = argparse.ArgumentParser(description="Create processed graph datasets from raw CSV.")
    parser.add_argument("--datafile", default="new_labels_0_10", help="Base csv name without extension.")
    parser.add_argument(
        "--cellfile",
        default="data/cell_features.csv",
        help="Path to cell feature csv.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing raw csv files.")
    parser.add_argument("--work-root", default="data", help="Output root for processed .pt files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_data(
        datafile=args.datafile,
        cellfile=args.cellfile,
        data_dir=args.data_dir,
        work_root=args.work_root,
    )
