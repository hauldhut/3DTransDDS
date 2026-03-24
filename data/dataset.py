import os
from itertools import islice
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset


GraphItem = Tuple[int, List[np.ndarray], np.ndarray, np.ndarray, List[List[float]]]


class TestbedDataset(InMemoryDataset):
    def __init__(
        self,
        root: str = "./data",
        dataset: str = "_drug1",
        xd=None,
        xt=None,
        y=None,
        xt_feature=None,
        transform=None,
        pre_transform=None,
        smile_graph: Dict[str, GraphItem] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print(f"Pre-processed data found: {self.processed_paths[0]}, loading ...")
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print(f"Pre-processed data {self.processed_paths[0]} not found, creating...")
            self.process(xd, xt, xt_feature, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + ".pt"]

    def download(self):
        return

    def _download(self):
        return

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    @staticmethod
    def get_cell_feature(cell_id, cell_features):
        for row in islice(cell_features, 0, None):
            if str(cell_id) == str(row[0]):
                return row[1:]
        return False

    def process(self, xd, xt, xt_feature, y, smile_graph):
        assert len(xd) == len(xt) == len(y), "xd/xt/y must have the same length"
        data_list = []
        print("number of data", len(xd))

        for i in range(len(xd)):
            smiles = xd[i]
            target = xt[i]
            label = y[i]

            graph_data = smile_graph.get(smiles) if smile_graph is not None else None
            if graph_data is None:
                continue

            c_size, features, edge_index, pos, edge_attr = graph_data
            gcn_data = DATA.Data(
                x=torch.tensor(np.array(features), dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor([label], dtype=torch.float32),
                pos=torch.tensor(pos, dtype=torch.float32),
                edge_attr=torch.tensor(np.array(edge_attr), dtype=torch.float32),
            )

            cell = self.get_cell_feature(target, xt_feature)
            if np.any(cell == False):  # noqa: E712
                continue

            new_cell = [float(n) for n in cell]
            gcn_data.cell = torch.tensor([new_cell], dtype=torch.float32)
            gcn_data.__setitem__("c_size", torch.tensor([c_size], dtype=torch.long))
            data_list.append(gcn_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print(f"Graph construction done. Total valid graphs: {len(data_list)}")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
