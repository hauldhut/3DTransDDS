import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


class PointNet(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.fc_bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, batch):
        x = x.t().unsqueeze(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.squeeze(0).t()
        global_feature = gmp(x, batch)
        global_feature = F.relu(self.fc_bn1(self.fc1(global_feature)))
        global_feature = self.dropout(global_feature)
        return self.fc2(global_feature)


class GCNPointNetGateTransformer(nn.Module):
    def __init__(self, n_output=2, num_features_xd=78, num_features_xt=954, output_dim=128, dropout=0.2):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.n_output = n_output

        self.gcn_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.gcn_conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.gcn_conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.gcn_fc1 = nn.Linear(num_features_xd * 4, num_features_xd * 2)
        self.gcn_fc2 = nn.Linear(num_features_xd * 2, output_dim)

        self.pointnet = PointNet(input_dim=3, output_dim=output_dim)
        self.cell_reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
        self.fuse_gate = nn.Linear(output_dim * 2, output_dim)
        self.fuse_proj = nn.Linear(output_dim * 2, output_dim)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2)

        self.final_fc1 = nn.Linear(3 * output_dim, 512)
        self.final_fc2 = nn.Linear(512, 256)
        self.final_out = nn.Linear(256, self.n_output)

    def _process_drug(self, x, edge_index, pos, batch):
        x_gcn = self.relu(self.gcn_conv1(x, edge_index))
        x_gcn = self.relu(self.gcn_conv2(x_gcn, edge_index))
        x_gcn = self.relu(self.gcn_conv3(x_gcn, edge_index))
        x_gcn = gmp(x_gcn, batch)
        x_gcn = self.relu(self.gcn_fc1(x_gcn))
        x_gcn = self.dropout(x_gcn)
        x_gcn = self.gcn_fc2(x_gcn)
        x_pointnet = self.pointnet(pos, batch)
        return x_gcn, x_pointnet

    def forward(self, data1, data2):
        x1_gcn, x1_pointnet = self._process_drug(data1.x, data1.edge_index, data1.pos, data1.batch)
        x2_gcn, x2_pointnet = self._process_drug(data2.x, data2.edge_index, data2.pos, data2.batch)

        drug1_combined = torch.cat((x1_gcn, x1_pointnet), dim=1)
        drug2_combined = torch.cat((x2_gcn, x2_pointnet), dim=1)

        gate_1 = torch.sigmoid(self.fuse_gate(drug1_combined))
        proj_1 = torch.tanh(self.fuse_proj(drug1_combined))
        drug1_fused = gate_1 * proj_1 + (1 - gate_1) * x1_gcn

        gate_2 = torch.sigmoid(self.fuse_gate(drug2_combined))
        proj_2 = torch.tanh(self.fuse_proj(drug2_combined))
        drug2_fused = gate_2 * proj_2 + (1 - gate_2) * x2_gcn

        cell_vector = self.cell_reduction(data1.cell)
        transformer_input = torch.stack([drug1_fused, drug2_fused, cell_vector], dim=1)
        transformer_output = self.transformer_encoder(transformer_input)
        xc = transformer_output.view(transformer_output.size(0), -1)

        out = self.relu(self.final_fc1(xc))
        out = self.dropout(out)
        out = self.relu(self.final_fc2(out))
        out = self.dropout(out)
        return self.final_out(out)
