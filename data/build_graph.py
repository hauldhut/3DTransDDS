from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom) -> np.ndarray:
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "Mg",
                "Na",
                "Ca",
                "Fe",
                "As",
                "Al",
                "I",
                "B",
                "V",
                "K",
                "Tl",
                "Yb",
                "Sb",
                "Sn",
                "Ag",
                "Pd",
                "Co",
                "Se",
                "Ti",
                "Zn",
                "H",
                "Li",
                "Ge",
                "Cu",
                "Au",
                "Ni",
                "Cd",
                "In",
                "Mn",
                "Zr",
                "Cr",
                "Pt",
                "Hg",
                "Pb",
                "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
        + [atom.GetIsAromatic()]
    )


def smile_to_3d_graph(smile: str) -> Optional[tuple]:
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)

        res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if res == -1:
            AllChem.Compute2DCoords(mol)

        AllChem.UFFOptimizeMolecule(mol)
        c_size = mol.GetNumAtoms()

        features: List[np.ndarray] = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append(feature / np.sum(feature))

        conformer = mol.GetConformer()
        positions = conformer.GetPositions()

        edges = []
        edge_attr = []
        for bond in mol.GetBonds():
            s_idx = bond.GetBeginAtomIdx()
            e_idx = bond.GetEndAtomIdx()
            edges.extend([(s_idx, e_idx), (e_idx, s_idx)])
            bond_length = np.linalg.norm(positions[s_idx] - positions[e_idx])
            edge_attr.extend([[bond_length], [bond_length]])

        edge_index = np.array(edges).T
        return c_size, features, edge_index, positions, edge_attr
    except Exception as exc:
        print(f"Error processing SMILES {smile}: {exc}")
        return None
