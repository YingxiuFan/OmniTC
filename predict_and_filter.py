#!/usr/bin/env python3
"""
TargetSynergy prediction and filtering script for the simplified pipeline.
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DISEASE_CONFIGS, PATHS
from models.target_synergy import TargetSynergyModel


def normalize_input_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize supported input column layouts to gene1/gene2/disease."""
    column_map = {}

    if {'gene1', 'gene2'}.issubset(df.columns):
        column_map['gene1'] = 'gene1'
        column_map['gene2'] = 'gene2'
    elif {'genename1', 'genename2'}.issubset(df.columns):
        column_map['genename1'] = 'gene1'
        column_map['genename2'] = 'gene2'
    else:
        raise ValueError("Input CSV must contain gene1/gene2 or genename1/genename2")

    for disease_col in ['disease', 'specific_cancer', 'fd_name', 'md_name']:
        if disease_col in df.columns:
            column_map[disease_col] = 'disease'
            break
    else:
        raise ValueError("Input CSV must contain disease, specific_cancer, fd_name, or md_name")

    normalized = df.rename(columns=column_map)[['gene1', 'gene2', 'disease']].copy()
    normalized['gene1'] = normalized['gene1'].astype(str).str.strip()
    normalized['gene2'] = normalized['gene2'].astype(str).str.strip()
    normalized['disease'] = normalized['disease'].astype(str).str.strip()
    return normalized


class TargetPairPredictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._disease_feature_cache = {}
        self._load_full_graph()
        self._load_model()

    def _load_model(self):
        print(f"[Loading Model] {self.checkpoint_path}")
        self.model, _ = TargetSynergyModel.load_checkpoint(
            str(self.checkpoint_path),
            device=self.device,
        )
        self.model.eval()
        print(
            f"[Model] Loaded successfully "
            f"(hidden_dim={self.model.hidden_dim}, jk_num_layers={self.model.jk_num_layers})"
        )

    def _load_full_graph(self):
        print(f"[Loading Graph] {PATHS['full_graph_h5']}")
        with h5py.File(PATHS['full_graph_h5'], 'r') as f:
            self.node_features = f['nodes']['node_features'][:]
            node_names_raw = f['nodes']['node_names'][:]
            self.node_names = np.array([
                name.decode('utf-8') if isinstance(name, bytes) else name
                for name in node_names_raw
            ])
            self.node_types = f['nodes']['node_type'][:]

            edge_list = []
            for edge_type in f['edges']:
                src = f['edges'][edge_type]['source'][:]
                tgt = f['edges'][edge_type]['target'][:]
                edge_list.append(np.stack([src, tgt], axis=0))

            if not edge_list:
                raise ValueError("No edges found in graph")

            edge_index = np.concatenate(edge_list, axis=1)
            self.edge_index = torch.from_numpy(edge_index).long()

        priority = {2: 0, 4: 1, 3: 2, 1: 3, 0: 4, 5: 5}
        name_to_all_indices = defaultdict(list)
        for idx, name in enumerate(self.node_names):
            name_to_all_indices[name].append(idx)

        self.gene_name_to_idx = {
            name: min(indices, key=lambda i: priority.get(self.node_types[i], 6))
            for name, indices in name_to_all_indices.items()
        }

        print(f"  - Nodes: {len(self.node_names):,}")
        print(f"  - Edges: {self.edge_index.shape[1]:,}")
        print(f"  - Feature dim: {self.node_features.shape[1]}")

    def _normalize_disease_code(self, disease: str) -> str:
        disease = disease.strip()

        if disease in DISEASE_CONFIGS:
            return disease

        for disease_code, config in DISEASE_CONFIGS.items():
            if disease == config['name_en']:
                return disease_code

        raise ValueError(f"Unsupported disease: {disease}")

    def _load_disease_features(self, disease_code: str):
        if disease_code in self._disease_feature_cache:
            return self._disease_feature_cache[disease_code]

        feature_path = PATHS['output_dir'] / "disease_features" / "ttd_based" / f"{disease_code}_n2v_features.npz"
        if not feature_path.exists():
            raise FileNotFoundError(f"Disease features not found: {feature_path}")

        data = np.load(feature_path)
        disease_features = torch.from_numpy(data['disease_features']).float()
        n_clusters = int(data['n_clusters'])
        self._disease_feature_cache[disease_code] = (disease_features, n_clusters)
        print(f"  [Loaded Features] {disease_code}: {n_clusters} clusters")
        return disease_features, n_clusters

    def predict_pairs(self, pairs_df: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
        normalized = normalize_input_pairs(pairs_df)
        normalized['disease_code'] = normalized['disease'].map(
            lambda disease: self._normalize_disease_code(disease)
        )

        all_results = []
        node_features = torch.from_numpy(self.node_features).float().to(self.device)
        edge_index = self.edge_index.to(self.device)

        for disease_code, group in normalized.groupby('disease_code', sort=False):
            try:
                disease_features, n_clusters = self._load_disease_features(disease_code)
            except Exception as exc:
                print(f"  Failed to load disease features for {disease_code}: {exc}")
                for _, row in group.iterrows():
                    all_results.append({
                        'disease_input': row['disease'],
                        'disease': disease_code,
                        'gene1': row['gene1'],
                        'gene2': row['gene2'],
                        'probability': np.nan,
                        'prediction': np.nan,
                        'label': 'error',
                    })
                continue

            valid_rows = []
            valid_indices_1 = []
            valid_indices_2 = []

            for _, row in group.iterrows():
                idx1 = self.gene_name_to_idx.get(row['gene1'])
                idx2 = self.gene_name_to_idx.get(row['gene2'])

                if idx1 is None or idx2 is None:
                    all_results.append({
                        'disease_input': row['disease'],
                        'disease': disease_code,
                        'gene1': row['gene1'],
                        'gene2': row['gene2'],
                        'probability': np.nan,
                        'prediction': np.nan,
                        'label': 'oog',
                    })
                    continue

                valid_rows.append(row)
                valid_indices_1.append(idx1)
                valid_indices_2.append(idx2)

            if not valid_rows:
                continue

            disease_features = disease_features.to(self.device)

            for start in range(0, len(valid_rows), batch_size):
                end = min(start + batch_size, len(valid_rows))
                batch_rows = valid_rows[start:end]
                batch_target1 = torch.LongTensor(valid_indices_1[start:end]).to(self.device)
                batch_target2 = torch.LongTensor(valid_indices_2[start:end]).to(self.device)
                batch_size_actual = end - start

                batch_disease_features = disease_features.unsqueeze(0).repeat(batch_size_actual, 1, 1)
                batch_n_clusters = torch.LongTensor([n_clusters] * batch_size_actual).to(self.device)

                with torch.no_grad():
                    outputs = self.model(
                        node_features=node_features,
                        edge_index=edge_index,
                        target1_indices=batch_target1,
                        target2_indices=batch_target2,
                        disease_features=batch_disease_features,
                        n_clusters=batch_n_clusters,
                    )

                probs = outputs['prob'].detach().cpu().numpy()
                preds = (probs > 0.5).astype(int)

                for row, prob, pred in zip(batch_rows, probs, preds):
                    all_results.append({
                        'disease_input': row['disease'],
                        'disease': disease_code,
                        'gene1': row['gene1'],
                        'gene2': row['gene2'],
                        'probability': float(prob),
                        'prediction': int(pred),
                        'label': 'valid',
                    })

        results = pd.DataFrame(all_results)
        if results.empty:
            return pd.DataFrame(columns=[
                'disease_input', 'disease', 'gene1', 'gene2', 'probability', 'prediction', 'label'
            ])

        return results.sort_values(
            by=['label', 'probability'],
            ascending=[True, False],
            na_position='last',
        ).reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description='TargetSynergy Prediction and Filtering (Simplified)')
    parser.add_argument('--checkpoint', '-c', required=True, help='Model checkpoint path')
    parser.add_argument('--input-pairs', '-i', required=True, help='Input target-pair CSV')
    parser.add_argument('--output', '-o', required=True, help='Output CSV path for all predictions')
    parser.add_argument('--min-score', type=float, default=0.5, help='Threshold for filtered predictions')
    parser.add_argument('--batch-size', type=int, default=256, help='Prediction batch size')
    parser.add_argument('--device', type=str, default=None, help='PyTorch device, e.g. cuda:0 or cpu')
    parser.add_argument('--filtered-output', type=str, default=None, help='Optional output CSV for filtered predictions')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("TargetSynergy Prediction (Simplified)")
    print("=" * 60)

    pairs_df = pd.read_csv(args.input_pairs)
    normalized = normalize_input_pairs(pairs_df)
    print(
        f"[Input] {len(normalized)} target pairs, "
        f"{normalized['disease'].nunique()} disease labels"
    )

    predictor = TargetPairPredictor(args.checkpoint, device=args.device)
    results = predictor.predict_pairs(normalized, batch_size=args.batch_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\n[Output] Saved all predictions to {output_path}")

    valid_results = results[results['label'] == 'valid'].copy()
    if not valid_results.empty:
        valid_results['probability'] = valid_results['probability'].astype(float)
        filtered = valid_results[valid_results['probability'] >= args.min_score].copy()
    else:
        filtered = valid_results

    filtered_output = (
        Path(args.filtered_output)
        if args.filtered_output is not None
        else output_path.with_name(f"{output_path.stem}_filtered{output_path.suffix}")
    )
    filtered.to_csv(filtered_output, index=False)

    print(f"[Filter] Valid predictions: {len(valid_results)}")
    print(f"[Filter] Predictions >= {args.min_score}: {len(filtered)}")
    print(f"[Output] Saved filtered predictions to {filtered_output}")

    if not filtered.empty:
        top_results = filtered.sort_values('probability', ascending=False).head(10)
        print("\n[Top Predictions]")
        for _, row in top_results.iterrows():
            print(
                f"  {row['gene1']} + {row['gene2']} "
                f"({row['disease']}): {row['probability']:.4f}"
            )


if __name__ == '__main__':
    main()
