# -*- coding: utf-8 -*-
"""
Simplified Data Splitter (v2.0) - Only stratified_kfold + sample_level_stratified
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

from config import DISEASE_CONFIGS


def create_multi_disease_splits_v2(
    positive_pairs_csv: str,
    diseases: List[str],
    test_ratio: float = 0.1,
    eval_mode: Literal['stratified_kfold'] = 'stratified_kfold',
    split_mode: Literal['sample_level_stratified'] = 'sample_level_stratified',
    n_folds: int = 5,
    random_state: int = 42,
    min_test_size: int = 10
) -> Dict:
    print(f"\n{'='*80}")
    print("Multi-Disease Data Split (Simplified)")
    print(f"{'='*80}")
    print(f"[Eval Mode] {eval_mode}")
    print(f"[Split Mode] {split_mode}")
    print(f"[Num Diseases] {len(diseases)}")
    print(f"[Test Ratio] {test_ratio}")

    all_disease_data = {}
    pair_to_diseases = defaultdict(set)
    total_pos_samples = 0

    for disease in diseases:
        if disease in DISEASE_CONFIGS:
            disease_name_en = DISEASE_CONFIGS[disease]['name_en']
        else:
            disease_name_en = disease

        df = pd.read_csv(positive_pairs_csv)

        if 'specific_cancer' in df.columns:
            df_disease = df[df['specific_cancer'] == disease_name_en].copy()
        elif 'fd_name' in df.columns:
            df_disease = df[df['fd_name'] == disease_name_en].copy()
        elif 'md_name' in df.columns:
            df_disease = df[df['md_name'] == disease_name_en].copy()
        else:
            raise ValueError("CSV 文件中没有疾病列")

        if len(df_disease) == 0:
            print(f"  [Warning] {disease}: No samples found")
            continue

        pos_pairs = df_disease[['genename1', 'genename2']].values
        pos_labels = np.ones(len(pos_pairs))

        for pair in pos_pairs:
            pair_tuple = tuple(sorted(pair))
            pair_to_diseases[pair_tuple].add(disease)

        all_disease_data[disease] = {
            'pairs': pos_pairs,
            'labels': pos_labels,
            'disease_name_en': disease_name_en,
        }

        total_pos_samples += len(pos_pairs)
        print(f"  {disease} ({disease_name_en}): {len(pos_pairs)} samples")

    print(f"\n[Total Positive Samples] {total_pos_samples}")

    multi_disease_pairs = {p: ds for p, ds in pair_to_diseases.items() if len(ds) > 1}
    print(f"[Multi-Disease Pairs] {len(multi_disease_pairs)} / {len(pair_to_diseases)}")

    print(f"\n[Splitting External Test Set] {test_ratio*100}%")

    external_test = _split_external_test_pair_level(
        all_disease_data, pair_to_diseases, test_ratio, random_state
    )

    print(f"\n[Creating {eval_mode} + {split_mode} splits]")

    if eval_mode == 'stratified_kfold':
        cv_splits = _create_stratified_kfold_splits_sample_level_stratified(
            external_test['train_val_disease_pairs'],
            n_folds=n_folds,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")

    statistics = {
        'total_pairs': len(pair_to_diseases),
        'multi_disease_pairs': len(multi_disease_pairs),
        'multi_disease_ratio': len(multi_disease_pairs) / len(pair_to_diseases),
    }

    return {
        'split_mode': split_mode,
        'eval_mode': eval_mode,
        'diseases': diseases,
        'external_test': external_test['external_test'],
        'cv_splits': cv_splits,
        'train_val_disease_pairs': external_test['train_val_disease_pairs'],
        'statistics': statistics,
    }


def _split_external_test_pair_level(
    all_disease_data: Dict,
    pair_to_diseases: Dict[tuple, Set[str]],
    test_ratio: float,
    random_state: int
) -> Dict:
    rng = np.random.RandomState(random_state)

    all_pairs = list(pair_to_diseases.keys())
    n_test_pairs = int(len(all_pairs) * test_ratio)

    test_pair_indices = rng.choice(len(all_pairs), size=n_test_pairs, replace=False)
    test_pairs_set = set(all_pairs[i] for i in test_pair_indices)
    train_pairs_set = set(all_pairs) - test_pairs_set

    print(f"  Pair-level split:")
    print(f"    Total pairs: {len(all_pairs)}")
    print(f"    Test pairs: {len(test_pairs_set)} ({len(test_pairs_set)/len(all_pairs)*100:.1f}%)")
    print(f"    Train pairs: {len(train_pairs_set)} ({len(train_pairs_set)/len(all_pairs)*100:.1f}%)")

    test_disease_pairs = {}
    train_val_disease_pairs = {}

    for disease, data in all_disease_data.items():
        pairs = data['pairs']
        test_pairs = []
        train_val_pairs = []

        for pair in pairs:
            pair_tuple = tuple(sorted(pair))
            if pair_tuple in test_pairs_set:
                test_pairs.append(pair)
            else:
                train_val_pairs.append(pair)

        test_pairs = np.array(test_pairs) if test_pairs else np.empty((0, 2))
        train_val_pairs = np.array(train_val_pairs) if train_val_pairs else np.empty((0, 2))

        train_val_disease_pairs[disease] = {
            'pairs': train_val_pairs,
            'labels': np.ones(len(train_val_pairs)),
        }

        if len(test_pairs) > 0:
            test_disease_pairs[disease] = {
                'pairs': test_pairs,
                'labels': np.ones(len(test_pairs)),
            }

        print(f"  {disease}: train_val={len(train_val_pairs)}, test={len(test_pairs)}")

    all_test_pairs = []
    all_test_labels = []
    all_test_diseases = []

    for disease, data in test_disease_pairs.items():
        n_samples = len(data['pairs'])
        all_test_pairs.append(data['pairs'])
        all_test_labels.append(data['labels'])
        all_test_diseases.extend([disease] * n_samples)

    if all_test_pairs:
        all_test_pairs = np.concatenate(all_test_pairs, axis=0)
        all_test_labels = np.concatenate(all_test_labels, axis=0)
        all_test_diseases = np.array(all_test_diseases)
    else:
        all_test_pairs = np.empty((0, 2))
        all_test_labels = np.array([])
        all_test_diseases = np.array([])

    print(f"External test set: {len(all_test_pairs)} samples")

    return {
        'external_test': {
            'disease_pairs': test_disease_pairs,
            'all_pairs': all_test_pairs,
            'all_labels': all_test_labels,
            'all_diseases': all_test_diseases,
            'test_pairs_set': test_pairs_set,
        },
        'train_val_disease_pairs': train_val_disease_pairs,
        'train_pairs_set': train_pairs_set,
    }


def _create_stratified_kfold_splits_sample_level_stratified(
    train_val_disease_pairs: Dict,
    n_folds: int,
    random_state: int
) -> List[Dict]:
    print(f"  Strategy: Stratified K-Fold (sample-level)")

    all_samples = []
    all_diseases = []

    for disease, data in train_val_disease_pairs.items():
        for pair in data['pairs']:
            sample = (pair[0], pair[1], disease)
            all_samples.append(sample)
            all_diseases.append(disease)

    all_samples = np.array(all_samples, dtype=object)
    all_diseases = np.array(all_diseases)

    print(f"  Total samples: {len(all_samples)}")

    disease_to_id = {d: i for i, d in enumerate(sorted(set(all_diseases)))}
    stratify_labels = [disease_to_id[d] for d in all_diseases]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_samples, stratify_labels)):
        train_samples = all_samples[train_idx]
        val_samples = all_samples[val_idx]
        train_diseases = all_diseases[train_idx]
        val_diseases = all_diseases[val_idx]

        train_pairs = [(s[0], s[1]) for s in train_samples]
        val_pairs = [(s[0], s[1]) for s in val_samples]

        val_disease_counts = {}
        for d in val_diseases:
            val_disease_counts[d] = val_disease_counts.get(d, 0) + 1

        print(f"  Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)} ({len(val_idx)/len(all_samples)*100:.1f}%)")
        print(f"    Val disease distribution: {val_disease_counts}")

        cv_splits.append({
            'mode': 'stratified_kfold_sample_level_stratified',
            'fold': fold_idx,
            'train_pairs': np.array(train_pairs),
            'val_pairs': np.array(val_pairs),
            'train_pair_diseases': train_diseases,
            'val_pair_diseases': val_diseases,
            'train_diseases': list(set(train_diseases)),
            'val_diseases': list(set(val_diseases)),
            'val_disease_counts': val_disease_counts,
        })

    return cv_splits
