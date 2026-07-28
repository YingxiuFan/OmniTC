# -*- coding: utf-8 -*-
"""
TargetSynergy Core Training Functions (Simplified)
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

from models.target_synergy import TargetSynergyModel
from losses.hybrid_loss import HybridLoss
from data.dataset import collate_fn
from utils.metrics import compute_metrics, print_metrics
from config import TRAINING_CONFIG


def train_one_epoch(
    model: TargetSynergyModel,
    dataloader: DataLoader,
    criterion: HybridLoss,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        node_features = batch['node_features'].to(device)
        edge_index = batch['edge_index'].to(device)

        target1_indices = batch['target1_indices']
        target2_indices = batch['target2_indices']

        if target1_indices.dim() > 1:
            target1_indices = target1_indices.squeeze()
        if target2_indices.dim() > 1:
            target2_indices = target2_indices.squeeze()

        target1_indices = target1_indices.to(device)
        target2_indices = target2_indices.to(device)
        disease_features = batch['disease_features'].to(device)

        n_clusters = batch['n_clusters']
        if n_clusters.dim() > 1:
            n_clusters = n_clusters.squeeze()
        n_clusters = n_clusters.to(device)

        labels = batch['labels']
        if labels.dim() > 1:
            labels = labels.squeeze()
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            node_features=node_features,
            edge_index=edge_index,
            target1_indices=target1_indices,
            target2_indices=target2_indices,
            disease_features=disease_features,
            n_clusters=n_clusters,
        )

        loss_dict = criterion(
            pred_logits=outputs['logits'],
            labels=labels,
            target_pair_emb=outputs['target_pair_emb'],
            disease_global_emb=outputs['global_disease_emb'],
            cluster_reps=outputs['disease_cluster_emb'],
            n_clusters=n_clusters,
        )

        loss = loss_dict['total']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = outputs['prob']
        preds = (probs > 0.5).long()

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    metrics = compute_metrics(y_true=np.array(all_labels), y_pred=np.array(all_preds), y_prob=np.array(all_probs))
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def validate(
    model: TargetSynergyModel,
    dataloader: DataLoader,
    criterion: HybridLoss,
    device: str,
    return_details: bool = False,
    node_names: np.ndarray = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="[Val]")

        for batch in pbar:
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)

            target1_indices = batch['target1_indices']
            target2_indices = batch['target2_indices']

            if target1_indices.dim() > 1:
                target1_indices = target1_indices.squeeze()
            if target2_indices.dim() > 1:
                target2_indices = target2_indices.squeeze()

            target1_indices = target1_indices.to(device)
            target2_indices = target2_indices.to(device)
            disease_features = batch['disease_features'].to(device)

            n_clusters = batch['n_clusters']
            if n_clusters.dim() > 1:
                n_clusters = n_clusters.squeeze()
            n_clusters = n_clusters.to(device)

            labels = batch['labels']
            if labels.dim() > 1:
                labels = labels.squeeze()
            labels = labels.to(device)

            outputs = model(
                node_features=node_features,
                edge_index=edge_index,
                target1_indices=target1_indices,
                target2_indices=target2_indices,
                disease_features=disease_features,
                n_clusters=n_clusters,
            )

            loss_dict = criterion(
                pred_logits=outputs['logits'],
                labels=labels,
                target_pair_emb=outputs['target_pair_emb'],
                disease_global_emb=outputs['global_disease_emb'],
                cluster_reps=outputs['disease_cluster_emb'],
                n_clusters=n_clusters,
            )

            loss = loss_dict['total']
            total_loss += loss.item()

            probs = outputs['prob']
            preds = (probs > 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    metrics = compute_metrics(y_true=np.array(all_labels), y_pred=np.array(all_preds), y_prob=np.array(all_probs))
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def train(
    model: TargetSynergyModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str,
    save_dir: Path,
    early_stopping_patience: int = 20,
    alpha_orth: float = 0.1,
    beta_con: float = 0.5,
    temperature: float = 0.1,
    optim_metric: str = 'auprc',
    gamma_adversarial: float = 1.0,
    adversarial_temperature: float = 0.5,
) -> Dict[str, list]:
    criterion = HybridLoss(
        alpha_orth=alpha_orth,
        beta_con=beta_con,
        temperature=temperature,
        gamma_adversarial=gamma_adversarial,
        adversarial_temperature=adversarial_temperature,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=TRAINING_CONFIG['weight_decay'])

    history = {
        'train_loss': [],
        'train_auc': [],
        'train_f1': [],
        'train_mcc': [],
        'train_auprc': [],
        'val_loss': [],
        'val_auc': [],
        'val_f1': [],
        'val_mcc': [],
        'val_auprc': [],
    }

    best_metric_value = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        history['train_loss'].append(train_metrics['loss'])
        history['train_auc'].append(train_metrics['auc_roc'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_mcc'].append(train_metrics['mcc'])
        history['train_auprc'].append(train_metrics['auc_pr'])

        val_metrics = validate(model, val_loader, criterion, device)

        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc_roc'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_mcc'].append(val_metrics['mcc'])
        history['val_auprc'].append(val_metrics['auc_pr'])

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: Train Loss={train_metrics['loss']:.4f}, Val AUC={val_metrics['auc_roc']:.4f}, Val AUPRC={val_metrics['auc_pr']:.4f}")

        current_metric_value = val_metrics[optim_metric]

        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_epoch = epoch
            patience_counter = 0

            save_path = save_dir / 'best_model.pt'
            model.save_checkpoint(
                str(save_path),
                epoch=epoch,
                val_auc=val_metrics['auc_roc'],
                val_f1=val_metrics['f1'],
                val_mcc=val_metrics['mcc'],
                val_auprc=val_metrics['auc_pr'],
            )
            print(f"  Best model saved: {save_path} (Val {optim_metric}: {best_metric_value:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print(f"\nTraining completed: Best Epoch={best_epoch}, Best {optim_metric}={best_metric_value:.4f}")

    return history


def test(
    model: TargetSynergyModel,
    dataloader: DataLoader,
    device: str,
    return_details: bool = False,
    node_names: np.ndarray = None,
    disease_features_list: list = None,
    idx_to_disease: dict = None,
):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    all_target1_indices = []
    all_target2_indices = []
    all_disease_indices = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="[Test]")

        for batch in pbar:
            node_features = batch['node_features'].to(device)
            edge_index = batch['edge_index'].to(device)

            target1_indices = batch['target1_indices']
            target2_indices = batch['target2_indices']

            if target1_indices.dim() == 2 and target1_indices.shape[1] == 1:
                target1_indices = target1_indices.squeeze(-1)
            if target2_indices.dim() == 2 and target2_indices.shape[1] == 1:
                target2_indices = target2_indices.squeeze(-1)

            target1_indices = target1_indices.to(device)
            target2_indices = target2_indices.to(device)
            disease_features = batch['disease_features'].to(device)

            n_clusters = batch['n_clusters']
            if n_clusters.dim() > 1:
                n_clusters = n_clusters.squeeze()
            n_clusters = n_clusters.to(device)

            outputs = model(
                node_features=node_features,
                edge_index=edge_index,
                target1_indices=target1_indices,
                target2_indices=target2_indices,
                disease_features=disease_features,
                n_clusters=n_clusters,
            )

            probs = outputs['prob']
            preds = (probs > 0.5).long()

            all_labels.extend(batch['labels'].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            if return_details:
                all_target1_indices.extend(target1_indices.cpu().numpy())
                all_target2_indices.extend(target2_indices.cpu().numpy())

                if 'disease_indices' in batch:
                    all_disease_indices.extend(batch['disease_indices'].cpu().numpy())
                elif disease_features_list is not None:
                    batch_disease_features = batch['disease_features'].cpu().numpy()
                    for i, df in enumerate(batch_disease_features):
                        disease_idx = -1
                        for j, ref_df in enumerate(disease_features_list):
                            if np.allclose(df, ref_df, rtol=1e-5):
                                disease_idx = j
                                break
                        all_disease_indices.append(disease_idx)
                else:
                    all_disease_indices.append(-1)

    metrics = compute_metrics(y_true=np.array(all_labels), y_pred=np.array(all_preds), y_prob=np.array(all_probs))

    if return_details:
        detailed_results = []
        for i in range(len(all_probs)):
            gene1 = node_names[all_target1_indices[i]] if node_names is not None else str(all_target1_indices[i])
            gene2 = node_names[all_target2_indices[i]] if node_names is not None else str(all_target2_indices[i])

            disease_idx = all_disease_indices[i] if all_disease_indices else -1
            disease = idx_to_disease.get(disease_idx, 'unknown') if idx_to_disease is not None else 'unknown'

            detailed_results.append({
                'gene1': gene1,
                'gene2': gene2,
                'disease': disease,
                'label': int(all_labels[i]),
                'probability': float(all_probs[i]),
                'prediction': int(all_preds[i]),
            })
        return metrics, detailed_results

    return metrics
