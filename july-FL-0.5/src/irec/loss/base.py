from irec.utils import (
    MetaParent,
    maybe_to_list,
)

import copy
import torch
import torch.nn as nn
import pickle
import os
import logging


class BaseLoss(metaclass=MetaParent):
    pass


class TorchLoss(BaseLoss, nn.Module):
    pass


class IdentityLoss(BaseLoss, config_name="identity"):
    def __call__(self, inputs):
        return inputs


class CompositeLoss(TorchLoss, config_name="composite"):
    def __init__(self, losses, weights=None, output_prefix=None):
        super().__init__()
        self._losses = losses
        self._weights = weights or [1.0] * len(losses)
        self._output_prefix = output_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        losses = []
        weights = []

        for loss_cfg in copy.deepcopy(config)["losses"]:
            weight = loss_cfg.pop("weight") if "weight" in loss_cfg else 1.0
            loss_function = BaseLoss.create_from_config(loss_cfg)

            weights.append(weight)
            losses.append(loss_function)

        return cls(
            losses=losses,
            weights=weights,
            output_prefix=config.get("output_prefix"),
        )

    def forward(self, inputs):
        total_loss = 0.0
        for loss, weight in zip(self._losses, self._weights):
            total_loss += weight * loss(inputs)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = total_loss.cpu().item()

        return total_loss


class FpsLoss(TorchLoss, config_name="fps"):
    def __init__(
        self,
        fst_embeddings_prefix,
        snd_embeddings_prefix,
        tau,
        normalize_embeddings=False,
        use_mean=True,
        output_prefix=None,
    ):
        super().__init__()
        self._fst_embeddings_prefix = fst_embeddings_prefix
        self._snd_embeddings_prefix = snd_embeddings_prefix
        self._tau = tau
        self._loss_function = nn.CrossEntropyLoss(
            reduction="mean" if use_mean else "sum",
        )
        self._normalize_embeddings = normalize_embeddings
        self._output_prefix = output_prefix
        print(self._tau)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            fst_embeddings_prefix=config["fst_embeddings_prefix"],
            snd_embeddings_prefix=config["snd_embeddings_prefix"],
            tau=config.get("temperature", 1.0),
            normalize_embeddings=config.get("normalize_embeddings", False),
            use_mean=config.get("use_mean", True),
            output_prefix=config.get("output_prefix"),
        )

    def forward(self, inputs):
        fst_embeddings = inputs[self._fst_embeddings_prefix]  # (x, embedding_dim)
        snd_embeddings = inputs[self._snd_embeddings_prefix]  # (x, embedding_dim)

        batch_size = fst_embeddings.shape[0]

        combined_embeddings = torch.cat(
            (fst_embeddings, snd_embeddings),
            dim=0,
        )  # (2 * x, embedding_dim)

        if self._normalize_embeddings:
            combined_embeddings = torch.nn.functional.normalize(
                combined_embeddings,
                p=2,
                dim=-1,
                eps=1e-6,
            )  # (2 * x, embedding_dim)

        similarity_scores = (
            torch.mm(combined_embeddings, combined_embeddings.T) / self._tau
        )  # (2 * x, 2 * x)

        positive_samples = torch.cat(
            (
                torch.diag(similarity_scores, batch_size),
                torch.diag(similarity_scores, -batch_size),
            ),
            dim=0,
        ).reshape(
            2 * batch_size, 1
        )  # (2 * x, 1)
        assert torch.allclose(
            torch.diag(similarity_scores, batch_size),
            torch.diag(similarity_scores, -batch_size),
        )

        mask = torch.ones(
            2 * batch_size,
            2 * batch_size,
            dtype=torch.bool,
        )  # (2 * x, 2 * x)
        mask = mask.fill_diagonal_(0)  # Remove equal embeddings scores
        for i in range(batch_size):  # Remove positives
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        negative_samples = similarity_scores[mask].reshape(
            2 * batch_size,
            -1,
        )  # (2 * x, 2 * x - 2)

        labels = (
            torch.zeros(2 * batch_size).to(positive_samples.device).long()
        )  # (2 * x)
        logits = torch.cat(
            (positive_samples, negative_samples),
            dim=1,
        )  # (2 * x, 2 * x - 1)

        loss = self._loss_function(logits, labels) / 2  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class SASRecLoss(TorchLoss, config_name="sasrec"):

    def __init__(self, positive_prefix, negative_prefix, output_prefix=None):
        super().__init__()
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        positive_scores = inputs[self._positive_prefix]  # (x)
        negative_scores = inputs[self._negative_prefix]  # (x)
        assert positive_scores.shape[0] == negative_scores.shape[0]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            positive_scores, torch.ones_like(positive_scores)
        ) + torch.nn.functional.binary_cross_entropy_with_logits(
            negative_scores, torch.zeros_like(negative_scores)
        )

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class SamplesSoftmaxLoss(TorchLoss, config_name="sampled_softmax"):
    def __init__(
        self,
        queries_prefix,
        positive_prefix,
        negative_prefix,
        output_prefix=None,
    ):
        super().__init__()
        self._queries_prefix = queries_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        queries_embeddings = inputs[self._queries_prefix]  # (batch_size, embedding_dim)
        positive_embeddings = inputs[
            self._positive_prefix
        ]  # (batch_size, embedding_dim)
        negative_embeddings = inputs[
            self._negative_prefix
        ]  # (num_negatives, embedding_dim) or (batch_size, num_negatives, embedding_dim)

        # b -- batch_size, d -- embedding_dim
        positive_scores = torch.einsum(
            "bd,bd->b",
            queries_embeddings,
            positive_embeddings,
        ).unsqueeze(
            -1
        )  # (batch_size, 1)

        if negative_embeddings.dim() == 2:  # (num_negatives, embedding_dim)
            # b -- batch_size, n -- num_negatives, d -- embedding_dim
            negative_scores = torch.einsum(
                "bd,nd->bn",
                queries_embeddings,
                negative_embeddings,
            )  # (batch_size, num_negatives)
        else:
            assert (
                negative_embeddings.dim() == 3
            )  # (batch_size, num_negatives, embedding_dim)
            # b -- batch_size, n -- num_negatives, d -- embedding_dim
            negative_scores = torch.einsum(
                "bd,bnd->bn",
                queries_embeddings,
                negative_embeddings,
            )  # (batch_size, num_negatives)
        all_scores = torch.cat(
            [positive_scores, negative_scores],
            dim=1,
        )  # (batch_size, 1 + num_negatives)

        logits = torch.log_softmax(
            all_scores,
            dim=1,
        )  # (batch_size, 1 + num_negatives)
        loss = (-logits)[:, 0]  # (batch_size)
        loss = loss.mean()  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class MCLSRLogqInBatchLoss(TorchLoss, config_name="mclsr_logq_inbatch"):
    """
    LogQ-corrected In-Batch Sampled Softmax Loss for MCLSR model.

    Uses in-batch negatives: positive items of other users in the batch serve as negatives.
    This naturally produces a popularity-proportional sampling distribution,
    which LogQ correction precisely compensates.

    LogQ correction is applied only to negatives (not to the positive).
    """

    def __init__(
        self,
        queries_prefix,
        positive_prefix,
        positive_ids_prefix,
        path_to_item_counts,
        logq_lambda=1.0,
        output_prefix=None,
    ):
        super().__init__()
        self._queries_prefix = queries_prefix
        self._positive_prefix = positive_prefix
        self._positive_ids_prefix = positive_ids_prefix
        self._output_prefix = output_prefix
        self._logq_lambda = logq_lambda

        if not os.path.exists(path_to_item_counts):
            raise FileNotFoundError(
                f"Item counts file not found at {path_to_item_counts}"
            )

        with open(path_to_item_counts, "rb") as f:
            counts = pickle.load(f)

        counts_tensor = torch.tensor(counts, dtype=torch.float32)
        probs = torch.clamp(counts_tensor / counts_tensor.sum(), min=1e-10)
        log_q = torch.log(probs)
        self.register_buffer("_log_q_table", log_q)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            queries_prefix=config["queries_prefix"],
            positive_prefix=config["positive_prefix"],
            positive_ids_prefix=config["positive_ids_prefix"],
            path_to_item_counts=config["path_to_item_counts"],
            logq_lambda=config.get("logq_lambda", 1.0),
            output_prefix=config.get("output_prefix"),
        )

    def forward(self, inputs):
        queries = inputs[self._queries_prefix]  # (B, D)
        pos_embs = inputs[self._positive_prefix]  # (B, D)
        pos_ids = inputs[self._positive_ids_prefix]  # (B,)

        if self._log_q_table.device != queries.device:
            self._log_q_table = self._log_q_table.to(queries.device)

        batch_size = queries.size(0)

        # All-pairs scores: each query against all positive items in batch
        # all_scores[i, j] = query_i · pos_emb_j
        # Diagonal (i, i) = positive score, off-diagonal = in-batch negatives
        all_scores = torch.mm(queries, pos_embs.T)  # (B, B)

        # LogQ correction on negatives only:
        # 1) Apply correction to ALL candidates (columns)
        # 2) Undo correction on the diagonal (positives)
        log_q = self._log_q_table[pos_ids]  # (B,)
        all_scores = all_scores - self._logq_lambda * log_q.unsqueeze(0)  # (B, B)
        all_scores.diagonal().add_(self._logq_lambda * log_q)

        # Cross-entropy: positive is the diagonal (target index i for row i)
        labels = torch.arange(batch_size, device=queries.device)
        loss = torch.nn.functional.cross_entropy(all_scores, labels)

        if self._output_prefix:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss
