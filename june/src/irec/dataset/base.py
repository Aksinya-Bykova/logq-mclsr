from collections import defaultdict

from tqdm import tqdm

from irec.dataset.samplers import TrainSampler, EvalSampler

from irec.utils import MetaParent, DEVICE

import pickle
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import os
import logging

logger = logging.getLogger(__name__)


class BaseDataset(metaclass=MetaParent):
    def get_samplers(self):
        raise NotImplementedError

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def max_sequence_length(self):
        return self._max_sequence_length


class BaseSequenceDataset(BaseDataset):
    def __init__(
        self,
        train_sampler,
        validation_sampler,
        test_sampler,
        num_users,
        num_items,
        max_sequence_length,
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @staticmethod
    def _create_sequences(data, max_sample_len):  # TODO
        user_sequences = []
        item_sequences = []

        max_user_id = 0
        max_item_id = 0
        max_sequence_length = 0

        for sample in data:
            sample = sample.strip("\n").split(" ")
            item_ids = [int(item_id) for item_id in sample[1:]][-max_sample_len:]
            user_id = int(sample[0])

            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, max(item_ids))
            max_sequence_length = max(max_sequence_length, len(item_ids))

            user_sequences.append(user_id)
            item_sequences.append(item_ids)

        return (
            user_sequences,
            item_sequences,
            max_user_id,
            max_item_id,
            max_sequence_length,
        )

    def get_samplers(self):
        return (
            self._train_sampler,
            self._validation_sampler,
            self._test_sampler,
        )

    @property
    def meta(self):
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "max_sequence_length": self.max_sequence_length,
        }


class SequenceDataset(BaseDataset, config_name="sequence"):
    def __init__(
        self,
        train_sampler,
        validation_sampler,
        test_sampler,
        num_users,
        num_items,
        max_sequence_length,
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_users = num_users
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir_path = os.path.join(
            config["path_to_data_dir"],
            config["name"],
        )

        common_params_for_creation = {
            "dir_path": data_dir_path,
            "max_sequence_length": config["max_sequence_length"],
            "use_cached": config.get("use_cached", False),
        }

        train_dataset, train_max_user_id, train_max_item_id, train_seq_len = (
            cls._create_dataset(part="train", **common_params_for_creation)
        )
        validation_dataset, valid_max_user_id, valid_max_item_id, valid_seq_len = (
            cls._create_dataset(part="valid", **common_params_for_creation)
        )
        test_dataset, test_max_user_id, test_max_item_id, test_seq_len = (
            cls._create_dataset(part="test", **common_params_for_creation)
        )

        max_user_id = max(
            [train_max_user_id, valid_max_user_id, test_max_user_id],
        )
        max_item_id = max(
            [train_max_item_id, valid_max_item_id, test_max_item_id],
        )
        max_seq_len = max([train_seq_len, valid_seq_len, test_seq_len])

        logger.info("Train dataset size: {}".format(len(train_dataset)))
        logger.info("Test dataset size: {}".format(len(test_dataset)))
        logger.info("Max user id: {}".format(max_user_id))
        logger.info("Max item id: {}".format(max_item_id))
        logger.info("Max sequence length: {}".format(max_seq_len))

        train_interactions = sum(
            list(map(lambda x: len(x), train_dataset)),
        )  # whole user history as a sample
        valid_interactions = len(
            validation_dataset,
        )  # each new interaction as a sample
        test_interactions = len(
            test_dataset,
        )  # each new interaction as a sample
        logger.info(
            "{} dataset sparsity: {}".format(
                config["name"],
                (train_interactions + valid_interactions + test_interactions)
                / max_user_id
                / max_item_id,
            ),
        )

        samplers_config = config["samplers"]
        train_sampler = TrainSampler.create_from_config(
            samplers_config,
            dataset=train_dataset,
            num_users=max_user_id,
            num_items=max_item_id,
        )
        validation_sampler = EvalSampler.create_from_config(
            samplers_config,
            dataset=validation_dataset,
            num_users=max_user_id,
            num_items=max_item_id,
        )
        test_sampler = EvalSampler.create_from_config(
            samplers_config,
            dataset=test_dataset,
            num_users=max_user_id,
            num_items=max_item_id,
        )

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_users=max_user_id,
            num_items=max_item_id,
            max_sequence_length=max_seq_len,
        )

    @classmethod
    def _create_dataset(
        cls,
        dir_path,
        part,
        max_sequence_length=None,
        use_cached=False,
    ):
        cache_path = os.path.join(dir_path, "{}.pkl".format(part))

        if use_cached and os.path.exists(cache_path):
            logger.info("Loading cached dataset from {}".format(cache_path))
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        return cls._build_and_cache_dataset(
            dir_path, part, max_sequence_length, cache_path, use_cached
        )

    @classmethod
    def _build_and_cache_dataset(
        cls, dir_path, part, max_sequence_length, cache_path, use_cached
    ):
        logger.info(
            "Cache is forcefully ignored."
            if not use_cached
            else "No cached dataset has been found."
        )
        dataset_path = os.path.join(dir_path, "{}.txt".format(part))
        logger.info("Creating a dataset from {}...".format(dataset_path))

        with open(dataset_path, "r") as f:
            sequence_info = cls._create_sequences(f, max_sequence_length)
        (
            user_sequences,
            item_sequences,
            max_user_id,
            max_item_id,
            max_sequence_len,
        ) = sequence_info

        dataset = []
        for user_id, item_ids in zip(user_sequences, item_sequences):
            dataset.append(
                {
                    "user.ids": [user_id],
                    "user.length": 1,
                    "item.ids": item_ids,
                    "item.length": len(item_ids),
                },
            )

        logger.info("{} dataset size: {}".format(part, len(dataset)))
        logger.info(
            "{} dataset max sequence length: {}".format(
                part,
                max_sequence_len,
            ),
        )

        with open(cache_path, "wb") as dataset_file:
            pickle.dump(
                (dataset, max_user_id, max_item_id, max_sequence_len),
                dataset_file,
            )

        return dataset, max_user_id, max_item_id, max_sequence_len

    @staticmethod
    def _create_sequences(data, max_sample_len):  # TODO
        user_sequences = []
        item_sequences = []

        max_user_id = 0
        max_item_id = 0
        max_sequence_length = 0

        for sample in data:
            sample = sample.strip("\n").split(" ")
            item_ids = [int(item_id) for item_id in sample[1:]][-max_sample_len:]
            user_id = int(sample[0])

            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, max(item_ids))
            max_sequence_length = max(max_sequence_length, len(item_ids))

            user_sequences.append(user_id)
            item_sequences.append(item_ids)

        return (
            user_sequences,
            item_sequences,
            max_user_id,
            max_item_id,
            max_sequence_length,
        )

    def get_samplers(self):
        return (
            self._train_sampler,
            self._validation_sampler,
            self._test_sampler,
        )


class GraphDataset(BaseDataset, config_name="graph"):
    def __init__(
        self,
        dataset,
        graph_dir_path,
        use_train_data_only=True,
        use_user_graph=False,
        use_item_graph=False,
        neighborhood_size=None,
    ):
        self._dataset = dataset
        self._graph_dir_path = graph_dir_path
        self._use_train_data_only = use_train_data_only
        self._use_user_graph = use_user_graph
        self._use_item_graph = use_item_graph
        self._neighborhood_size = neighborhood_size

        self._num_users = dataset.num_users
        self._num_items = dataset.num_items

        train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

        interactions_data = self._collect_interactions(
            train_sampler, validation_sampler, test_sampler
        )
        train_interactions = interactions_data["train_interactions"]
        train_user_interactions = interactions_data["train_user_interactions"]
        train_item_interactions = interactions_data["train_item_interactions"]
        train_user_2_items = interactions_data["train_user_2_items"]
        train_item_2_users = interactions_data["train_item_2_users"]

        self._train_interactions = np.array(train_interactions)
        self._train_user_interactions = np.array(train_user_interactions)
        self._train_item_interactions = np.array(train_item_interactions)

        self._graph = self._build_or_load_bipartite_graph(
            graph_dir_path, train_user_interactions, train_item_interactions
        )

        self._user_graph = (
            self._build_or_load_similarity_graph(
                "user",
                self._train_user_interactions,
                self._train_item_interactions,
                train_item_2_users,
                train_user_2_items,
            )
            if self._use_user_graph
            else None
        )

        self._item_graph = (
            self._build_or_load_similarity_graph(
                "item",
                self._train_user_interactions,
                self._train_item_interactions,
                train_item_2_users,
                train_user_2_items,
            )
            if self._use_item_graph
            else None
        )

    def _build_or_load_similarity_graph(
        self,
        entity_type,
        train_user_interactions,
        train_item_interactions,
        train_item_2_users,
        train_user_2_items,
    ):
        if entity_type not in ["user", "item"]:
            raise ValueError("entity_type must be either 'user' or 'item'")
        k_suffix = (
            f"k{self._neighborhood_size}"
            if self._neighborhood_size is not None
            else "full"
        )
        train_suffix = "trainOnly" if self._use_train_data_only else "withValTest"
        filename = f"{entity_type}_graph_{k_suffix}_{train_suffix}.npz"
        path_to_graph = os.path.join(self._graph_dir_path, filename)

        is_user_graph = entity_type == "user"
        num_entities = self._num_users if is_user_graph else self._num_items

        if os.path.exists(path_to_graph):
            graph_matrix = sp.load_npz(path_to_graph)
        else:
            R = csr_matrix(
                (
                    np.ones(len(train_user_interactions)),
                    (train_user_interactions, train_item_interactions),
                ),
                shape=(self._num_users + 2, self._num_items + 2),
            )

            if is_user_graph:
                connections = R.dot(R.T)
            else:
                connections = R.T.dot(R)

            connections.setdiag(0)
            connections.eliminate_zeros()

            if self._neighborhood_size is not None:
                connections = self._filter_matrix_by_top_k(
                    connections, self._neighborhood_size
                )

            graph_matrix = self.get_sparse_graph_layer(
                connections, num_entities + 2, num_entities + 2, biparite=False
            )
            sp.save_npz(path_to_graph, graph_matrix)

        return self._convert_sp_mat_to_sp_tensor(graph_matrix).coalesce().to(DEVICE)

    def _build_or_load_bipartite_graph(
        self, graph_dir_path, train_user_interactions, train_item_interactions
    ):
        train_suffix = "trainOnly" if self._use_train_data_only else "withValTest"
        filename = f"general_graph_{train_suffix}.npz"
        path_to_graph = os.path.join(graph_dir_path, filename)

        if os.path.exists(path_to_graph):
            graph_matrix = sp.load_npz(path_to_graph)
        else:
            user2item_connections = csr_matrix(
                (
                    np.ones(len(train_user_interactions)),
                    (train_user_interactions, train_item_interactions),
                ),
                shape=(self._num_users + 2, self._num_items + 2),
            )  # (num_users + 2, num_items + 2), bipartite graph
            graph_matrix = self.get_sparse_graph_layer(
                user2item_connections,
                self._num_users + 2,
                self._num_items + 2,
                biparite=True,
            )
            sp.save_npz(path_to_graph, graph_matrix)

        return self._convert_sp_mat_to_sp_tensor(graph_matrix).coalesce().to(DEVICE)

    def _collect_interactions(self, train_sampler, validation_sampler, test_sampler):
        train_interactions = []
        train_user_interactions, train_item_interactions = [], []

        train_user_2_items = defaultdict(set)
        train_item_2_users = defaultdict(set)
        visited_user_item_pairs = set()

        samplers_to_process = [train_sampler]
        if not self._use_train_data_only:
            samplers_to_process.extend([validation_sampler, test_sampler])

        for sampler in samplers_to_process:
            for sample in sampler.dataset:
                user_id = sample["user.ids"][0]
                for item_id in sample["item.ids"]:
                    pair_key = (user_id << 32) | item_id

                    if pair_key not in visited_user_item_pairs:
                        train_interactions.append((user_id, item_id))

                        train_user_interactions.append(user_id)
                        train_item_interactions.append(item_id)

                        train_user_2_items[user_id].add(item_id)
                        train_item_2_users[item_id].add(user_id)

                        visited_user_item_pairs.add(pair_key)

        return {
            "train_interactions": train_interactions,
            "train_user_interactions": train_user_interactions,
            "train_item_interactions": train_item_interactions,
            "train_user_2_items": train_user_2_items,
            "train_item_2_users": train_item_2_users,
        }

    @classmethod
    def create_from_config(cls, config):
        dataset = BaseDataset.create_from_config(config["dataset"])
        return cls(
            dataset=dataset,
            graph_dir_path=config["graph_dir_path"],
            use_user_graph=config.get("use_user_graph", False),
            use_item_graph=config.get("use_item_graph", False),
            neighborhood_size=config.get("neighborhood_size", None),
        )

    @staticmethod
    def get_sparse_graph_layer(
        sparse_matrix,
        fst_dim,
        snd_dim,
        biparite=False,
    ):
        if not biparite:
            adj_mat = sparse_matrix.tocsr()
        else:
            R = sparse_matrix.tocsr()

            upper_right = R
            lower_left = R.T

            upper_left = sp.csr_matrix((fst_dim, fst_dim))
            lower_right = sp.csr_matrix((snd_dim, snd_dim))

            adj_mat = sp.bmat([[upper_left, upper_right], [lower_left, lower_right]])
            assert adj_mat.shape == (
                fst_dim + snd_dim,
                fst_dim + snd_dim,
            ), f"Got shape {adj_mat.shape}, expected {(fst_dim+snd_dim, fst_dim+snd_dim)}"
        rowsum = np.array(adj_mat.sum(1)).flatten()
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0

        norm_adj = adj_mat.multiply(d_inv[:, np.newaxis]).multiply(d_inv)

        return norm_adj.tocsr()

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo()
        values = torch.from_numpy(coo.data).float()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    @staticmethod
    def _filter_matrix_by_top_k(matrix, k):
        mat = matrix.tocsr()

        for i in range(mat.shape[0]):
            start = mat.indptr[i]
            end = mat.indptr[i + 1]

            if end - start > k:
                row_slice = mat.data[start:end]

                threshold = np.partition(row_slice, -k)[-k]

                row_slice[row_slice < threshold] = 0

        mat.eliminate_zeros()
        return mat

    def get_samplers(self):
        return self._dataset.get_samplers()

    @property
    def meta(self):
        meta = {
            "user_graph": self._user_graph,
            "item_graph": self._item_graph,
            "graph": self._graph,
            **self._dataset.meta,
        }
        return meta


class MCLSRDataset(BaseSequenceDataset, config_name="mclsr"):
    @staticmethod
    def _create_sequences_from_file(filepath, max_len=None):
        sequences = {}
        max_user, max_item = 0, 0

        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                user_id = int(parts[0])
                item_ids = [int(i) for i in parts[1:]]
                if max_len:
                    item_ids = item_ids[-max_len:]
                sequences[user_id] = item_ids
                max_user = max(max_user, user_id)
                if item_ids:
                    max_item = max(max_item, max(item_ids))
        return sequences, max_user, max_item

    @staticmethod
    def _read_train_samples(filepath, max_len=None):
        samples = []
        max_user, max_item = 0, 0

        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                user_id = int(parts[0])
                item_ids = [int(i) for i in parts[1:]]
                if max_len:
                    item_ids = item_ids[-max_len:]
                samples.append((user_id, item_ids))
                max_user = max(max_user, user_id)
                if item_ids:
                    max_item = max(max_item, max(item_ids))
        return samples, max_user, max_item

    @classmethod
    def _create_evaluation_sets(cls, data_dir, max_seq_len):
        valid_hist, u2, i2 = cls._create_sequences_from_file(
            os.path.join(data_dir, "valid_history.txt"), max_seq_len
        )
        valid_trg, u3, i3 = cls._create_sequences_from_file(
            os.path.join(data_dir, "valid_target.txt")
        )

        validation_dataset = [
            {"user.ids": [uid], "history": valid_hist[uid], "target": valid_trg[uid]}
            for uid in valid_hist
            if uid in valid_trg
        ]

        test_hist, u4, i4 = cls._create_sequences_from_file(
            os.path.join(data_dir, "test_history.txt"), max_seq_len
        )
        test_trg, u5, i5 = cls._create_sequences_from_file(
            os.path.join(data_dir, "test_target.txt")
        )

        test_dataset = [
            {"user.ids": [uid], "history": test_hist[uid], "target": test_trg[uid]}
            for uid in test_hist
            if uid in test_trg
        ]

        return (
            validation_dataset,
            test_dataset,
            max(u2, u3, u4, u5),
            max(i2, i3, i4, i5),
        )

    @classmethod
    def create_from_config(cls, config, **kwargs):
        data_dir = os.path.join(config["path_to_data_dir"], config["name"])
        max_seq_len = config.get("max_sequence_length")

        train_samples, u1, i1 = cls._read_train_samples(
            os.path.join(data_dir, "train_mclsr.txt"), max_seq_len
        )
        train_dataset = [
            {
                "user.ids": [uid],
                "user.length": 1,
                "item.ids": seq,
                "item.length": len(seq),
            }
            for uid, seq in train_samples
        ]

        user_to_all_seen_items = defaultdict(set)
        for sample in train_dataset:
            user_to_all_seen_items[sample["user.ids"][0]].update(sample["item.ids"])
        kwargs["user_to_all_seen_items"] = user_to_all_seen_items

        validation_dataset, test_dataset, u_eval, i_eval = cls._create_evaluation_sets(
            data_dir, max_seq_len
        )
        num_users = max(u1, u_eval)
        num_items = max(i1, i_eval)

        train_sampler = TrainSampler.create_from_config(
            config["samplers"],
            dataset=train_dataset,
            num_users=num_users,
            num_items=num_items,
            **kwargs,
        )
        validation_sampler = EvalSampler.create_from_config(
            config["samplers"],
            dataset=validation_dataset,
            num_users=num_users,
            num_items=num_items,
            **kwargs,
        )
        test_sampler = EvalSampler.create_from_config(
            config["samplers"],
            dataset=test_dataset,
            num_users=num_users,
            num_items=num_items,
            **kwargs,
        )

        return cls(
            train_sampler,
            validation_sampler,
            test_sampler,
            num_users,
            num_items,
            max_seq_len,
        )
