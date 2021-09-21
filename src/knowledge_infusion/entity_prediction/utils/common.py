import argparse
import gzip
import os
import random
import time
import zipfile
from functools import wraps

import metis
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tabulate import tabulate
from tqdm.auto import tqdm


def _construct_adj(edges, num_data):
    adj = sp.csr_matrix(
        (np.ones((edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(num_data, num_data),
    )
    adj += adj.transpose()
    return adj


def partition_graph(adj, idx_nodes, num_clusters):
    """partition a graph by METIS."""

    start_time = time.time()
    num_nodes = len(idx_nodes)
    num_all_nodes = adj.shape[0]

    neighbor_intervals = []
    neighbors = []
    edge_cnt = 0
    neighbor_intervals.append(0)
    train_adj_lil = adj[idx_nodes, :][:, idx_nodes].tolil()
    train_ord_map = dict()
    train_adj_lists = [[] for _ in range(num_nodes)]
    for i in tqdm(range(num_nodes)):
        rows = train_adj_lil[i].rows[0]
        # self-edge needs to be removed for valid format of METIS
        if i in rows:
            rows.remove(i)
        train_adj_lists[i] = rows
        neighbors += rows
        edge_cnt += len(rows)
        neighbor_intervals.append(edge_cnt)
        train_ord_map[idx_nodes[i]] = i

    if num_clusters > 1:
        _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    else:
        groups = [0] * num_nodes

    part_row = []
    part_col = []
    part_data = []
    parts = [[] for _ in range(num_clusters)]
    for nd_idx in tqdm(range(num_nodes)):
        gp_idx = groups[nd_idx]
        nd_orig_idx = idx_nodes[nd_idx]
        parts[gp_idx].append(nd_orig_idx)
        for nb_orig_idx in adj[nd_orig_idx].indices:
            nb_idx = train_ord_map[nb_orig_idx]
            if groups[nb_idx] == gp_idx:
                part_data.append(1)
                part_row.append(nd_orig_idx)
                part_col.append(nb_orig_idx)
    part_data.append(0)
    part_row.append(num_all_nodes - 1)
    part_col.append(num_all_nodes - 1)
    part_adj = sp.coo_matrix((part_data, (part_row, part_col))).tocsr()

    print("Partitioning done. %f seconds.", time.time() - start_time)
    return part_adj, parts


def save_to_csv(result, result_file):
    """
    Save a result dict to disk.

    Args:
        result: The result dict to be saved.
        result_file: The file path to be saved.

    Returns:
        None

    """
    for k, v in result.items():
        result[k] = [v]
    result_df = pd.DataFrame(result)
    if os.path.exists(result_file):
        print(result_file, " already exists, appending result to it")
        total_result = pd.read_csv(result_file)
        total_result = total_result.append(result_df)
    else:
        print("Create new result_file:", result_file)
        total_result = result_df
    total_result.to_csv(result_file, index=False)


def normalized_adj_single(adj):
    """Missing docs.

    Args:
        adj:

    Returns:
        None.
    """
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    # norm_adj = adj.dot(d_mat_inv)
    print("generate single-normalized adjacency matrix.")
    return norm_adj.tocoo()


def ensureDir(dir_path):
    """Ensure a dir exist, otherwise create the path.

    Args:
        dir_path (str): the target dir.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def update_args(config, args):
    """Update config parameters by the received parameters from command line.

    Args:
        config (dict): Initial dict of the parameters from JSON config file.
        args (object): An argparse Argument object with attributes being the parameters to be updated.
    """
    args_dic = {}
    for cfg in ["system", "model"]:
        for k, v in vars(args).items():
            if v is not None and k in config[cfg]:
                config[cfg][k] = v
                args_dic[f"{cfg}:{k}"] = v
    print_dict_as_table(args_dic, "Received parameters from command line (or default):")


def parse_gzip_file(path):
    """Parse gzip file.

    Args:
        path: the file path of gzip file.
    """
    g = gzip.open(path, "rb")
    for l in g:
        yield eval(l)


def get_data_frame_from_gzip_file(path):
    """Get dataframe from a gzip file.

    Args:
        path the file path of gzip file.

    Returns:
        A dataframe extracted from the gzip file.
    """
    i = 0
    df = {}
    for d in parse_gzip_file(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def un_zip(file_name, target_dir=None):
    """Unzip zip files.

    Args:
        file_name (str or Path): zip file path.
        target_dir (str or Path): target path to be save the unzipped files.
    """
    if target_dir is None:
        target_dir = os.path.dirname(file_name)
    zip_file = zipfile.ZipFile(file_name)
    for names in zip_file.namelist():
        print(f"unzip file {names} ...")
        zip_file.extract(names, target_dir)
    zip_file.close()


def print_dict_as_table(dic, tag=None, columns=["keys", "values"]):
    """Print a dictionary as table.

    Args:
        dic (dict): dict object to be formatted.
        tag (str): A name for this dictionary.
        columns ([str,str]):  default ["keys", "values"]. columns name for keys and values.

    Returns:
        None
    """
    print("-" * 80)
    if tag is not None:
        print(tag)
    df = pd.DataFrame(dic.items(), columns=columns)
    print(tabulate(df, headers=columns, tablefmt="psql"))
    print("-" * 80)
    return tabulate(df, headers=columns, tablefmt="psql")


def print_args_as_table(args, tag=None, columns=["keys", "values"]):
    """Print a ArgumentParser as table.

    Args:
        dic (dict): dict object to be formatted.
        tag (str): A name for this dictionary.
        columns ([str,str]):  default ["keys", "values"]. columns name for keys and values.

    Returns:
        None
    """
    return print_dict_as_table(vars(args), tag, columns)


class DictToObject(object):
    """Python dict to object."""

    def __init__(self, dictionary):
        """Initialize DictToObject Class."""

        def _traverse(key, element):
            if isinstance(element, dict):
                return key, DictToObject(element)
            else:
                return key, element

        objd = dict(_traverse(k, v) for k, v in dictionary.items())
        self.__dict__.update(objd)


def get_random_rep(raw_num, dim):
    """Generate a random embedding from a normal (Gaussian) distribution.

    Args:
        raw_num: Number of raw to be generated.
        dim: The dimension of the embeddings.
    Returns:
        ndarray or scalar.
        Drawn samples from the normal distribution.
    """
    return np.random.normal(size=(raw_num, dim))


def timeit(method):
    """Generate decorator for tracking the execution time for the specific method.

    Args:
        method: The method need to timeit.

    To use:
        @timeit
        def method(self):
            pass
    Returns:
        None
    """

    @wraps(method)
    def wrapper(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print(
                "Execute [{}] method costing {:2.2f} ms".format(
                    method.__name__, (te - ts) * 1000
                )
            )
        return result

    return wrapper


def set_seed(seed):
    """Initialize all the seed in the system.

    Args:
        seed: A global random seed.
    """
    if type(seed) != int:
        raise ValueError("Error: seed is invalid type")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(v):
    """Convert a string to a bool variable."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
