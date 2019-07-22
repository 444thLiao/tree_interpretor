from pandas  import DataFrame
import numpy as np
import tqdm
import pandas as pd
from collections import Counter
def parsed_dump_files(model_file_path):
    dump_str = open(model_file_path).read()
    boosters = dump_str.split('booster')
    boosters = [_ for _ in boosters if _]
    boosters = [_.partition('\n')[2] for _ in boosters]
    return boosters
def get_dumps_xgb_model(str_dump):
    tree_arr = []
    for i_tree, tree in enumerate(str_dump):
        arr_lvls = tree.split('\n\t')
        a_tree = {}
        for lvl in arr_lvls:
            a_lvl = {}
            dum1 = lvl.split(',')
            if ('leaf' in lvl):
                dum1[0].replace('\t', '')
                dum10 = dum1[0].split(':')
                lvl_id = int(dum10[0])
                dum11 = dum10[1].split('leaf=')
                leaf = float(dum11[1])

                cover = float(dum1[1].replace('\n', '').split('cover=')[1])
                a_lvl['lvl_id'] = lvl_id
                a_lvl['leaf'] = leaf
                a_lvl['cover'] = cover
            else:
                dum10 = dum1[0].replace('\t', '').replace('\n', '')
                dum11 = dum10.split(':')
                lvl_id = int(dum11[0])
                dum12 = dum11[1].split('yes=')
                dum13 = dum12[0].replace('[', '').replace(']', '').split('<')
                feat_name = dum13[0]

                yes_to = int(dum12[1])
                no_to = int(dum1[1].split('no=')[1])
                missing = int(dum1[2].split('missing=')[1])
                gain = float(dum1[3].split('gain=')[1])
                cover = float(dum1[4].split('cover=')[1])
                feat_thr = float(dum13[1])

                a_lvl['lvl_id'] = lvl_id
                a_lvl['feat_name'] = feat_name
                a_lvl['feat_thr'] = feat_thr
                a_lvl['yes_to'] = yes_to
                a_lvl['no_to'] = no_to
                a_lvl['missing'] = missing
                a_lvl['gain'] = gain
                a_lvl['cover'] = cover

            a_tree[lvl_id] = a_lvl
        tree_arr.append(a_tree)
    return tree_arr

def one_data_path(subset,tree,cur = 0):

    if not tree[cur].get('feat_name'):
        return [cur]
    if subset[tree[cur]['feat_name']] < tree[cur]['feat_thr']:
        cur_new = tree[cur]['yes_to']
    else:
        cur_new = tree[cur]['no_to']
    return [cur] + one_data_path(subset,tree,cur = cur_new)

def parsed_data_into_node(X_train,tree):
    if type(X_train) != DataFrame:
        raise ValueError
    all_paths = []
    for idx in range(X_train.shape[0]):
        new_subset = X_train.iloc[idx,:]
        if len(tree.keys()) == 1:
            this_path = [0]
        else:
            this_path = one_data_path(new_subset,tree)
        all_paths.append(this_path)
    return np.array(all_paths)

def parse_all_tree(tree_arr,X_train):
    """
    list of trees.

    :param tree_arr:
    :param X_train:
    :return:
    """
    all_tree_paths = []
    for tree in tqdm.tqdm(tree_arr):
        all_tree_paths.append(parsed_data_into_node(X_train,tree))
    return all_tree_paths

if __name__ == '__main__':
    X_train = pd.read_csv('../example.csv')
    str_dump = parsed_dump_files(
        '/home/liaoth/data2/16s/shandong/16s_pipelines/v_analysis_dechimera/ML_haokui/output_model/models/model_0')
    tree_arr = get_dumps_xgb_model(str_dump)
    all_tree_paths = parse_all_tree(tree_arr, X_train)
    for result in all_tree_paths[:10]:
        print(Counter([';'.join([str(i) for i in _]) for _ in result]))
