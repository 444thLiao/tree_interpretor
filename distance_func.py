import pandas as pd
import itertools
import tqdm
from scipy.spatial.distance import squareform,pdist

def fetch_last_node(decision_path_multi_tres):
    new_trees_pack = []
    for tree in decision_path_multi_tres:
        tree_of_sampels = [sample_path[-1] for sample_path in tree]
        new_trees_pack.append(tree_of_sampels)
    return new_trees_pack

def proximity_distance(leaf_nodes_of_trees):
    """
    Format of dicesion_path_multi_tres need same as 'parse_all_tree' of for_xgboost module.

    Take 2:11s for 1675 samples and 6079 tree's forest.
    :param data:
    :param decision_path_multi_tres:
    :return:
    """
    num_s = len(leaf_nodes_of_trees[0])
    distance = np.zeros(( num_s , num_s))
    all_sn = range(num_s)
    total_val_trees_num = float(len(leaf_nodes_of_trees))
    for tree in tqdm.tqdm(leaf_nodes_of_trees):
        poss_leafs = set(tree)
        poss_leafs = [np.where(np.array(tree) == poss_leaf)[0] for poss_leaf in poss_leafs]
        if len(poss_leafs) != 1:
            for samples in poss_leafs:
                for a,b in itertools.combinations(samples,r=2):
                    distance[a,b] += 1.0
                    distance[b,a] += 1.0
        else:
            distance += 1.0
    for idx in all_sn:
        distance[idx,idx] = total_val_trees_num

    distance /= total_val_trees_num
    distance = 1 - distance
    return distance

def expand_node(leaf_nodes_of_trees,metrics = 'euclidean'):
    """
    Take 00:25s for 1675 samples and 6079 tree's forest.
    :param leaf_nodes_of_trees:
    :param metrics:
    :return:
    """
    distance_along_y = []
    for tree in tqdm.tqdm(leaf_nodes_of_trees):
        packet = sorted(set(tree))
        distance_along_x = []
        for leaf_s in tree:
            init_tmp = np.zeros(len(packet))
            init_tmp[packet.index(leaf_s)] = 1
            distance_along_x.append(init_tmp)
        distance_along_y.append(np.array(distance_along_x))

    final_distance = np.concatenate(distance_along_y,axis=1)
    distance = squareform(pdist(final_distance, metric=metrics))
    return distance

def cal_dis_tmp(leaf_nodes_of_trees,tree_arr,metrics = 'euclidean'):
    """
    Take 00:05s for 1675 samples and 6079 tree's forest.

    only for xgboost.
    :param leaf_nodes_of_trees:
    :param metrics:
    :return:
    """
    num_s = len(leaf_nodes_of_trees[0])
    new_x_data = np.zeros((num_s,len(leaf_nodes_of_trees)))
    for idx,tree_ in tqdm.tqdm(enumerate(leaf_nodes_of_trees),total=len(leaf_nodes_of_trees)):
        cur_tree = tree_arr[idx]
        new_x_data[:,idx] = [cur_tree[leaf_num]['leaf'] for leaf_num in tree_]
    distance = squareform(pdist(new_x_data, metric=metrics))
    return distance


if __name__ == '__main__':
    from for_xgboost.parsed_dum_models import *
    str_dump = parsed_dump_files(
        '/home/liaoth/data2/16s/shandong/16s_pipelines/v_analysis_dechimera/ML_haokui/output_model/models/model_0')
    data_ = pd.read_csv('/home/liaoth/data2/project/Parsed_tree_model/example/data.csv')
    tree_arr = get_dumps_xgb_model(str_dump)
    all_tree_paths = parse_all_tree(tree_arr, data_)

    leaf_nodes_of_trees = fetch_last_node(all_tree_paths)
    # d1 = proximity_distance(leaf_nodes_of_trees)
    d2 = expand_node(leaf_nodes_of_trees)
    d3 = cal_dis_tmp(leaf_nodes_of_trees,tree_arr)
    import plotly
    import plotly.graph_objs as go
    from skbio.stats.ordination import pcoa
    vals2 = pcoa(d2).samples.values
    vals3 = pcoa(d3).samples.values

    metadata = pd.read_csv('/home/liaoth/data2/16s/shandong/16s_pipelines/v_analysis_dechimera/merged_metadata_fixed_180120.tab', sep='\t', header=0, index_col=0)
    groups_dict = {}
    # base on metadata to subtype
    for sample in list(X_train.index):
        if sample in list(metadata.index):
            g_info = metadata.loc[sample, 'host_status']
            if len(g_info) == 2:
                print(g_info)
                continue
            # g_info += '_' + metadata.loc[sample, 'lib_name'] + '_' + metadata.loc[sample, 'organism/tissue']
            if metadata.loc[sample, 'organism/tissue'] == 'Saliva':
                continue
            if g_info in groups_dict.keys():
                groups_dict[g_info].append(list(X_train.index).index(sample))
            else:
                groups_dict[g_info] = [list(X_train.index).index(sample)]
        else:
            # print sample
            pass

    draw_data = []
    for idx, group in enumerate(groups_dict.keys()):
        if 'diabetes' in group:
            draw_data.append(go.Scatter(x=vals3[groups_dict[group], 0], y=vals3[groups_dict[group], 1],
                                        marker=dict(size=10, symbol='diamond'),
                                        textfont=dict(size=20),
                                        mode='markers',
                                        hoverinfo='text',
                                        name='%s(num:%s)' % (group, len(groups_dict[group]))))
        elif 'Health' in group:
            draw_data.append(go.Scatter(x=vals3[groups_dict[group], 0], y=vals3[groups_dict[group], 1],
                                        marker=dict(size=10, ),
                                        textfont=dict(size=20),
                                        mode='markers',
                                        hoverinfo='text',
                                        name='%s(num:%s)' % (group, len(groups_dict[group]))))
    layout = go.Layout(
        title='MDS projection',
        font=dict(size=20), width=1460, height=770)
    plotly.offline.plot(dict(data=draw_data, layout=layout),
                        filename='/home/liaoth/data2/project/Parsed_tree_model/example/pcoa_with_new_distance_d3.html')
