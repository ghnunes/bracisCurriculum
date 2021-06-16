import numpy as np
import os
import pandas as pd
import pickle
import pyhard_measures


ih_kwargs = {'kDN': {'k': 5, 'distance': 'minkowski'},
             'MV': {},
             'CB': {},
             'N1': {},
             'N2': {'distance': 'minkowski'},
             'F1': {},
             'F2': {},
             'F3': {},
             'F4': {},
             'LSC': {'distance': 'minkowski'},
             'LSR': {'distance': 'minkowski'},
             'Harmfulness': {'distance': 'minkowski'},
             'Usefulness': {'distance': 'minkowski'},
             'CLD': {},
             'DCP': {}}


def ih_exists(dataset, ih_name, transfer_net='inception', alternative_data_dir="."):
    """
    Tests if a given instance hardness measure was already calculates and stored
    Parameters
    ----------
    dataset: analysed dataset
    ih_name: name of the instance hardness measure being used
    transfer_net: network used to extract features from the data
    alternative_data_dir: default path to data directory

    Returns
    -------
    test result as boolean
    """

    if dataset is None:
        data_dir = alternative_data_dir
    else:
        data_dir = dataset.data_dir

    ih_train_path = os.path.join(data_dir, '{}_'.format(dataset.name) + transfer_net
                                 + '_{}'.format(ih_name) + '_train_values.pkl')
    ih_test_path = os.path.join(data_dir, '{}_'.format(dataset.name) + transfer_net
                                + '_{}'.format(ih_name) + '_test_values.pkl')

    return os.path.exists(ih_train_path) and os.path.exists(ih_test_path)


def get_ih_scores(transfer_values_train, y_train, transfer_values_test, y_test,
                  dataset, ih_name, transfer_net='inception',
                  alternative_data_dir="."):
    """
    Gets the instance hardness measure of the data
    Parameters
    ----------
    transfer_values_train: extracted features of train set
    y_train: labels of test set
    transfer_values_test: extracted features of train set
    y_test: labels of test set
    dataset: analysed dataset
    ih_name: name of the instance hardness measure being used
    transfer_net: network used to extract features from the data
    alternative_data_dir: default path to data directory

    Returns
    -------
    hardness measures of the train and test datasets
    """

    if dataset is None:
        data_dir = alternative_data_dir
    else:
        data_dir = dataset.data_dir

    ih_train_path = os.path.join(data_dir, '{}_'.format(dataset.name) + transfer_net
                                 + '_{}'.format(ih_name) + '_train_values.pkl')
    ih_test_path = os.path.join(data_dir, '{}_'.format(dataset.name) + transfer_net
                                + '_{}'.format(ih_name) + '_test_values.pkl')

    if not os.path.exists(ih_train_path) or not os.path.exists(ih_test_path):
        # If score do not already exist, calculate and save them
        train_scores, test_scores = ih_scores(transfer_values_train, y_train,
                                              transfer_values_test, y_test,
                                              ih_name)

        with open(ih_train_path, 'wb') as file_pi:
            pickle.dump(train_scores, file_pi)

        with open(ih_test_path, 'wb') as file_pi:
            pickle.dump(test_scores, file_pi)
    else:
        # If the scores already exist, load them
        with open(ih_train_path, 'rb') as file_pi:
            train_scores = pickle.load(file_pi)

        with open(ih_test_path, 'rb') as file_pi:
            test_scores = pickle.load(file_pi)

    return train_scores, test_scores


def ih_scores(transfer_values_train, y_train, transfer_values_test, y_test, ih_name):
    """
    Calculates the instance hardness measure of the data
    Parameters
    ----------
    transfer_values_train: extracted features of train set
    y_train: labels of test set
    transfer_values_test: extracted features of train set
    y_test: labels of test set
    ih_name: name of the instance hardness measure being used

    Returns
    -------

    """

    # Converting features in a pandas dataframe
    train_df = pd.concat([pd.DataFrame(transfer_values_train), pd.DataFrame({'label': y_train})], axis=1)
    test_df = pd.concat([pd.DataFrame(transfer_values_test), pd.DataFrame({'label': y_test})], axis=1)
    #print(train_df)

    # Creating measures class
    train_measures = pyhard_measures.Measures(train_df, labels_col='label')
    #print(train_measures)
    test_measures = pyhard_measures.Measures(test_df, labels_col='label')

    # Calculating ih measures for train and test datasets
    train_scores = train_measures._call_method(pyhard_measures._measures_dict.get(ih_name),
                                               **ih_kwargs.get(ih_name, {}))
    test_scores = test_measures._call_method(pyhard_measures._measures_dict.get(ih_name),
                                             **ih_kwargs.get(ih_name, {}))

    return train_scores, test_scores

#mudei aqui
def rank_data_according_to_ih(hardness_score, reverse=True, random=False):

    """ 
    Ranks the data according to instance hardness scores
    ----------
    train_scores: instance hardness scores of the data
    y_train: labels of the data
    reverse
        False: if larger scores indicate easier examples
        True: if larger scores indicate harder examples
    random: if to randomize the order of the scores

    Returns
    -------
    res: list of index of the data in order of increasing difficulty
    """

    # train_size, _ = hardness_score.shape
    res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))
    

    if reverse:
        res = np.flip(res, 0)
        with open('output/dificuldadesCurriculum-17-DCP.txt', 'w') as f:
            for item in res:
                f.write("%s\n" % hardness_score[item])
        # with open('output/ordss.txt', 'w') as f:
        #     for item in res:
        #         f.write("%s\n" % item)

    if random:
        np.random.shuffle(res)
        with open('output/dificuldadesRandom-17-DCP.txt', 'w') as f:
            for item in res:
                f.write("%s\n" % hardness_score[item])
    return res
