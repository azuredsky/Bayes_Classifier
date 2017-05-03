import numpy as np
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold='nan')

"""Costruzione della matrice di features a partire dal file di excel.
    I valori di ritorno sono la matrice di features (comprensiva delle labels associate) e il vettore dei nomi"""
def get_matrix_from_sheet(wb):
    entropy_discretized = wb.sheet_by_index(1)
    n_rows = entropy_discretized.nrows - 1
    n_cols = entropy_discretized.ncols - 1
    names = []
    a = np.zeros((n_rows, n_cols), dtype=np.int)
    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            if entropy_discretized.cell_type(row, col) == 1:
                if ">" in entropy_discretized.cell(row, col).value:
                    a[row-1, col-1] = 0
                elif "<=" in entropy_discretized.cell(row, col).value:
                    a[row-1, col-1] = 1
                elif "(" in entropy_discretized.cell(row,col).value:
                    a[row-1, col-1] = 2
            elif entropy_discretized.cell_type(row, col) == 2:
                    a[row-1, col-1] = entropy_discretized.cell(row, col).value
        names.append(entropy_discretized.cell(row, 0).value.encode('utf8'))
    for col in range(0, a.shape[1]):
        val = np.unique(a[:, col])
        if 0 not in val:
            for j in range(a.shape[0]):
                value = a[j, col]
                a[j, col] = value - 1
    names = np.array(names)
    names = np.reshape(names, (a.shape[0], 1))
    return a, names

"""Suddivide le features in essenziali e non essenziali.
    I valori di ritorno sono due matrici di features (suddivise per essenzialita', escluse le labels) e i relativi
    vettori di nomi"""
def split_matrix(features, nam):
    ess_number = essential_length(features)
    cols = features.shape[1]
    essential = np.zeros((ess_number, cols), dtype=np.int)
    ess_name = []
    non_essential = np.zeros((features.shape[0]-ess_number, cols), dtype=np.int)
    non_ess_name = []
    count_essential = 0
    count_nessential = 0
    for i in range(features.shape[0]):
        if int(features[i, 0]) == 1:
            essential[count_essential] = features[i]
            count_essential += 1
            ess_name.append(nam[i])
        else:
            non_essential[count_nessential] = features[i]
            count_nessential += 1
            non_ess_name.append(nam[i])
    return np.delete(essential, 0, 1), np.array(ess_name), np.delete(non_essential, 0, 1), np.array(non_ess_name)

"""Suddivide il set in train e test set"""
def split_set(es_X, es_name, n_es_X, n_es_name):
    test_size = 0.5

    es_y = np.ones((es_X.shape[0], 1), dtype=np.int)
    es_X, es_y, es_name = shuffle_in_unison(es_X, es_y, np.array(es_name))
    n_es_y = np.zeros((n_es_X.shape[0], 1), dtype=np.int)
    n_es_X, n_es_y, n_es_name = shuffle_in_unison(n_es_X, n_es_y, np.array(n_es_name))

    es_train, es_test, es_train_labels, es_test_labels, es_name_train, es_name_test = \
        train_test_split(es_X, es_y, es_name, test_size=test_size)
    n_es_train, n_es_test, n_es_train_labels, n_es_test_labels, n_es_name_train, n_es_name_test = \
        train_test_split(n_es_X, n_es_y, n_es_name, test_size=test_size)

    train_set = np.concatenate((es_train, n_es_train), axis=0)
    test_set = np.concatenate((es_test, n_es_test), axis=0)
    train_labels = np.concatenate((es_train_labels, n_es_train_labels), axis=0)
    train_labels = train_labels.reshape((train_labels.shape[0], 1))
    test_labels = np.concatenate((es_test_labels, n_es_test_labels), axis=0)
    test_labels = test_labels.reshape((test_labels.shape[0], 1))
    train_name = np.concatenate((es_name_train, n_es_name_train), axis=0)
    train_name = train_name.reshape((train_name.shape[0], 1))
    test_name = np.concatenate((es_name_test, n_es_name_test), axis=0)
    test_name = test_name.reshape((test_name.shape[0], 1))

    train_set, train_labels, train_name  = shuffle_in_unison(train_set, train_labels, train_name)
    test_set, test_labels, test_name = shuffle_in_unison(test_set, test_labels, test_name)

    return train_set, train_labels, train_name, test_set, test_labels, test_name

"""Il valore di ritorno e' il numero di geni essenziali"""
def essential_length(set): #return number of essential genes
    count = 0
    for i in set[:, 0]:
        if i == 1:
            count += 1
    return count

"""Esegue lo shuffle all'unisono delle matrici in ingresso restituendole in uscita"""
def shuffle_in_unison(a, b, c):
    assert len(a) == len(b) and len(a) == len(c)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    shuffled_c = np.empty(c.shape, dtype=c.dtype)

    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        shuffled_c[new_index] = c[old_index]
    return shuffled_a, shuffled_b, shuffled_c

"""Utilizzato per eseguire la feature selection"""
def select_features(matrix, name):
    num_of_feature = 5
    if name == 'coli':
        features = [0, 1, 2, 12, 24]
    else:
        features = [0, 3, 4, 6, 36]

    set = np.zeros((matrix.shape[0], num_of_feature), dtype=np.int)
    for i in range(num_of_feature):
        set[:, i] = matrix[:, features[i]]
    return set