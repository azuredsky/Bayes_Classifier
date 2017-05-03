import xlrd
import utility
import Naive_Bayes
import operator
import predict_functions
from timeit import default_timer as timer

def start_function(name, feature_selection):
    if name == 'coli':
        file_name = 'coli.xls'
    else:
        file_name = 'cerevisiae.xls'

    wb = xlrd.open_workbook(file_name)
    final_mat, nam = utility.get_matrix_from_sheet(wb)
    y = final_mat[:, 0]

    if feature_selection:
        final_mat = utility.select_features(final_mat, name)

    nb = Naive_Bayes.Naive_Bayes(final_mat)
    es_X, es_name, n_es_X, n_es_name = utility.split_matrix(final_mat, nam)

    pred_dic = {}  # tupla (somma predizioni, numero di volte in cui e' stato nel test_set, label)

    for j in range(y.shape[0]):
        pred_dic[nam[j][0]] = (0, 0, y[j])

    n = 100
    total_time = timer()
    for i in range(n):
        time = timer()
        print ""
        print "Start training for iteration ", int(i+1)
        train_set, train_labels, train_name, test_set, test_labels, test_name = \
            utility.split_set(es_X, es_name, n_es_X, n_es_name)
        nb.train(train_set, train_labels)
        print "End training"
        print ""
        print "Start classification"
        predictions = nb.classify(test_set, test_name)
        end_time = timer() - time
        print "End classification"
        print ""
        print "Execution time: ", end_time
        for key in predictions:
            pred_dic[key] = (pred_dic[key][0] + predictions[key], pred_dic[key][1] + 1, pred_dic[key][2])
    total_time = timer() - total_time
    print ""
    print ""
    print "Total execution time: ", total_time

    for s in pred_dic:
        if pred_dic[s][1] != 0:
            pred_dic[s] = (pred_dic[s][0] / pred_dic[s][1], pred_dic[s][2])

    predictions = sorted(pred_dic.items(), key=operator.itemgetter(1), reverse=True)

    predict_functions.print_ppv_hist_ROC(predictions, name)

