import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

"""Calcola e stampa i PPV per le varie percentuali di top selected features e stampa la ROC associata"""
def print_ppv_hist_ROC(predictions, name):
    print ""
    print "PPV of Top 1% of predictions"
    a = get_ppv(predictions, 0.01)
    print str(a)
    print ""
    print "PPV of Top 5% of predictions"
    b = get_ppv(predictions, 0.05)
    print str(b)
    print ""
    print "PPV of Top 10% of predictions"
    c = get_ppv(predictions, 0.1)
    print str(c)
    print ""
    print "PPV of Top 15% of predictions"
    d = get_ppv(predictions, 0.15)
    print str(d)
    print ""
    print "PPV of Top 20% of predictions"
    e = get_ppv(predictions, 0.2)
    print str(e)
    print ""
    print_histogram(a, b, c, d, e, name)
    print_ROC(predictions, name)

"""Calcola il ppv"""
def get_ppv(pred, m):
    n = int(len(pred)*m)
    num = sum([p[1][1] for p in pred[:n]])
    ppv = float(num)/float(n)
    return ppv*100

"""Stampa l'istogramma"""
def print_histogram(a, b, c, d, e, name):
    y = [a, b, c, d, e]
    x = [0.5, 4.5, 9.5, 14.5, 19.5]
    if name == "cerevisiae":
        plt.bar(x, y, width=1, color='#0060C0')
        plt.title('S. Cerevisiae')
    else:
        plt.bar(x, y, width=1, color='#C70303')
        plt.title('E. Coli')

    plt.xlabel('Top percentage of predictions')
    plt.ylabel('PPV (%)')
    plt.show()

"""Stampa la ROC curve"""
def print_ROC(predictions, name):
    y_true = []
    y_predicted = []
    for j in predictions:
        y_predicted.append(j[1][0])
        y_true.append(j[1][1])
    # Compute ROC curve and ROC area for each class
    y_true = np.reshape(y_true, (len(y_true), 1))
    fpr, tpr, thres = roc_curve(y_true, np.array(y_predicted))
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = 'ROC curve for ' + str(name)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()