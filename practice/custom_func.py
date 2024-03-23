import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as xticks
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import Binarizer


def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)  # 오차행렬
    accuracy = accuracy_score(y_test, pred)  # 정확도
    precision = precision_score(y_test, pred)  # 정밀도
    recall = recall_score(y_test, pred)  # 재현율
    f1 = f1_score(y_test, pred)  # F1 score
    roc_auc = roc_auc_score(y_test, pred_proba)  # ROC AUC

    print("오차 행렬")
    print(confusion)
    print("정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1: {3:.4f}, AUC: {4:.4f}".format(
        accuracy, precision, recall, f1, roc_auc))


def precision_recall_curve_plot(y_test, pred_proba_c1):
    # threshold에 따른 정밀도, 재현율 추출
    precisions, recalls, thresholds = precision_recall_curve(
        y_test, pred_proba_c1)

    # x축: threshold
    # y축: 정밀도, 재현율
    # 정밀도는 점선으로 표현
    plt.figure(figsize=(8, 6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary],
             linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')

    # x축의 scale을 0.1 단위로 설정
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    plt.xlabel("Treshold value")
    plt.ylabel("Precision and recall")
    plt.legend()
    plt.grid()
    plt.show()


def get_eval_by_threshold(y_test, pred_proba, threshold):
    pred_proba_c1 = pred_proba[:, 1].reshape(-1, 1)

    for custom_threshold in threshold:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print("임계값: ", custom_threshold)
        get_clf_eval(y_test, custom_predict, pred_proba_c1)
        print()

    precision_recall_curve_plot(y_test, pred_proba_c1)
