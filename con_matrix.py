

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn



def con_matrix(Y_test,y_hat):
    '''
    func:输出矩阵的召回率以及精确度，并且构造混淆矩阵的图像
    
    paramas: 
    Y_test:输入Y的实际值
    y_hat：输入Y的预测值
    
    '''
    print('recall:',recall_score(Y_test,y_hat))
    print('precision:',precision_score(Y_test,y_hat))
    cm = confusion_matrix(Y_test,y_hat,labels=[1,0])
    print('confusion matrix',cm)
    sn.heatmap(cm, annot=True,xticklabels=[1,0],yticklabels=[1,0])
    plt.show()  

