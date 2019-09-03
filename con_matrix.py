

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


# In[ ]:


def con_matrix(Y_test,y_hat):
    print('accuracy:',accuracy_score(Y_test,y_hat))
    print('precision:',precision_score(Y_test,y_hat))
    cm = confusion_matrix(Y_test,y_hat,labels=[1,0])
    print('confusion matrix',cm)
    sn.heatmap(cm, annot=True,xticklabels=[1,0],yticklabels=[1,0])
    plt.show()  

