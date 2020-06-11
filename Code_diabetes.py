
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix


# # Data Analytics

# In[2]:

diabetesDF = pd.read_csv('diabetes.csv')
diabetesDF.head()


# In[3]:

diabetesDF.info()


# In[4]:

plt.figure(figsize=(15,10))
sns.set()
sns.heatmap(diabetesDF.corr(),cmap='coolwarm',annot=True)
plt.show()


# In[5]:

sns.countplot(diabetesDF['Outcome'])


# # Preparation dataset model

# In[6]:

dfTrain = diabetesDF[:650]
dfTest = diabetesDF[650:750]
dfCheck = diabetesDF[750:]


# In[7]:

trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome',1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome',1))


# In[8]:

#Normalizar
means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)


trainData = (trainData - means)/stds
testData = (testData - means)/stds


# In[9]:

diabetesCheck = LogisticRegression(C=100)
diabetesCheck.fit(trainData, trainLabel)


# In[10]:

accuracy = diabetesCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")


# In[11]:

lr_predicted = diabetesCheck.predict(testData)
confusion = confusion_matrix(testLabel, lr_predicted)

print('Logistic regression classifier (default settings)\n', confusion)


# In[12]:

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Accuracy = TP + TN / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
# F1 = 2 * Precision * Recall / (Precision + Recall) 
#print('Accuracy: {:.2f}'.format(accuracy_score(testLabel, lr_predicted)))
print('Precision: {:.2f}'.format(precision_score(testLabel, lr_predicted)))
print('Recall: {:.2f}'.format(recall_score(testLabel, lr_predicted)))
print('F1: {:.2f}'.format(f1_score(testLabel, lr_predicted)))


# In[13]:

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(testLabel, lr_predicted)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]

plt.figure()
plt.xlim([0.3, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label='Precision-Recall Curve')
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)

plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.axes().set_aspect('equal')
plt.show()


# In[14]:

from sklearn import metrics
from sklearn.metrics import roc_curve, auc

#testLabel, lr_predicted

fpr, tpr, _ = metrics.roc_curve(testLabel,lr_predicted)
auc = metrics.roc_auc_score(testLabel, lr_predicted)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc=4)
plt.show()


# In[19]:

from matplotlib import cm
from sklearn.svm import SVC

#trainLabel = np.asarray(dfTrain['Outcome'])
#trainData = np.asarray(dfTrain.drop('Outcome',1))
#testLabel = np.asarray(dfTest['Outcome'])
#testData = np.asarray(dfTest.drop('Outcome',1))


#svm_predicted_mc = svm.predict(X_test_mc)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
for g in [0.1, 0.5, 1, 10, 100]:
    
    svm = SVC(gamma=g).fit(trainData, trainLabel)
    
    y_score_svm = svm.decision_function(testData)
    
    svm_predicted_mc = svm.predict(testData)
    
    fpr_svm, tpr_svm, _ = roc_curve(testLabel, y_score_svm)
    roc_auc_svm = metrics.roc_auc_score(testLabel, lr_predicted)
    accuracy_svm = svm.score(testData, testLabel)
    #print(classification_report(y_test_mc, svm_predicted_mc))
    print("gamma = {:.2f}  accuracy = {:.2f}   AUC = {:.2f}".format(g, accuracy_svm, 
                                                                   roc_auc_svm))
    #print(recall_score(testLabel,svm_predicted_mc))
    
    plt.plot(fpr_svm, tpr_svm, lw=3, alpha=0.7, 
             label='SVM (gamma = {:0.2f}, area = {:0.2f})'.format(g, roc_auc_svm))

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate (Recall)', fontsize=16)
plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
plt.legend(loc="lower right", fontsize=11)
plt.title('ROC curve: (1-of-10 digits classifier)', fontsize=16)
plt.axes().set_aspect('equal')


# In[20]:

coeff = list(diabetesCheck.coef_[0])
labels = list(dfTrain.drop('Outcome',1))

features = pd.DataFrame()

features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)

features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')


# # Data PREDICTION 

# In[21]:

joblib.dump([diabetesCheck, means, stds], 'diabeteseModel.pkl')


# In[22]:

diabetesLoadedModel, means, stds = joblib.load('diabeteseModel.pkl')
accuracyModel = diabetesLoadedModel.score(testData, testLabel)
print("accuracy = ",accuracyModel * 100,"%")


# In[23]:

sampleData = dfCheck
sampleData.head()


# In[24]:

# prepare sample
sampleDataFeatures = np.asarray(sampleData.drop('Outcome',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds


# In[25]:

# predict
predictionProbability = diabetesLoadedModel.predict_proba(sampleDataFeatures)
prediction = diabetesLoadedModel.predict(sampleDataFeatures)
score = diabetesLoadedModel.score(sampleDataFeatures,sampleData['Outcome'])
print('Probability:', predictionProbability)
print('prediction:', prediction)
print('score', score)

