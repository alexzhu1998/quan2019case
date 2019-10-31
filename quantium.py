import sys
print('Python: {}'.format(sys.version))


# Scipy versions
import scipy
print('scipy: {}'.format(scipy.__version__))

# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))

# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

#pandas
import pandas as pd
print('pandas: {}'.format(pd.__version__))

#scikit-learn
import sklearn
print('sklearn:{}'.format(sklearn.__version__))

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import os
print(os.getcwd())

###change directory
#os.chdir(default_path)

### read the file
df = pd.read_csv("Quantium Combined Dataset.csv")
# df = pd.read_csv("Quantium Combined Dataset.csv",encoding = 'latin1')
# df = pd.read_csv("Quantium Combined Dataset.csv", dtype={'COST_TO_HOTEL:float'})

### Change columns in upper case
df.columns = df.columns.str.upper()

### Renaming columns
df.rename(columns = {'CHECK IN DATE':'CHECK_IN_DATE','CHECK OUT DATE':'CHECK_OUT_DATE','CHECK IN DAY':'CHECK_IN_DAY','CHECK OUT DAY':'CHECK_OUT_DAY','COST TOTAL':'COST_TOTAL','PROFIT MARGIN':'PROFIT_MARGIN','LEAD TIME':'LEAD_TIME'},inplace=True)
# print(list(df.columns))

### Check and impute for Missing Data
# print(df.isnull().any())
# print(df.isnull().sum())
# df_with_0 = df.fillna(0)
# df_with_mean = df.CHECK_IN_DAY.fillna(df['CHECK_IN_DAY'].mean())
# df_drop = df.dropna()
## how = 'any' means any rows, thresh = 2 means threshold of 2
# df_with_condition = df.dropna(how = 'any',thresh = 2)

### Check for Duplicates
# print(df.duplicated().sum())
# df_drop_dup = df.drop_duplicates()

### Data type Check


### Data type change: int to float
df.COST_TO_HOTEL = df.COST_TO_HOTEL.astype(float)
df.PROFIT = df.PROFIT.astype(float)
df.COST_TOTAL = df.COST_TOTAL.astype(float)
df.ANNUAL_INCOME = df.ANNUAL_INCOME.astype(float)
# print(df.dtypes)

### Data type change: replacing strings
df.PROFIT_MARGIN = df.PROFIT_MARGIN.str.replace('%','').astype(float)
df.PROFIT_MARGIN = df.PROFIT_MARGIN/100
# print(df.PROFIT_MARGIN)

# Checking for strings containing specific string name
# df.DESCRIPTION.str.contains('food').astype(int).head()

# Changing dates data type
df.CHECK_IN_DATE = pd.to_datetime(df.CHECK_IN_DATE)
df.CHECK_OUT_DATE = pd.to_datetime(df.CHECK_OUT_DATE)
df.SURVEY_DATE = pd.to_datetime(df.SURVEY_DATE)
df.BOOK_DATE = pd.to_datetime(df.BOOK_DATE)
df.TXN_DATE = pd.to_datetime(df.TXN_DATE)
# print(df.dtypes)
# print(df[['SURVEY_DATE','BOOK_DATE','TXN_DATE']])
### Changing data types



###Output dimension of df
#print(df.shape)

###Output head of the df
#print(df.head(20))

### Subsetting columns
# customerData = df[['CUST_ID', 'HOTEL_ID', 'BOOKING_ID', 'SERVICE_ID', 'COST_TO_HOTEL', 'AGE', 'ANNUAL_INCOME', 'RESIDENTIAL_STATE', 'STATE', 'CITY', 'CHECK IN DATE', 'CHECK OUT DATE', 'Check In Day', 'Check Out Day', 'Cost Total', 'Profit', 'Profit Margin', 'Lead Time']]
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(customerData.head(20))


###descriptions
# print(df.describe())

###class distribution
#print(df.groupby('class').size())

####
####DATA VISUALISATION
####

# ### Univariate Plots
# df.plot(kind = 'box', subplots=True,layout  =(2,2), sharex=False, sharey=False)
#
# ### histograms
# df.hist()
#
# ### scatter plot matrix
# scatter_matrix(df)
# #plt.show()
#
#
# # Split-out validation df
# array = df.values
#
# X = array[:, 0:4]
# Y = array[:,4]
# validation_size = 0.20
# seed = 7
# X_train,X_validation,Y_train,Y_validation = model_selection.train_test_split(X,Y,test_size = validation_size,random_state = seed)
#
#
# #Test options and evaluation metric
# seed = 7
# scoring = 'accuracy'
#
# # Spot Check Algorithms
#
# models = []
# models.append(('LR',LogisticRegression(solver='liblinear',multi_class ='ovr')))
# models.append(('LDA',LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART',DecisionTreeClassifier()))
# models.append(('NB',GaussianNB()))
# models.append(('SVM', SVC(gamma = 'auto')))
