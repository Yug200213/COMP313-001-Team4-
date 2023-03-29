# -*- coding: utf-8 -*-

#Library
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib

#Load the data set into pandas dataframe
path="C:/Users/Yug Patel/OneDrive/Desktop/Centennial/AI Supervised Learning/"
filename="Bicycle_Thefts.csv"
fullpath=os.path.join(path,filename)
data_group4=pd.read_csv(fullpath)

#Column Values with names
print(data_group4.columns.values)
#Info of the each column
print(data_group4.info(verbose=True))
#Check if there is any null values
print(data_group4.isnull().sum())
#Data type of each column
print(data_group4.dtypes)
 

#Stastical Assessment 
print(data_group4.mean())
print(data_group4.min())
print(data_group4.median())
print(data_group4.max())
print(data_group4.count())
print(data_group4.describe())
# Printing the values of status column which is target class
#Before Balancing the data
print("Target class count Before Balancing the data\n",data_group4['Status'].value_counts())
integer_features = data_group4.select_dtypes(exclude="object").columns.tolist()
print("\nTotal No of integer_features:",len(integer_features),"\n List of integer_features:", integer_features)



categorical_features = data_group4.select_dtypes(include="object").columns.tolist()
print("\nTotal No of categorical_features:",len(categorical_features),"\n List of integer_features:", categorical_features)


#Using Pearson Correlation we are finding corealtion between numerical values
plt.figure(figsize=(12,10))
cor = data_group4.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

#ax1 = data_group4.value_counts().plot.pie(autopct='%.2f')

# Replacing null values of the numeric data with the median of the specific column
data_group4["Cost_of_Bike"]=data_group4["Cost_of_Bike"].fillna(data_group4["Cost_of_Bike"].median())


Temp_data=data_group4.copy()

"""
Data Cleaning
# We are getting 90% corelation between Occurance Day and Report Day of year 
# We are getting 100% corelation between Occurance year and report year so we can drop one of them
# We dropped X and Y columns because it is corelating with longitude and latitude
# We are getting 61% corelation between Occurance DayofMonth and reported DayofMonth
# Same values for OBJECTID_1 and OBJECTID so we dropped OBJECTID_1
"""
Temp_data=Temp_data.drop(['OBJECTID_1','X','Y','Occurrence_DayOfYear','Report_Year','Occurrence_DayOfMonth'],axis=1)
#Droping the Latitude columns with the value 0
integer_features1 = Temp_data.select_dtypes(exclude="object").columns.tolist()
print("\nTotal No of integer_features:",len(integer_features1),"\n List of integer_features:", integer_features1)

# We used this for loop for ploting the graph between status and integer Features
for col in integer_features1:
    sns.catplot(x="Status",y=col, data=Temp_data)
    
# Removing Outliers from columns
Temp_data.drop(Temp_data.loc[Temp_data['Latitude']==0].index, inplace=True)

#We have removed the outliers related to the cost of bike
Temp_data.drop(Temp_data.loc[Temp_data['Cost_of_Bike']==0].index, inplace=True)
Temp_data.drop(Temp_data.loc[Temp_data['Cost_of_Bike']>10000].index, inplace=True)
sns.catplot(x="Status",y="Cost_of_Bike", data=Temp_data)

# Removing the outliers for Occurance_Year column
Temp_data.drop(Temp_data.loc[Temp_data['Occurrence_Year']==2009].index, inplace=True)
Temp_data.drop(Temp_data.loc[Temp_data['Occurrence_Year']==2010].index, inplace=True)
Temp_data.drop(Temp_data.loc[Temp_data['Occurrence_Year']==2011].index, inplace=True)
Temp_data.drop(Temp_data.loc[Temp_data['Occurrence_Year']==2012].index, inplace=True)
Temp_data.drop(Temp_data.loc[Temp_data['Occurrence_Year']==2013].index, inplace=True)


#We have drop the duplicates value for the event unique id except first occurance
Temp_data=Temp_data.drop_duplicates(subset=['event_unique_id'])
Temp_data['Status'].value_counts()
# We can drop the event unique id because event uniqueID is only for removing duplicates value.
Temp_data=Temp_data.drop(['event_unique_id'],axis=1)

#We converted the Occurance_Date column to the specified date formate and than change to %Y-%m-%d
Temp_data['Occurrence_Date'] = pd.to_datetime(Temp_data.Occurrence_Date, format='%Y/%m/%d %H:%M:%S')
Temp_data['Occurrence_Date'] = Temp_data['Occurrence_Date'].dt.strftime('%Y%m%d')

convert={"Occurrence_Month": {"January":1,"February":2,"March":3,
                          "April":4,"May":5,"June":6,"July":7,"August":8,
                          "September":9,"October":10,"November":11,"December":12} }
Temp_data=Temp_data.replace(convert)


convert1={"Premises_Type": {"Apartment":1,"Commercial":2,"Educational":3,
                          "Outside":4,"House":5,"Transit":6,"Other":7} }
Temp_data=Temp_data.replace(convert1)



convert2={"Occurrence_DayOfWeek": {"Monday":1,"Tuesday":2,"Wednesday":3,
                          "Thursday":4,"Friday":5,"Saturday":6,"Sunday":7} }
Temp_data=Temp_data.replace(convert2)

# It's having longitude and latitude valus as 0 and in graph it shows outliers
Temp_data.drop(Temp_data.loc[Temp_data['City']=='NSA'].index, inplace=True)
# Dropping this columns because haveing same value
Temp_data=Temp_data.drop(['City'],axis=1)
# We are droping rows with the Hood_ID Value=NSA
Temp_data.drop(Temp_data.loc[Temp_data['Hood_ID']=='NSA'].index, inplace=True)
Temp_data['Hood_ID']=Temp_data['Hood_ID'].astype(int)
sns.catplot(x="Status",y="Hood_ID", data=Temp_data)

   
# Just for prediction we are dropping the Primary_Columns which is not having any relation with the target class
Temp_data=Temp_data.drop(['Primary_Offence'],axis=1)


Temp_data['Occurrence_Date']=Temp_data['Occurrence_Date'].astype(int)

# Droping the columns for the feature selection
Temp_data=Temp_data.drop(['OBJECTID','Report_Month','Report_Date','Report_DayOfWeek',
                          'Report_Hour','NeighbourhoodName','Location_Type','Bike_Make','Bike_Model',
                          'Bike_Type','Bike_Speed','Bike_Colour'],axis=1)
# Converting the Division to numerical values and than converting it to int
Temp_data['Division']=Temp_data['Division'].str[1:]
Temp_data['Division']=Temp_data['Division'].astype(int)

# We are Predicting our model for stolen or recovered so we dropped rows with unknown status
# Since We Didn't find any corelation with other other status so we dropped it
Temp_data.drop(Temp_data.loc[Temp_data['Status']=="UNKNOWN"].index, inplace=True)

# Selection of features and target for prediction
features = Temp_data.drop(['Status'], axis=1)
target = Temp_data['Status']  

# We thought it would be better to oversample rather than undersample 
# Undersampling would have led us to throw away more than 90% of our data.
#managing the imbalanced data class
ros = RandomOverSampler(sampling_strategy="not majority")
X_res, y_res = ros.fit_resample(features, target)
ax = y_res.value_counts().plot.pie(autopct='%.2f')

print(y_res.value_counts())

X_res.head(30)
y_res.head(30)

Dummy= pd.DataFrame(X_res,y_res)


#Split the data set into 20% for testing and 80% for training
X_Train, X_Test,Y_Train, Y_Test=train_test_split(X_res,y_res,test_size=0.20,random_state=27)

X_Test.head(30)
Y_Test.head(30)


#Stadard Scalar
scalar= StandardScaler()
pipeline_gp1=Pipeline([("scalar",StandardScaler())])
pipeline_gp1.fit_transform(X_Train,Y_Train)


# Model Building and Fine Tuning
#Logistic Regression
LR_clf_Group_1=LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=1000,C=10, tol=0.1,random_state=42)
LR_clf_Group_1.fit(X_Train,Y_Train)
X_Predict_LR=LR_clf_Group_1.predict(X_Train)
print("Confusion Matrix for Logistic Regression:-",confusion_matrix(Y_Train, X_Predict_LR))
print("Classfication report for Logistic Regression:-\n",classification_report(Y_Train, X_Predict_LR))
print(cross_val_score(LR_clf_Group_1,X_Train,Y_Train,cv=3, scoring="accuracy"))
print("Accuracy Score for Training for Logistic Regression:-",LR_clf_Group_1.score(X_Train, Y_Train))
print("Accuracy Score for Testing for Logistic Regression:-",LR_clf_Group_1.score(X_Test, Y_Test))



#SVM
pipe_svm_gp1=Pipeline(steps=[['pipeline_gp1',pipeline_gp1],["svm_gp1_clf",SVC(random_state=27)]])

#gird search parameter
param_grid_svm={'svm_gp1_clf__kernel':['rbf'],'svm_gp1_clf__C':[0.1,1],
            'svm_gp1_clf__gamma':[0.3],'svm_gp1_clf__degree':[2,3]}


#grid search classifier with fitting the X_Train and Y_Train
grid_search_gp1=GridSearchCV(estimator=pipe_svm_gp1,param_grid=param_grid_svm,scoring='accuracy',
                             refit=True,verbose=3,cv=3)
grid_search_gp1.fit(X_Train,Y_Train)
print("The best parameters are :- %s"% (grid_search_gp1.best_params_))
print("The best estimator are  :-%s"%(grid_search_gp1.best_estimator_))
X_predict_SVM=grid_search_gp1.predict(X_Train)
print("Confusion Matrix for svm:-\n",confusion_matrix(Y_Train, X_predict_SVM))
print("Classfication report for svm :- \n",classification_report(Y_Train, X_predict_SVM))
print("Accuracy Score for Training for svm :-",grid_search_gp1.score(X_Train,Y_Train))
print("Accuracy Score for Testing for svm :-",grid_search_gp1.score(X_Test,Y_Test))




# Creating the decision tree classifier
clf_tree_gp1 = DecisionTreeClassifier(max_depth=5, criterion = 'entropy', random_state=27)
# Creating the pipeline of two steps to fit the model and the transformer
pipeline_tree=Pipeline(steps=[['pipeline_gp1',pipeline_gp1],['clf_tree_gp1',clf_tree_gp1]])
# Setting the parameter for the grid search
parameters={'clf_tree_gp1__min_samples_split' : range(10,300,20),'clf_tree_gp1__max_depth':
range(1,30,2),'clf_tree_gp1__min_samples_leaf':range(1,15,3)}

#Decision Tree Without randomized Grid Search
pipeline_tree.fit(X_Train,Y_Train)
pipeline_tree.score(X_Train,Y_Train)
    
# randomized grid search model    
grid_searchrandom_gp1=RandomizedSearchCV(estimator= pipeline_tree,scoring='accuracy',
                             param_distributions=parameters,cv=5,n_iter=7,refit=True,verbose=3)
grid_searchrandom_gp1.fit(X_Train,Y_Train)
grid_searchrandom_gp1.fit(X_Test,Y_Test)
print("The best parameters are :- %s"% (grid_searchrandom_gp1.best_params_))
print("The best estimator are :-  %s"%(grid_searchrandom_gp1.best_estimator_))
X_predict_DT=grid_searchrandom_gp1.predict(X_Train)
print("Confusion Matrix for DecisionTree :-\n",confusion_matrix(Y_Train, X_predict_DT))
print("Classfication report for DecisionTree :-\n",classification_report(Y_Train, X_predict_DT))
print("Accuracy Score for Training for DecisionTree :-",grid_searchrandom_gp1.score(X_Train,Y_Train))
print("Accuracy Score for Testing for DecisionTree :-",grid_searchrandom_gp1.score(X_Test,Y_Test))


#Creating pickle files for Serialize    
filename='model.pkl'
joblib.dump(grid_searchrandom_gp1, filename)

model_columns=list(X_Test.columns)
print(model_columns)
column_filename='model_columns.pkl'
joblib.dump(model_columns, column_filename)




