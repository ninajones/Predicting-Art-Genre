#import statements
import numpy as np
import pandas as pd
 
#read dataset
df = pd.read_csv('artists.csv')

# convert years to centuries
df.loc[df['years'].str.contains('- 13', na=False), 'years'] = "14th century" 
df.loc[df['years'].str.contains('- 14', na=False), 'years'] = "15th century" 
df.loc[df['years'].str.contains('- 15', na=False), 'years'] = "16th century" 
df.loc[df['years'].str.contains('– 15', na=False), 'years'] = "16th century" 
df.loc[df['years'].str.contains('- 16', na=False), 'years'] = "17th century" 
df.loc[df['years'].str.contains('- 17', na=False), 'years'] = "18th century" 
df.loc[df['years'].str.contains('- 18', na=False), 'years'] = "19th century" 
df.loc[df['years'].str.contains('– 18', na=False), 'years'] = "19th century" 
df.loc[df['years'].str.contains('- 19', na=False), 'years'] = "20th century" df.loc[df['years'].str.contains('– 19', na=False), 'years'] = "20th century" 
df.loc[df['years'].str.contains('- 20', na=False), 'years'] = "21st century"

#Using regex, remove values after first comma in nationality and genre (strings are immutable so assign the reference to the result of string.replace) 
df['nationality'] = df['nationality'].astype(str).str.replace(r"[,\/]((\w*)|(.)).*","") df['genre'] = 
df['genre'].astype(str).str.replace(r"[,\/]((\w*)|(.)).*","")

#Convert categorical columns to numerical ones
from sklearn.preprocessing import LabelEncoder  
var_mod = ['years','genre','nationality']  
le = LabelEncoder()  
for i in var_mod:     
    df[i] = le.fit_transform(df[i])  
df.dtypes

#Import models from scikit learn module: 
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import KFold  #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier, export_graphviz  
from sklearn import metrics
 
#Generic function for making a classification model and accessing performance: 
def classification_model(model, data, predictors, outcome):   
  #Fit the model:      
  model.fit(data[predictors],data[outcome])            

  #Make predictions on training set: 
  predictions = model.predict(data[predictors])         

  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])      
  print ("Accuracy of predictions1: %s" % "{0:.3%}".format(accuracy))      

  #Perform k-fold cross-validation with 5 folds   
  kf = KFold(n_splits=5)      
  error = []      
  for train, test in kf.split(data[predictors]):     
    #Filter training data    
    train_predictors = (data[predictors].iloc[train,:])
               
    #The target we're using to train the algorithm.       
    train_target = data[outcome].iloc[train]   
          
    # Training the algorithm using the predictors and target.    
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run 
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))   

  #Fit the model again so that it can be refered outside the  function:  
  model.fit(data[predictors],data[outcome])   
  
  return predictions

#model created by classification_model function uses logistic regression algorithm
outcome_var = 'genre' 
model = LogisticRegression() 
predictor_var = ['years','nationality'] 
classification_model(model, df, predictor_var, outcome_var)

#model created by classification_model function uses decision trees algorithm
model = DecisionTreeClassifier() 
predictor_var = ['years','nationality']
classification_model(model, df,predictor_var,outcome_var)
 
#model created by classification_model function uses random forests algorithm
model = RandomForestClassifier(n_estimators=100) 
predictor_var = ['years','nationality'] 
classification_model(model, df,predictor_var,outcome_var)
