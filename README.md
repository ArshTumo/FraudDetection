About the dataset:

This dataset shows us normal day-to-day transactions including the occurrence of fraudulent transactions.

Dataset features:

step: represents a unit of time where 1 step equals 1 hour
type: type of online transaction
amount: the amount of the transaction
nameOrig: customer starting the transaction
oldbalanceOrg: balance before the transaction
newbalanceOrig: balance after the transaction
nameDest: recipient of the transaction
oldbalanceDest: initial balance of recipient before the transaction
newbalanceDest: the new balance of recipient after the transaction
isFraud: fraud transaction

Our goal is to build a machine learning model to classify fraud and non-fraud transactions.

I checked if there are any null values and duplicated rows. Encoded the 'type' column using LabelEncoder from sklearn.
Visualised the correlation matrix using plotly.express to see what features are correlated to each other. 
Then I removed unnecessary columns. Plotted the histogram of some columns.
Applied log transformation to those columns using np.log and removed outliers using zscore from scipy.
Because our dataset was imbalanced, I used RandomUnderSampler from imblearn to undersample the data.
Then I scaled the dataset using StandardScaler from sklearn.
And finally I trained the model using DecisionTreeClassifier and tested it. I got a score of 0.95.
And at the end I saved the model.
