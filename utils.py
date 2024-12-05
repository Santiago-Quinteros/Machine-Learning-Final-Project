import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


#Determine if a row is numeric 
def is_decimal(s):
    if (s[0] == '-' and s[1:].replace('.', '').isnumeric()) or \
       (s[0] != '-' and s.replace('.', '').isnumeric()):
        return True
    return False

#Map text to indices
def map_to_indices(a, b):
    b_map = {word.lower(): i for i, word in enumerate(b)}
    return [b_map.get(word.lower(), None) for word in a]


# This function centers and normalices all the vectors 
def center_normalize(array):
    # Center and normalize the data
    scaler = StandardScaler()
    array = scaler.fit_transform(array)
    
    return array

#Map text, and change type to float and return the training vectors
#Also removes index columns
def mapping(df):
    X_data=[]
    Y_data=[]
    X_data=df.iloc[:,:-1].to_numpy().reshape(-1, df.shape[1]-1)
    Y_data=df.iloc[:,-1].to_numpy().reshape(-1)

    le = LabelEncoder()
    Y_data = le.fit_transform(Y_data)
    Y_data =[int(i) for i in Y_data ]

    for i in range(X_data.shape[1]):
        if isinstance(X_data[0, i], str):
            if not is_decimal(X_data[0, i]):
                unique_strs = np.unique(X_data[:,i])
                X_data[:,i]= map_to_indices(X_data[:,i], unique_strs)
            else :
                X_data[:,i]=X_data[:,i].astype(float) 
        if np.array_equal(X_data[:,i],np.array(range(len(X_data[:,i]))).astype(float)): #remove id columns
            X_data[:,i]=np.zeros(len(X_data[:,i])).astype(float)


    return X_data, Y_data


# Replace any non-alphanumeric character with NaN
def replace_nonalphanumerical(df): 
    for col in df.columns:
                
        df[col] = df[col].astype(str).str.replace(r"\t", "", regex=True)
        df[col] = df[col].astype(str).str.replace(r" ", "", regex=True)
        df[col] = df[col].astype(str).str.rstrip("?")
        df[col].replace('', np.nan, inplace=True)
        df[col].replace('nan', np.nan, inplace=True)  
        
    return df

# This function replace missing values by median values. It recives the data frame and returns it with the missing values
def fill_missing_values(df):
    # Numerical values 
    numeric_columns = df.select_dtypes(include=['number']).columns 
    if len(numeric_columns) != 0:
        imputer = SimpleImputer(strategy='mean')
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns]) 

    # Text value 
    text_columns = df.select_dtypes(include=['object']).columns 
    if len(text_columns) != 0:
        imputer = SimpleImputer(strategy='most_frequent')
        df[text_columns] = imputer.fit_transform(df[text_columns])
    return df
