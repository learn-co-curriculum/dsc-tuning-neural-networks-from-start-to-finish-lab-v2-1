# Tuning and Optimizing Neural Networks - Lab

## Introduction

Now that you've practiced regularization, initialization, and optimization techniques, its time to synthesize these concepts into a cohesive modeling pipeline.  

With this pipeline, you will not only fit an initial model but also attempt to improve it. Your final model selection will pertain to the test metrics across these models. This will more naturally simulate a problem you might be faced with in practice, and the various modeling decisions you are apt to encounter along the way.  

Recall that our end objective is to achieve a balance between overfitting and underfitting. You've seen the bias variance trade-off, and the role of regularization in order to reduce overfitting on training data and improving generalization to new cases. Common frameworks for such a procedure include train/validate/test methodology when data is plentiful, and K-folds cross-validation for smaller, more limited datasets. In this lab, you'll perform the latter, as the dataset in question is fairly limited. 

## Objectives

You will be able to:

* Apply normalization as a preprocessing technique 
* Implement a K-folds cross validation modeling pipeline for deep learning models 
* Apply regularization techniques to improve your model's performance 

## Load the data

First, run the following cell to import all the neccessary libraries and classes you will need in this lab. 


```python
# Necessary libraries and classes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from keras import models
from keras import layers
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor
```


```python
# __SOLUTION__ 
# Necessary libraries and classes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from keras import models
from keras import layers
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor
```

    Using TensorFlow backend.


In this lab you'll be working with the *The Lending Club* data. 

- Import the data available in the file `'loan_final.csv'` 
- Drop rows with missing values in the `'total_pymnt'` column (this is your target column) 
- Print the first five rows of the data 
- Print the dimensions of the data 


```python
# Import the data
data = None

# Drop rows with no target value


# Print the first five rows

```


```python
# __SOLUTION__ 
# Import the data
data = pd.read_csv('loan_final.csv', header=0)

# Drop rows with no target value
data.dropna(subset=['total_pymnt'], inplace=True)

# Print the first five rows
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>loan_status</th>
      <th>purpose</th>
      <th>addr_state</th>
      <th>total_acc</th>
      <th>total_pymnt</th>
      <th>application_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000.0</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>B</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>24000.0</td>
      <td>Verified</td>
      <td>Fully Paid</td>
      <td>credit_card</td>
      <td>AZ</td>
      <td>9.0</td>
      <td>5863.155187</td>
      <td>Individual</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>C</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>30000.0</td>
      <td>Source Verified</td>
      <td>Charged Off</td>
      <td>car</td>
      <td>GA</td>
      <td>4.0</td>
      <td>1014.530000</td>
      <td>Individual</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>C</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>12252.0</td>
      <td>Not Verified</td>
      <td>Fully Paid</td>
      <td>small_business</td>
      <td>IL</td>
      <td>10.0</td>
      <td>3005.666844</td>
      <td>Individual</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>C</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>49200.0</td>
      <td>Source Verified</td>
      <td>Fully Paid</td>
      <td>other</td>
      <td>CA</td>
      <td>37.0</td>
      <td>12231.890000</td>
      <td>Individual</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>67.79</td>
      <td>B</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>80000.0</td>
      <td>Source Verified</td>
      <td>Fully Paid</td>
      <td>other</td>
      <td>OR</td>
      <td>38.0</td>
      <td>4066.908161</td>
      <td>Individual</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the dimensions of data 

```


```python
# __SOLUTION__ 
# Print the dimensions of data
data.shape
```




    (42535, 16)



## Generating a Hold Out Test Set

While we will be using K-fold cross validation to select an optimal model, we still want a final hold out test set that is completely independent of any modeling decisions. As such, pull out a sample of 30% of the total available data. For consistency of results, use random seed 42. 


```python
# Features to build the model
features = ['loan_amnt', 'funded_amnt_inv', 'installment', 'annual_inc', 
            'home_ownership', 'verification_status', 'emp_length']

X = data[features]
y = data[['total_pymnt']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = None
```


```python
# __SOLUTION__ 
# Features to build the model
features = ['loan_amnt', 'funded_amnt_inv', 'installment', 'annual_inc', 
            'home_ownership', 'verification_status', 'emp_length']

X = data[features]
y = data[['total_pymnt']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## Preprocessing (Numerical features) 

- Fill all missing values in numeric features with their respective means 
- Standardize all the numeric features  
- Convert the final results into DataFrames 


```python
# Select continuous features
cont_features = ['loan_amnt', 'funded_amnt_inv', 'installment', 'annual_inc']

X_train_cont = X_train.loc[:, cont_features]
X_test_cont = X_test.loc[:, cont_features]

# Instantiate SimpleImputer - fill the missing values with the mean
si = None

# Fit and transform the training data
X_train_imputed = si.fit_transform(X_train_cont)

# Transform test data
X_test_imputed = si.transform(X_test_cont)

# Instantiate StandardScaler
ss_X = None

# Fit and transform the training data
X_train_scaled = None

# Transform test data
X_test_scaled = None
```


```python
# __SOLUTION__ 
# Select continuous features
cont_features = ['loan_amnt', 'funded_amnt_inv', 'installment', 'annual_inc']
X_train_cont = X_train.loc[:, cont_features]
X_test_cont = X_test.loc[:, cont_features]

# Instantiate SimpleImputer - fill the missing values with the mean
si = SimpleImputer(strategy='mean')

# Fit and transform the training data
X_train_imputed = si.fit_transform(X_train_cont)

# Transform test data
X_test_imputed = si.transform(X_test_cont)

# Instantiate StandardScaler
ss_X = StandardScaler()

# Fit and transform the training data
X_train_scaled = pd.DataFrame(ss_X.fit_transform(X_train_imputed), columns=cont_features)

# Transform test data
X_test_scaled = pd.DataFrame(ss_X.transform(X_test_imputed), columns=cont_features)
```

## Preprocessing (Categorical features) 

- Fill all missing values in categorical features with the string `'missing'` 
- One-hot encode all categorical features 
- Convert the final results into DataFrames 



```python
# Select only the categorical features
cat_features = ['home_ownership', 'verification_status', 'emp_length']
X_train_cat = X_train.loc[:, cat_features]
X_test_cat = X_test.loc[:, cat_features]

# Fill missing values with the string 'missing'




# OneHotEncode categorical variables
ohe = None

# Transform training and test sets
X_train_ohe = None
X_test_ohe = None

# Get all categorical feature names
cat_columns = ohe.get_feature_names_out(input_features=X_train_ohe.columns)

# Fit and transform the training data
X_train_categorical = None

# Transform test data
X_test_categorical = None
```


```python
# __SOLUTION__ 
# Select only the categorical features
cat_features = ['home_ownership', 'verification_status', 'emp_length']
X_train_cat = X_train.loc[:, cat_features]
X_test_cat = X_test.loc[:, cat_features]

# Fill missing values with the string 'missing'
X_train_cat.fillna(value='missing', inplace=True)
X_test_cat.fillna(value='missing', inplace=True)

# OneHotEncode categorical variables
ohe = OneHotEncoder(handle_unknown='ignore')

# Transform training and test sets
X_train_ohe = ohe.fit_transform(X_train_cat)
X_test_ohe = ohe.transform(X_test_cat)

# Get all categorical feature names
cat_columns = ohe.get_feature_names_out(input_features=X_train_cat.columns)

# Fit and transform the training data
X_train_categorical = pd.DataFrame(X_train_ohe.todense(), columns=cat_columns)

# Transform test data
X_test_categorical = pd.DataFrame(X_test_ohe.todense(), columns=cat_columns)
```

Run the below cell to combine the numeric and categorical features. 


```python
# Combine continuous and categorical feature DataFrames
X_train_all = pd.concat([X_train_scaled, X_train_categorical], axis=1)
X_test_all = pd.concat([X_test_scaled, X_test_categorical], axis=1)

# Number of input features
n_features = X_train_all.shape[1]
```


```python
# __SOLUTION__ 
# Combine continuous and categorical feature DataFrames
X_train_all = pd.concat([X_train_scaled, X_train_categorical], axis=1)
X_test_all = pd.concat([X_test_scaled, X_test_categorical], axis=1)

# Number of input features
n_features = X_train_all.shape[1]
```

- Standardize the target DataFrames (`y_train` and `y_test`) 


```python
# Instantiate StandardScaler
ss_y = None

# Fit and transform Y (train)
y_train_scaled = None

# Transform test Y (test)
y_test_scaled = None
```


```python
# __SOLUTION__ 
# Instantiate StandardScaler
ss_y = StandardScaler()

# Fit and transform Y (train)
y_train_scaled = ss_y.fit_transform(y_train)

# Transform test Y (test)
y_test_scaled = ss_y.transform(y_test)
```

## Define a K-fold Cross Validation Methodology

Now that your have a complete holdout test set, you will perform k-fold cross-validation using the following steps: 

- Create a function that returns a compiled deep learning model 
- Use the wrapper function `KerasRegressor()` that defines how these folds are trained 
- Call the `cross_val_predict()` function to perform k-fold cross-validation 

In the cell below, we've defined a baseline model that returns a compiled Keras models. 


```python
# Define a function that returns a compiled Keras model 
def create_baseline_model():
    
    # Initialize model
    model = models.Sequential()

    # First hidden layer
    model.add(layers.Dense(10, activation='relu', input_shape=(n_features,)))

    # Second hidden layer
    model.add(layers.Dense(5, activation='relu'))

    # Output layer
    model.add(layers.Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='SGD', 
                  loss='mse',  
                  metrics=['mse']) 
    
    # Return the compiled model
    return model
```


```python
# __SOLUTION__ 
# Define a function that returns a compiled Keras model 
def create_baseline_model():
    
    # Initialize model
    model = models.Sequential()

    # First hidden layer
    model.add(layers.Dense(10, activation='relu', input_shape=(n_features,)))

    # Second hidden layer
    model.add(layers.Dense(5, activation='relu'))

    # Output layer
    model.add(layers.Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='SGD', 
                  loss='mse',  
                  metrics=['mse']) 
    
    # Return the compiled model
    return model
```

Wrap `create_baseline_model` inside a call to `KerasRegressor()`, and: 

- Train for 150 epochs 
- Set the batch size to 256 

> NOTE: Refer to the [documentation](https://keras.io/scikit-learn-api/) to learn about `KerasRegressor()`.  


```python
# Wrap the above function for use in cross-validation
keras_wrapper_1 = None
```


```python
# __SOLUTION__ 
# Wrap the above function for use in cross-validation
keras_wrapper_1 = KerasRegressor(create_baseline_model,  
                                 epochs=150, 
                                 batch_size=256, 
                                 verbose=0)
```

Use `cross_val_predict()` to generate cross-validated predictions with: 
- 5-fold cv 
- scaled input (`X_train_all`) and output (`y_train_scaled`) 


```python
# ⏰ This cell may take several mintes to run
# Generate cross-validated predictions
np.random.seed(123)
cv_baseline_preds = None
```


```python
# __SOLUTION__ 
# ⏰ This cell may take several mintes to run
# Generate cross-validated predictions
np.random.seed(123)
cv_baseline_preds = cross_val_predict(keras_wrapper_1, X_train_all, y_train_scaled, cv=5)
```

    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:986: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.
    
    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:973: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.
    


- Find the RMSE on train data 


```python
# RMSE on train data (scaled)

```


```python
# __SOLUTION__ 
# RMSE on train data (scaled)
np.sqrt(mean_squared_error(y_train_scaled, cv_baseline_preds))
```




    0.4441920534778701



- Convert the scaled predictions back to original scale 
- Calculate RMSE in the original units with `y_train` and `baseline_preds` 


```python
# Convert the predictions back to original scale
baseline_preds = None

# RMSE on train data (original scale)

```


```python
# __SOLUTION__ 
# Convert the predictions back to original scale
baseline_preds = ss_y.inverse_transform(cv_baseline_preds.reshape(-1, 1))

# RMSE on train data (original scale)
np.sqrt(mean_squared_error(y_train, baseline_preds))
```




    4045.940868567704



## Intentionally Overfitting a Model

Now that you've developed a baseline model, its time to intentionally overfit a model. To overfit a model, you can:
* Add layers
* Make the layers bigger
* Increase the number of training epochs

Again, be careful here. Think about the limitations of your resources, both in terms of your computers specs and how much time and patience you have to let the process run. Also keep in mind that you will then be regularizing these overfit models, meaning another round of experiments and more time and resources.


```python
# Define a function that returns a compiled Keras model 
def create_bigger_model():
    
    pass
```


```python
# __SOLUTION__ 
# Define a function that returns a compiled Keras model 
def create_bigger_model():
    
    # Initialize model
    model = models.Sequential()

    # First hidden layer
    model.add(layers.Dense(10, activation='relu', input_shape=(n_features,)))

    # Second hidden layer
    model.add(layers.Dense(10, activation='relu'))
    
    # Third hidden layer
    model.add(layers.Dense(10, activation='relu'))
    
    # Fourth hidden layer
    model.add(layers.Dense(10, activation='relu'))
    
    # Fifth hidden layer
    model.add(layers.Dense(10, activation='relu'))

    # Output layer
    model.add(layers.Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='SGD', 
                  loss='mse',  
                  metrics=['mse']) 
    
    # Return the compiled model
    return model
```


```python
# Wrap the above function for use in cross-validation
keras_wrapper_2 = None
```


```python
# __SOLUTION__ 
# Wrap the above function for use in cross-validation
keras_wrapper_2 = KerasRegressor(create_bigger_model,  
                                 epochs=150, 
                                 batch_size=256, 
                                 verbose=0)
```


```python
# ⏰ This cell may take several mintes to run
# Generate cross-validated predictions
np.random.seed(123)
cv_bigger_model_preds = None
```


```python
# __SOLUTION__ 
# ⏰ This cell may take several mintes to run
# Generate cross-validated predictions
np.random.seed(123)
cv_bigger_model_preds = cross_val_predict(keras_wrapper_2, X_train_all, y_train_scaled, cv=5)
```


```python
# RMSE on train data (scaled)

```


```python
# __SOLUTION__ 
# RMSE on train data (scaled)
np.sqrt(mean_squared_error(y_train_scaled, cv_bigger_model_preds))
```




    0.4470163121986446



## Regularizing the Model to Achieve Balance  

Now that you have a powerful model (albeit an overfit one), we can now increase the generalization of the model by using some of the regularization techniques we discussed. Some options you have to try include:  
* Adding dropout
* Adding L1/L2 regularization
* Altering the layer architecture (add or remove layers similar to above)  

This process will be constrained by time and resources. Be sure to test at least two different methodologies, such as dropout and L2 regularization. If you have the time, feel free to continue experimenting. 


```python
# Define a function that returns a compiled Keras model 
def create_regularized_model():
    
    pass
```


```python
# __SOLUTION__ 
# Define a function that returns a compiled Keras model 
def create_regularized_model():
    
    # Initialize model
    model = models.Sequential()

    # Input layer with dropout
    model.add(layers.Dropout(0.3, input_shape=(n_features,)))
    
    # First hidden layer
    model.add(layers.Dense(10, kernel_regularizer=regularizers.l2(0.005), activation='relu'))
    model.add(layers.Dropout(0.3))

    # Second hidden layer
    model.add(layers.Dense(10, kernel_regularizer=regularizers.l2(0.005), activation='relu'))
    model.add(layers.Dropout(0.3))
    
    # Third hidden layer
    model.add(layers.Dense(10, kernel_regularizer=regularizers.l2(0.005), activation='relu'))
    model.add(layers.Dropout(0.3))

    # Output layer
    model.add(layers.Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='SGD', 
                  loss='mse',  
                  metrics=['mse']) 
    
    # Return the compiled model
    return model
```


```python
# Wrap the above function for use in cross-validation
keras_wrapper_3 = None
```


```python
# __SOLUTION__ 
# Wrap the above function for use in cross-validation
keras_wrapper_3 = KerasRegressor(create_regularized_model,  
                                 epochs=150, 
                                 batch_size=256, 
                                 verbose=0)
```


```python
# ⏰ This cell may take several mintes to run
# Generate cross-validated predictions
np.random.seed(123)
cv_dropout_preds = None
```


```python
# __SOLUTION__ 
# ⏰ This cell may take several mintes to run
# Generate cross-validated predictions
np.random.seed(123)
cv_dropout_preds = cross_val_predict(keras_wrapper_3, X_train_all, y_train_scaled, cv=5)
```

    WARNING:tensorflow:From //anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.



```python
# RMSE on train data (scaled)

```


```python
# __SOLUTION__ 
# RMSE on train data (scaled)
np.sqrt(mean_squared_error(y_train_scaled, cv_dropout_preds))
```




    0.5864553945251834



## Final Evaluation

Now that you have selected a network architecture, tested various regularization procedures and tuned hyperparameters via a validation methodology, it is time to evaluate your final model on the test set. Fit the model using all of the training data using the architecture and hyperparameters that were most effective in your experiments above. Afterwards, measure the overall performance on the hold-out test data which has been left untouched (and hasn't leaked any data into the modeling process)! 


```python
# ⏰ This cell may take several mintes to run
```


```python
# __SOLUTION__ 
# Initialize model
model = models.Sequential()

# First hidden layer
model.add(layers.Dense(10, activation='relu', input_shape=(n_features,)))

# Second hidden layer
model.add(layers.Dense(5, activation='relu'))

# Output layer
model.add(layers.Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='SGD', 
              loss='mse',  
              metrics=['mse']) 

# Train the model 
model.fit(X_train_all, 
          y_train_scaled, 
          epochs=150, 
          batch_size=256, 
          verbose=0)
```




    <keras.callbacks.History at 0x1a4a070080>




```python

```


```python
# __SOLUTION__ 
# Generate predictions on test data 
final_preds_scaled = model.predict(X_test_all)

# Convert the predictions back to original scale 
final_preds = ss_y.inverse_transform(final_preds_scaled)

# RMSE on test data (original scale)
np.sqrt(mean_squared_error(y_test, final_preds))
```




    3852.5098261660673



## Summary

In this lab, you investigated some data from *The Lending Club* in a complete data science pipeline to build neural networks with good performance. You began with reserving a hold-out set for testing which never was touched during the modeling phase. From there, you implemented a k-fold cross validation methodology in order to assess an initial baseline model and various regularization methods. 
