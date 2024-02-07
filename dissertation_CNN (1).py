#!/usr/bin/env python
# coding: utf-8

# In[22]:


def TSF_using_CNN():
  
    import warnings
    warnings.filterwarnings("ignore")    
    
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from pandas.plotting import autocorrelation_plot

    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7


    dataset = pd.read_csv("C:\\Users\\Dell\\Documents\\dissertation_train$21.csv", 
                          header=0, parse_dates=[0],
                          index_col=0, squeeze=True)
 
    

    print()
    print(dataset.shape)
    print(dataset.head(25))


    plt.plot(dataset)
    plt.show()

   
    autocorrelation_plot(dataset)
    plt.show()

  
    from numpy import array
    def split_sequences(sequences, n_steps):
    	X, y = list(), list()
    	for i in range(len(sequences)):
  
    		end_ix = i + n_steps

    		if end_ix > len(sequences)-1:
    			break
 
            
    		seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
    		X.append(seq_x)
    		y.append(seq_y)
    	return array(X), array(y)

   
    n_steps = 5

    X, y = split_sequences(dataset, n_steps)
    
    print(X.shape)
    print(y)

    
    for i in range(len(X)):
        print(X[i], y[i])

  
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv1D
    from keras.layers.convolutional import MaxPooling1D

    
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))    
    model.add(Dense(1,activation ='linear'))
    model.compile(optimizer='adam', loss='mse')
    
    
    model.fit(X, y, epochs=5000,batch_size=2048, verbose=2)

    
    dataset = pd.read_csv("C:\\Users\\Dell\\Documents\\dissertation_train$21.csv")
    dataset = dataset['Magnitude']
    
    
    X, y = split_sequences(dataset, n_steps)    

    x_input = X.reshape((X.shape[0], X.shape[1], n_features))
    yhat = model.predict(x_input, verbose=1)
   

    df_pred = pd.DataFrame.from_records(yhat, columns = ['predicted'])
    df_pred = df_pred.reset_index(drop=True)
    
    df_actual = dataset[n_steps:len(dataset)]
    df_actual = df_actual.reset_index(drop=True)

    
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

    coefficient_of_dermination = r2_score(df_actual, df_pred)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(df_actual, df_pred)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(df_actual, df_pred)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(df_actual, df_pred)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

    
    ax = df_actual.plot(label='Observed', figsize=(9, 7))
    df_pred.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Magnitude')
    plt.legend()
    plt.show()



TSF_using_CNN()


# In[ ]:




