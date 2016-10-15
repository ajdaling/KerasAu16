from pandas import HDFStore

store = HDFStore('mnist_weights.h5')

store[0].to_csv('weights.csv')
