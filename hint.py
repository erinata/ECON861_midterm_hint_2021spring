import pandas
from sklearn import linear_model
from datetime import datetime
import numpy

dataset = pandas.read_csv("AB_NYC_2019.csv")   # load the dataset


dataset.dropna(inplace=True)	                 # Drop observations with missing data
print(dataset)


print(dataset.groupby('neighbourhood_group').count())



dataset['Bronx'] = numpy.where(dataset['neighbourhood_group']=='Bronx',1,0)
dataset['Brooklyn'] = numpy.where(dataset['neighbourhood_group']=='Brooklyn',1,0)
dataset['Manhattan'] = numpy.where(dataset['neighbourhood_group']=='Manhattan',1,0)
dataset['Queens'] = numpy.where(dataset['neighbourhood_group']=='Queens',1,0)
dataset['Staten Island'] =numpy.where( dataset['neighbourhood_group']=='Staten Island',1,0)



target = dataset['price'].values     # Get the column 'price' 

data = dataset.iloc[:,16:22].values    # Get the columns 'Bronx','Brooklyn' ......


machine = linear_model.LinearRegression()   # Construct the machine
machine.fit(data, target)   # Fit the data and the target

new_data = [
	[1,0,0,0,0],
	[0,1,0,0,0],
	[0,0,1,0,0],
	[0,0,0,1,0],
	[0,0,0,0,1]
]

new_target = machine.predict(new_data)  
print(new_target)



