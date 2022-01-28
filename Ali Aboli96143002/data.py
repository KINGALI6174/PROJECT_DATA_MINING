import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
class data_set():
    def __init__(self):
        self.pre_le=preprocessing.LabelEncoder()
        self.sales_data=pd.read_csv("car.data",header=None)
    def lebel(self) :
        self.sales_data[0] = le.fit_transform(self.sales_data[0])
        self.sales_data[1] = le.fit_transform(self.sales_data[1])
        self.sales_data[2] = le.fit_transform(self.sales_data[2])
        self.sales_data[3] = le.fit_transform(self.sales_data[3])
        self.sales_data[4] = le.fit_transform(self.sales_data[4])
        self.sales_data[5] = le.fit_transform(self.sales_data[5])
        self.sales_data[6] = le.fit_transform(self.sales_data[6])
    def lebel_output(self):
        self.lebel()
        cols = [col for col in self.sales_data.columns if col not in [6]]
        data = self.sales_data[cols]
        target = self.sales_data[6]
        data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)
        return data_train, data_test, target_train, target_test
        
        




