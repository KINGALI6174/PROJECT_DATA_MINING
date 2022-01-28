from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from data import data_set
import warnings

class MODEL():
    def __init__(self):
        self.svm=svm.SVC(kernel='linear', gamma=1)
        self.dtr=OneVsRestClassifier(DecisionTreeRegressor())
        self.lmt=OneVsRestClassifier(LogisticRegression())
        self.adb=AdaBoostClassifier(n_estimators=100, random_state=0)
        self.data=data_set()
        self.data_train, self.data_test, self.target_train, self.target_test=self.data.lebel_output()
class SVM(MODEL):

    def set_svm(self):
        self.svm.fit(self.data_train, self.target_train).predict(self.data_test)
    def out_svm_cross_validate(self):
       self.set_svm()
       cv_results=cross_validate(self.svm, self.data_train, self.target_train, cv=3,return_train_score=False)
       sorted(cv_results.keys())
       return cv_results['test_score']


class LMT(MODEL):
    def set_lmt(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.lmt.fit(self.data_train, self.target_train).predict(self.data_test)
    def out_lmt_cross_validate(self):
       self.set_lmt()
       cv_results=cross_validate(self.lmt, self.data_train, self.target_train, cv=3,return_train_score=False)
       sorted(cv_results.keys())
       return cv_results['test_score']
class DTR(MODEL):
    def set_dtr(self):
        self.dtr.fit(self.data_train, self.target_train).predict(self.data_test)
    def out_dtr_cross_validate(self):
       self.set_dtr()
       cv_results=cross_validate(self.dtr, self.data_train, self.target_train, cv=3,return_train_score=False)
       sorted(cv_results.keys())
       return cv_results['test_score']
class ADB(MODEL):
    def set_adb(self):
        self.adb.fit(self.data_train, self.target_train).predict(self.data_test)
    def out_adb_cross_validate(self):
       self.set_adb()
       cv_results=cross_validate(self.adb, self.data_train, self.target_train, cv=3,return_train_score=False)
       sorted(cv_results.keys())
       return cv_results['test_score']
    
    
