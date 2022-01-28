from models import *
import matplotlib.pyplot as plt
class main_code():
    def __init__(self):
        self.svm=SVM()
        self.lmt=LMT()
        self.dtr=DTR()
        self.adb=ADB()
    def output(self):
        svm=self.svm.out_svm_cross_validate()
        print("output SVM")
        print(svm)
        lmt=self.lmt.out_lmt_cross_validate()
        print("output LMT")
        print(lmt)
        dtr=self.dtr.out_dtr_cross_validate()
        print("output DTR")
        print(dtr)
        adb=self.adb.out_adb_cross_validate()
        print("output ADBOOST")
        print(adb)
        return svm,lmt,dtr,adb
    def show_output_system(self):
        svm,lmt,dtr,adb=self.output()
        data=[svm[0]*100,lmt[0]*100,dtr[0]*100,adb[0]*100]
        v=['SVM','LMT','DTR',"Adaboost"]
        xs=[i for i,_ in enumerate(v)]
        plt.bar(xs,data)
        plt.xticks(xs,v)
        plt.title("output")
        plt.xlabel("output")
        plt.show()
        
        



APP=main_code()
APP.show_output_system()
