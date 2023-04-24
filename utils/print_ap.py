'''
    save the mAP and AP into csv file
'''
import csv
import pandas as pd

def AP_to_csv(metrics,outpath):
    my_dict = metrics
    category = []
    ap = []
    result = []

    for key in my_dict.keys():
        category.append(key)
        ap.append(my_dict[key])
        result.append([key,my_dict[key]])


    test=pd.DataFrame(data=result)
    test.to_csv(outpath,header=False) 