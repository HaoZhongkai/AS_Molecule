import pandas as pd
import numpy as np


l = [[1,2,3,4],[5,6,7]]

df1s = []
for i in range(2):
    df1s.append(pd.DataFrame({i+10:l[i]}))
# pd1 = pd.DataFrame()

df1 = pd.concat(df1s,ignore_index=False,axis=1)
writer = pd.ExcelWriter('test.xlsx',engine='xlsxwriter')

df1.to_excel(writer)

writer.save()
