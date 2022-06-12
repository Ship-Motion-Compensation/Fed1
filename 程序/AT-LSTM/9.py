import numpy as np
import pandas as pd

A = np.array([[1, 2], [3, 2], [1, 1], [3, 5], [5,2]])
data = pd.DataFrame(A)

writer = pd.ExcelWriter('A.xlsx')		# 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()

writer.close()