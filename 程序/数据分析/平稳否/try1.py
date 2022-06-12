import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

filename = r'侯.csv'

f = pd.read_csv(filename, usecols=[5])
t = pd.read_csv(filename, usecols=[0])

# tem_result = adfuller(f)
# print(tem_result)
N = 450
for i in range(int(len(f)/N)-1):
    tem_result = adfuller(f[i*N:(i+1)*N])
    print(tem_result)
