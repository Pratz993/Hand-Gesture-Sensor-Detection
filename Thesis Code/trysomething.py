import pandas as pd 
try_df = pd.DataFrame({'cost': [250, 0, 100],'revenue': [100, 250, 0],'numbers' : [123,345,678]}, index=['A', 'B', 'C'])
print(try_df.loc[:, (try_df == 0).any(axis=0)])