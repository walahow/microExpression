import pandas as pd
import sys
import os

fpath = r'd:\proj\micpression\data\CASME2\CASME2-coding-20140508.xlsx'
try:
    df = pd.read_excel(fpath)
    df.columns = [c.strip() for c in df.columns]
    print("Original Excel Counts:")
    print(df['Estimated Emotion'].value_counts())
    
    with open('excel_dist.txt', 'w') as f:
        f.write(str(df['Estimated Emotion'].value_counts()))
except Exception as e:
    print(f"Error: {e}")
    with open('excel_dist.txt', 'w') as f:
        f.write(f"Error: {e}")
