#import needed packages for reading and manipulating CSV File, Basic EDA and Visualization 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load and read no show appointments data and print out a few lines 
appointment_df= pd.read_csv('/C:\Users\eng\Desktop\data/noshowappointments.csv')

#print few lines to explor the data

appointment_df.head()