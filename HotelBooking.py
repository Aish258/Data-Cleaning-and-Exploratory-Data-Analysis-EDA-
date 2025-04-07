import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

link="https://raw.githubusercontent.com/swapnilsaurav/Dataset/refs/heads/master/hotel_bookings.csv"
df = pd.read_csv(link)
print ("1.Shape of data set",df.shape) #Gives number of columns and rows count
print ("df.datatypes:\n",df.dtypes) #Gives data types of all columns

#Divide all data columns into numberic and non-numeric columns
df_numeric=df.select_dtypes(include=[np.number])
print("List of numeric columns=\n",df_numeric) #Gives numeric columns list
df_non_numeric=df.select_dtypes(exclude=[np.number])
print("List of non-numeric columns=\n",df_non_numeric) ##Gives non-numeric columns list

#heatmap showing missing values of first 30 columns
cols=df.columns[:30]
colors=['#00ff00','#ffff00']
sns.heatmap(df[cols].isnull(),cmap=sns.color_palette(colors))
plt.savefig("My heatmapofhoteldatanullcolumns.png")
plt.show()

#calculating missing values of all columns
print("1.checking missing values in all columns:")
for col in df.columns:
    pct_missing=np.mean(df[col].isnull())*100
    print(f"{col}-{pct_missing}%")

print("2.checking columns only with missing values:")
for col in df.columns:
    pct_missing=np.mean(df[col].isnull())*100
    if pct_missing>0.0:
      print(f"{col}-{pct_missing}%") #Finds %of missing values in each column
      df[f'{col}_ismissing']=df[col].isnull() #Adding columns with missing values

print("2.Shape of the data set=",df.shape)
ismissing_cols= [col for col in df.columns if 'ismissing' in col]
print("Missing columns added:",ismissing_cols)
df['num_missing']=df[ismissing_cols].sum(axis=1)
print("3.Shape of the data set=",df.shape)

#Plotting bar graph of missing values
df['num_missing'].value_counts().sort_index().plot.bar(x='index',y='num_missing')
plt.show()

#Deleting rows which has more than 15 missing values
ind_missing=df[df['num_missing']>=15].index
print("Rows which has more than 15 missing values:\n",ind_missing)
df=df.drop(ind_missing,axis=0)
print("4.Shape of the data set=",df.shape)
plt.savefig("My bar graph of missing values in columns")

cols_to_drop=['company']
df=df.drop(cols_to_drop,axis=1)
print("2.checking columns only with missing values after row drop:")
for col in df.columns:
    pct_missing=np.mean(df[col].isnull())*100
    if pct_missing>0.0:
      print(f"{col}-{pct_missing}%")

'''
Handle them:
children-2.0498257606219004% -float64
babies-11.311318858061922% -float64
meal-11.467129071170085% -object
country-0.40879238707947996%-object
deposit_type-8.232810615199035% -object
agent-13.687005763302507% -float64
'''
#Handeling columns with missing values by replacing missing values
#First handeling numberic columns
med=df["children"].median()
df["children"]=df["children"].fillna(med)

med=df["babies"].median()
df["babies"]=df["babies"].fillna(med)

med=df["agent"].median()
df["agent"]=df["agent"].fillna(med)

#Handeling non-numeric values

non_numeric_cols=df_non_numeric.columns.values
for col in non_numeric_cols:
    pct_missing = np.mean(df[col].isnull()) * 100
    if pct_missing > 0.0:
        print("Replacing missing values for the columns:",col)
        top=df[col].describe()['top']
        df[col]=df[col].fillna(top)

print("4.Do we still have missing values?")
for col in df.columns:
    pct_missing=np.mean(df[col].isnull())*100
    if pct_missing>0.0:
      print(f"{col}-{pct_missing}%")

#Drop all the newly added columns
ismissing_cols= [col for col in df.columns if 'ismissing' in col]
df=df.drop(ismissing_cols,axis=1)
print("Final shape of the dataset:",df.shape)

#Identifying outliners for specific columns
df['total_of_special_requests'].hist(bins=100)
df.boxplot(column=['total_of_special_requests'])
plt.show()

df['id'].hist(bins=100)
df.boxplot(column=['id'])
plt.show()

#Checking ouliners for all columns together
sns.boxplot(data=df, orient='h')
plt.title('Boxplot for Outlier Detection in All Columns')
plt.tight_layout()
plt.figure(figsize=(12, 6))
plt.show()

#Drop duplicate rows
df=df.drop('id',axis=1).drop_duplicates()
print("shape of the dataset after dropping duplicates:",df.shape)








