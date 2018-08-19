import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt


def bangalore(targetdata):
  dfbangalore=pd.read_csv("bangalore.csv",header=0,na_values=['-'])
  dfbangalore.set_index('﻿Year',inplace=True)
  dfbangalore=dfbangalore.loc[1980:2017]
  
  dfbangalore=dfbangalore[['T','PP','V','RA','FG']]
  dfbangalore.interpolate(inplace=True)
  
  dfbangalore.reset_index(inplace=True)
  
  
  features = '﻿Year'
  target =  targetdata
  x=dfbangalore[features].reshape(-1,1)                  # To convert to 2D array
  y=dfbangalore[target].reshape(-1,1)
  
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=4)
  
  regressor = LinearRegression()
  regressor.fit(x_train,y_train)
  
  y_prediction = regressor.predict(x_test)
  print("Few Predicted value for the test case : ")
  print(y_prediction)
  
  slope, intercept = np.polyfit(dfbangalore[features],dfbangalore[target],1)
  
  year = float(input('Enter the year to predict the value : '))
  temp = intercept + (slope*year)
  print(temp)
  
  plt.scatter(x_test,y_test,color='black')
  plt.plot(x_test, slope*x_test + intercept, '-')
  plt.show()


def mumbai(targetdata):
  dfmumbai=pd.read_csv("mumbai.csv",header=0,na_values=['-'])
  dfmumbai.set_index('﻿Year',inplace=True)
  dfmumbai=dfmumbai.loc[1980:2017]
  
  dfmumbai=dfmumbai[['T','PP','V','RA','FG']]
  dfmumbai.interpolate(inplace=True)
  
  dfmumbai.reset_index(inplace=True)
  
  
  features = '﻿Year'
  target =  targetdata
  x=dfmumbai[features].reshape(-1,1)                  # To convert to 2D array
  y=dfmumbai[target].reshape(-1,1)
  
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=4)
  
  regressor = LinearRegression()
  regressor.fit(x_train,y_train)
  
  y_prediction = regressor.predict(x_test)
  print("Few Predicted value for the test case : ")
  print(y_prediction)
  
  slope, intercept = np.polyfit(dfmumbai[features],dfmumbai[target],1)
  
  year = float(input('Enter the year to predict the value : '))
  temp = intercept + (slope*year)
  print(temp)
  
  plt.scatter(x_test,y_test,color='black')
  plt.plot(x_test, slope*x_test + intercept, '-')
  plt.show()


def delhi(targetdata):
  dfdelhi=pd.read_csv("delhi.csv",header=0,na_values=['-'])
  dfdelhi.set_index('﻿Year',inplace=True)
  dfdelhi=dfdelhi.loc[1980:2017]
  
  dfdelhi=dfdelhi[['T','PP','V','RA','FG']]
  dfdelhi.interpolate(inplace=True)
  
  dfdelhi.reset_index(inplace=True)
  
  
  features = '﻿Year'
  target =  targetdata
  x=dfdelhi[features].reshape(-1,1)                  # To convert to 2D array
  y=dfdelhi[target].reshape(-1,1)
  
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=4)
  
  regressor = LinearRegression()
  regressor.fit(x_train,y_train)
  
  y_prediction = regressor.predict(x_test)
  print("Few Predicted value for the test case : ")
  print(y_prediction)
  
  slope, intercept = np.polyfit(dfdelhi[features],dfdelhi[target],1)
  
  year = float(input('Enter the year to predict the value : '))
  temp = intercept + (slope*year)
  print(temp)
  
  plt.scatter(x_test,y_test,color='black')
  plt.plot(x_test, slope*x_test + intercept, '-')
  plt.show()


def chennai(targetdata):
  dfchennai=pd.read_csv("chennai.csv",header=0,na_values=['-'])
  dfchennai.set_index('﻿Year',inplace=True)
  dfchennai=dfchennai.loc[1980:2017]
  
  dfchennai=dfchennai[['T','PP','V','RA','FG']]
  dfchennai.interpolate(inplace=True)
  
  dfchennai.reset_index(inplace=True)
  
  
  features = '﻿Year'
  target =  targetdata
  x=dfchennai[features].reshape(-1,1)                  # To convert to 2D array
  y=dfchennai[target].reshape(-1,1)
  
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=4)
  
  regressor = LinearRegression()
  regressor.fit(x_train,y_train)
  
  y_prediction = regressor.predict(x_test)
  print("Few Predicted value for the test case : ")
  print(y_prediction)
  
  slope, intercept = np.polyfit(dfchennai[features],dfchennai[target],1)
  
  year = float(input('Enter the year to predict the value : '))
  temp = intercept + (slope*year)
  print(temp)
  
  plt.scatter(x_test,y_test,color='black')
  plt.plot(x_test, slope*x_test + intercept, '-')
  plt.show()

def kolkata(targetdata):
  dfkolkata=pd.read_csv("kolkata.csv",header=0,na_values=['-'])
  dfkolkata.set_index('﻿Year',inplace=True)
  dfkolkata=dfkolkata.loc[1980:2017]
  
  dfkolkata=dfkolkata[['T','PP','V','RA','FG']]
  dfkolkata.interpolate(inplace=True)
  
  dfkolkata.reset_index(inplace=True)
  
  
  features = '﻿Year'
  target =  targetdata
  x=dfkolkata[features].reshape(-1,1)                  # To convert to 2D array
  y=dfkolkata[target].reshape(-1,1)
  
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=4)
  
  regressor = LinearRegression()
  regressor.fit(x_train,y_train)
  
  y_prediction = regressor.predict(x_test)
  print("Few Predicted value for the test case : ")
  print(y_prediction)
  
  slope, intercept = np.polyfit(dfkolkata[features],dfkolkata[target],1)
  
  year = float(input('Enter the year to predict the value : '))
  temp = intercept + (slope*year)
  print(temp)
  
  plt.scatter(x_test,y_test,color='black')
  plt.plot(x_test, slope*x_test + intercept, '-')
  plt.show()



print("\n\n\t\t\t\t\t\tWelcome to Weather Prediction\n\n")

while(True):
  print("1.Predict Bangalore's Weather  2.Predict Mumbai's Weather  3.Predict Delhi's Weather  4.Predict Chennai's Weather  5.Predict Kolkata's Weather  6.Exit")
  print("Enter the choice to Predict Weather : ")
  choice= int(input())

  if choice==1:
    while(True):
      print("1.Avg Temp  2.Total Rainfall  3.Avg Windspeed  4.Number of days with rain  5.Number of days with Fog  6.Exit")
      print("Enter the choice of target data : ")
      targetchoice = int(input())

      if targetchoice==1:
    	  bangalore('T')
    	  break
      elif targetchoice==2:
    	  bangalore('PP')
    	  break
      elif targetchoice==3:
    	  bangalore('V')
    	  break
      elif targetchoice==4:
    	  bangalore('RA')
    	  break
      elif targetchoice==5:
    	  bangalore('FG')
    	  break
      elif targetchoice==6:
    	  exit()
      else:
    	  print("Invalid Choice")
    	  break
    break

  if choice==2:
    while(True):
      print("1.Avg Temp  2.Total Rainfall  3.Avg Windspeed  4.Number of days with rain  5.Number of days with Fog  6.Exit")
      print("Enter the choice of target data : ")
      targetchoice = int(input())

      if targetchoice==1:
    	  mumbai('T')
    	  break
      elif targetchoice==2:
    	  mumbai('PP')
    	  break
      elif targetchoice==3:
    	  mumbai('V')
    	  break
      elif targetchoice==4:
    	  mumbai('RA')
    	  break
      elif targetchoice==5:
    	  mumbai('FG')
    	  break
      elif targetchoice==6:
    	  exit()
      else:
    	  print("Invalid Choice")
    	  break
    break


  if choice==3:
    while(True):
      print("1.Avg Temp  2.Total Rainfall  3.Avg Windspeed  4.Number of days with rain  5.Number of days with Fog  6.Exit")
      print("Enter the choice of target data : ")
      targetchoice = int(input())

      if targetchoice==1:
    	  delhi('T')
    	  break
      elif targetchoice==2:
    	  delhi('PP')
    	  break
      elif targetchoice==3:
    	  delhi('V')
    	  break
      elif targetchoice==4:
    	  delhi('RA')
    	  break
      elif targetchoice==5:
    	  delhi('FG')
    	  break
      elif targetchoice==6:
    	  exit()
      else:
    	  print("Invalid Choice")
    	  break
    break

  if choice==4:
    while(True):
      print("1.Avg Temp  2.Total Rainfall  3.Avg Windspeed  4.Number of days with rain  5.Number of days with Fog  6.Exit")
      print("Enter the choice of target data : ")
      targetchoice = int(input())

      if targetchoice==1:
    	  chennai('T')
    	  break
      elif targetchoice==2:
    	  chennai('PP')
    	  break
      elif targetchoice==3:
    	  chennai('V')
    	  break
      elif targetchoice==4:
    	  chennai('RA')
    	  break
      elif targetchoice==5:
    	  chennai('FG')
    	  break
      elif targetchoice==6:
    	  exit()
      else:
    	  print("Invalid Choice")
    	  break
    break

  if choice==5:
    while(True):
      print("1.Avg Temp  2.Total Rainfall  3.Avg Windspeed  4.Number of days with rain  5.Number of days with Fog  6.Exit")
      print("Enter the choice of target data : ")
      targetchoice = int(input())

      if targetchoice==1:
    	  kolkata('T')
    	  break
      elif targetchoice==2:
    	  kolkata('PP')
    	  break
      elif targetchoice==3:
    	  kolkata('V')
    	  break
      elif targetchoice==4:
    	  kolkata('RA')
    	  break
      elif targetchoice==5:
    	  kolkata('FG')
    	  break
      elif targetchoice==6:
    	  exit()
      else:
    	  print("Invalid Choice")
    	  break
    break

  if choice==6:
      exit()
  else:
      print("Invalid Choice") 	  






