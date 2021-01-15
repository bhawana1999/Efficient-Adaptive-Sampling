

# importing modules 
import numpy as np 
import matplotlib.pyplot as plt 

# Y-axis values 
y1 = [260.9290817,
10.59254042,
3.692138361]

y2 = [155.0260896,
44.42823038,
4.50270455]

# Y-axis values 
y3 = [356.4083837,
248.6976256,
177.8417734]

x = [5,
10,
15]

# Function to plot 
plt.plot(x,y1) 
plt.plot(x,y2) 
plt.plot(x,y3)
plt.xlabel('Number of Iterations')
plt.ylabel('Mapping Error')
plt.title('Field 2')
# Function add a legend 
plt.legend(["OSP = 0", "OSP = 0.5", "OSP = 1"], loc ="upper right") 

# function to show the plot 
plt.show() 

