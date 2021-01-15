

# importing modules 
import numpy as np 
import matplotlib.pyplot as plt 

# Y-axis values 
y1 = [68,
136,
216]

y2 = [75,
167,
243]

# Y-axis values 
y3 = [75,
186,
265]

x = [5,
10,
15]

# Function to plot 
plt.plot(x,y1) 
plt.plot(x,y2) 
plt.plot(x,y3)
plt.xlabel('Number of Iterations')
plt.ylabel('Trajectory length')
plt.title('Field 1')
# Function add a legend 
plt.legend(["OSP = 0", "OSP = 0.5", "OSP = 1"], loc ="upper right") 

# function to show the plot 
plt.show() 

