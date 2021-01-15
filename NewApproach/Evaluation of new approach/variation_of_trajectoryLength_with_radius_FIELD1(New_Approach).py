

# importing modules 
import numpy as np 
import matplotlib.pyplot as plt 

# Y-axis values 
y1 = [93,
138,
243,
346,
423,
443,
463,
483]

y2 = [49,
144,
229,
316,
385,
439,
484,
593]

# Y-axis values 
y3 = [76,
179,
269,
346,
443,
529,
599,
669]

x = [5,
10,
15,
20,
25,
30,
35,
40]

# Function to plot 
plt.plot(x,y1) 
plt.plot(x,y2) 
plt.plot(x,y3)
plt.xlabel('Number of Iterations')
plt.ylabel('Trajectory Lengths')
plt.title('Field 1')
# Function add a legend 
plt.legend(["radius = 5", "radius = 10", "radius = 15"], loc ="upper right") 

# function to show the plot 
plt.show() 

