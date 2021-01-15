

# importing modules 
import numpy as np 
import matplotlib.pyplot as plt 

# Y-axis values 
y1 = [75,
170,
282,
386,
426,
453]

y2 = [71,
184,
282,
370,
453,
523]

# Y-axis values 
y3 = [99,
208,
313,
400,
510,
603]

x = [5,
10,
15,
20,
25,
30]

# Function to plot 
plt.plot(x,y1) 
plt.plot(x,y2) 
plt.plot(x,y3)
plt.xlabel('Number of Iterations')
plt.ylabel('Trajectory Lengths')
plt.title('Field 2')
# Function add a legend 
plt.legend(["radius = 5", "radius = 10", "radius = 15"], loc ="upper right") 

# function to show the plot 
plt.show() 

