

# importing modules 
import numpy as np 
import matplotlib.pyplot as plt 

# Y-axis values 
y1 = [36,
104,
161,
219,
286,
351]

y2 = [50,
124,
172,
241,
323,
393]

# Y-axis values 
y3 = [26,
45,
94,
164,
207,
281]

y4 = [23,
60,
132,
183,
230,
282
]

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
plt.plot(x,y4)
plt.xlabel('Number of Iterations')
plt.ylabel('Trajectory Lengths')
plt.title('Adaptive Radius Approach')
# Function add a legend 
plt.legend(["Field 1, starting radius = 5", "Field 1, starting radius = 10", "Field 2, starting radius = 5", "Field 2, starting radius = 10"], loc ="upper right") 

# function to show the plot 
plt.show() 

