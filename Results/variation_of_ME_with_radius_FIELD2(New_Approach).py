# importing modules 
import numpy as np 
import matplotlib.pyplot as plt 

# Y-axis values 
y1 = [275.1980442,
121.8281798,
2.447252836,
1.767776717,
1.309525311,
1.244351904]

y2 = [278.0370238,
23.55044121,
10.90577167,
9.27625202,
8.568303546,
8.282813795]

# Y-axis values 
y3 = [275.8919993,
52.43543499,
15.42471578,
11.88775916,
10.77909639,
10.18856205]

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
plt.ylabel('Mapping Error')
plt.title('Field 2')
# Function add a legend 
plt.legend(["radius = 5", "radius = 10", "radius = 15"], loc ="upper right") 

# function to show the plot 
plt.show() 

