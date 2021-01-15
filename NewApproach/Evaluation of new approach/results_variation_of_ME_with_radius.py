# importing modules 
import numpy as np 
import matplotlib.pyplot as plt 

# Y-axis values 
y1 = [277.049416,
230.0072479,
12.43113326,
1.323092107,
1.259725059,
1.245843504,
1.199940892,
1.193178404]

y2 = [185.7768221,
13.73947978,
5.616630917,
3.581353683,
3.233240402,
3.210788261,
1.085439137,
0.5094252628]

# Y-axis values 
y3 = [76.15116379,
18.42646744,
9.867499347,
8.734971346,
6.777692546,
6.364095482,
6.274222327,
6.137048233]

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
plt.ylabel('Mapping Error')
# Function add a legend 
plt.legend(["radius = 5", "radius = 10", "radius = 15"], loc ="upper right") 

# function to show the plot 
plt.show() 

