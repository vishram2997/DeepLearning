import numpy as np
color = ['B','W','G','Y','R','O']

c = color.copy()

c[1:]  =  [ x for x in color if(color.index(x) != len(color)-1)  ]
c[0] = color[5]
print(c)

    
     

Cubes =['L','C','R']
def getCube():
    cube = []
    for i in range(54):
        cube.append(i)
    cube = np.array(cube)
    cube = cube.reshape(54)
    return cube


import random

cube = getCube()
#random.shuffle(cube)
cube = cube.reshape(6,3,3)




def moveLR(cube, side, direction):
    if side =='L':
        s = 0
    if side == 'R':
        s = 2
    s = Cubes.index(side)
    
    if direction =='U':
        k = 3
        temp = cube[k-3][s]
        cube[k-3][s] = cube[k][s] 
        temp = cube
        cube[k][s]  = cube[k-1][s]
        cube[k-1][s] = cube[k-2][s]
        cube[k-2][s] = temp
    if direction == 'D': 
        k = 3   
        temp = cube[0][s]
        cube[k-3][s] = cube[k-2][s] 
        cube[k-2][s]  = cube[k-1][s]
        cube[k-1][s] = cube[k][s]
        cube[k][s] = temp
        
    return cube
            
            
            
def moveTB(cube, side, direction):
    if side =='T':
        s = 0
    if side == 'B':
        s = 2
    
    if direction =='L':
        k = 3
        temp = cube[k][0]
        cube[k][0] = cube[k+2][0]
        cube[k+2][0] = cube[2][0]
        cube[2][0] = cube[k+1][0]
        cube[k+1][0] = temp
        

        
        
    if direction == 'R': 
        k = 3   
        temp = cube[0][s]
        cube[k-3][s] = cube[k-2][s] 
        cube[k-2][s]  = cube[k-1][s]
        cube[k-1][s] = cube[k][s]
        cube[k][s] = temp
        
    return cube
            

            
#print(moveLR(cube,'R','U'))

