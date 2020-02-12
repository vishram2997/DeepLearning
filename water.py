h =[0,1,0,2,1,0,1,3,2,1,2,1]



def trap(height):
    output=0
    for i in range(1,len(height)-1):
        if((max(height[:i]) > height[i]) & (max(height[i:])> height[i])):
            output += min([max(height[:i]),max(height[i:])])-height[i]   
    return output
            



def minWindow(s, t):
    sa = list(s.strip())
    ta = list(t.strip())
    ln = len(ta)
    win = []
    for k in range(ln,len(sa)):
        for i in range(len(sa)):
            win = sa[i:i+k]
            winCorrect=True
            for te in ta:
                if (te in win) & winCorrect:
                    winCorrect =True
                else:
                    winCorrect =False
            if winCorrect:
                return win
    


print(minWindow("ADOBECODEBANC", "ABC"))