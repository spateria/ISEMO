from Utils import Cell
import sys

def astar(cag,endp, allowun=0):

    e = cag.env
    startp = cag.Loc
    
    #m = e.grid
    m = e.w.situation
    w,h = e.size, e.size		
    
    sx,sy = startp 	#Start Point
    ex,ey = endp 	#End Point
    
    #[parent node, x, y, g, f]
    node = [None,sx,sy,0,abs(ex-sx)+abs(ey-sy)] 
    closeList = [node]
    createdList = {}
    createdList[sy*w+sx] = node
    k=0
    #print(closeList)
    while(closeList):
        node = closeList.pop(0)
        x = node[1]
        y = node[2]
        l = node[3]+1
        k+=1
        #find neighbours 
        if k!=0:
            neighbours = ((x,y+1),(x,y-1),(x+1,y),(x-1,y))
        else:
            neighbours = ((x+1,y),(x-1,y),(x,y+1),(x,y-1))
            
        for nx,ny in neighbours:
            if (nx==ex and ny==ey):
                path = [(ex,ey)]
                while node:
                    path.append((node[1],node[2]))
                    node = node[0]
                return list(reversed(path))            
                
            if 0<=nx<w and 0<=ny<h:
                if allowun and m[nx][ny]!=Cell.vacant and ([nx,ny] not in cag.env.w.attrList[Cell.unknown]):  ###allow unknown cells; but cell neither vacant nor unknown
                    1
                elif (not allowun) and m[nx][ny]!=Cell.vacant:
                    1
                elif ny*w+nx not in createdList:
                    nn = (node,nx,ny,l,l+abs(nx-ex)+abs(ny-ey))
                    createdList[ny*w+nx] = nn
                    #adding to closelist ,using binary heap
                    nni = len(closeList)
                    closeList.append(nn)
                    while nni:
                        i = (nni-1)>>1
                        if closeList[i][4]>nn[4]:
                            closeList[i],closeList[nni] = nn,closeList[i]
                            nni = i
                        else:
                            break
                            
    return []
    
