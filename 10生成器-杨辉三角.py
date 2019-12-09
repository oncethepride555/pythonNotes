def triangles(max):
    L = [1]
    m = 0
    while m < max:
        yield L
        L1 = [1] # 每一行的开头的那个1
        for n in range(1,len(L)): # list(range(1,1))=[]
            L1.append(L[n-1]+L[n]) # +是相加，并不是连接符
        L1.append(1) # 每一行末尾的那个1
        L = L1
        m = m + 1
for t in triangles(10):
    print(t)