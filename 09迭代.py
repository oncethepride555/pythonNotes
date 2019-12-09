# 使用迭代查找一个list中最小和最大值，并返回一个tuple
def findMinAndMax(L):
    if len(L) == 0:
        return None, None
    else:
        min = L[0]
        max = L[0]
        for s in L:            
            if s < min:
                min = s
            if s >= max:
                max = s
        return min,max
# 测试
if findMinAndMax([]) != (None, None):
    print('测试失败!')
elif findMinAndMax([7]) != (7, 7):
    print('测试失败!')
elif findMinAndMax([7, 1]) != (1, 7):
    print('测试失败!')
elif findMinAndMax([7, 1, 3, 9, 5]) != (1, 9):
    print('测试失败!')
else:
    print('测试成功!')