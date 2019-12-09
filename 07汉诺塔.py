# 参数1是起始柱上的盘子数，参数2是起始柱，参数3是中间柱，参数4是目的柱
# 总有办法将初始柱上的n-1个圆盘，借助目的柱而移动到中间柱上，进而我们就回到了最简单的n为1的情况。
def m(n,a,b,c):
    if n == 1:
        print(a,'-->',c)
    else:
        m(n-1,a,c,b) # 借助C柱，将n-1个圆盘从A柱移动到B柱
        m(1,a,b,c) # 将A柱最底层的圆盘移动到C柱
        m(n-1,b,a,c) # 借助A柱，将n-1个圆盘从B柱移动到C柱
m(2,'A','B','C')
print('-----------------')
m(3,'A','B','C')
print('-----------------')
m(4,'A','B','C')