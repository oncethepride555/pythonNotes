from functools import reduce
# 字符串转化为整数
def str2int(s):
    DIGIST = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
    def char2num(c):
        return DIGIST[c]
    return reduce(lambda x,y:x*10+y,list(map(char2num,s)))
L = '123456'
print(str2int(L)) # 123456

# 字符串转化为浮点数
def str2float(s):
    DIGIST = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'.':-1}
    def char2num(c):
        return DIGIST[c]
    L = list(map(char2num,s)) # [1,2,3,-1,4,5,6]
    p = s.index('.') # 小数点的索引值 3
    L.remove(L[p]) # [1,2,3,4,5,6]
    return reduce(lambda x,y:x*10+y,L)*0.1**p # 得到的整数再乘以0.1的三次方
L1 = '123.456'
print(str2float(L1)) # 123.45600000000003

# 利用 map 函数实现 list 中的每个元素，首字母大写，其余小写
def normalize(name):
    return name[0].upper() + name[1:].lower()
L1 = ['abam','toM','LINDA']
L2 = list(map(normalize,L1))
print(L2)