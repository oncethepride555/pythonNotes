def my_abs(x):
    # 判断参数的数据类型
    if not isinstance(x,(int,float)):
        raise TypeError('数据类型错误')
    if x >= 0:
        return x
    else:
        return -x
print(my_abs('abc'))