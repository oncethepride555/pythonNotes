### 字典 dict 的使用
d = {'name':'wss','age':'24','height':'167cm'}
print(d['name'])

# -------判断 Key 的第1中方式---------
result1 = d.get('weight','不存在这个key') # get(参数1,参数2)  参数1：要判断是否存在的Key；参数2：不存在时返回的内容（不写的话默认为 None）。如果有该Key，返回它对应的value，如果没有，返回参数2
print(result1)
# -------判断 Key 的第2中方式---------
result2 = 'name' in d
print(result2) # True

### 删除用pop()
d.pop('height')
print(d) # {'name':'wss','age':'24'}