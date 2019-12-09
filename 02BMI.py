height = float(input('请输入您的身高（单位：米）：'))
weight = float(input('请输入您的体重（单位：公斤）：'))
BMI = weight/height**2
if BMI<18.5:
    print('BMI:%.1f,过轻' % BMI)
elif BMI>=18.5 and BMI<25:
    print('BMI:%.1f,正常' % BMI)
elif BMI>=25 and BMI<28:
    print('BMI:%.1f,过重' % BMI)
elif BMI>=28 and BMI<32:
    print('BMI:%.1f,肥胖' % BMI)
elif BMI>=32:
    print('BMI:%.1f,严重肥胖' % BMI)