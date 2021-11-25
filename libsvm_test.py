#   coding=utf-8
#   Author: Anys
#   Date:2021.11.19
#   b 1-4题代码

import libsvm.svmutil as svmutil

#  读取svmguide1数据集的数据
y_train,x_train = svmutil.svm_read_problem('dataset/svmguide1_train.txt',return_scipy=True)
y_test, x_test = svmutil.svm_read_problem('dataset/svmguide1_test.txt',return_scipy=True)

# 1.用默认参数进行训练svm分类器--model1
model1 = svmutil.svm_train(y_train,x_train)
predict1 = svmutil.svm_predict(y_test,x_test,model1)

# 2.使用 svm-scale 进行特征归一化后再训练 归一化到[0,1]
scale_para_train = svmutil.csr_find_scale_param(x_train,lower=0,upper=1)
scale_x_train = svmutil.csr_scale(x_train,scale_para_train)

scale_para_test = svmutil.csr_find_scale_param(x_test,lower=0,upper=1)
scale_x_test = svmutil.csr_scale(x_test,scale_para_test)

model2 = svmutil.svm_train(y_train,scale_x_train)
predict2 = svmutil.svm_predict(y_test,scale_x_test,model2)

# 3.使用线性核
model3 = svmutil.svm_train(y_train,x_train,'-t 0')
predict3 = svmutil.svm_predict(y_test,x_test,model3)

# 4.使用C=1000，RBF核
model4 = svmutil.svm_train(y_train,x_train,'-c 1000')
predict4 = svmutil.svm_predict(y_test,x_test,model4)


print("---------------------------------------")
print("---------------------------------------")
print('when using default setup, i.e. using 【RBF kernel】, and seeting 【C as 1】,the Accuracy is: %g%%'%predict1[1][0])
print('when scaling the data in [0,1], 【RBF kernel】, and seeting 【C as 1】,the Accuracy is: %g%%'%predict2[1][0])
print('when using【Linear kernel】 and seeting 【C as 1】, the Accuracy is: %g%%'%predict3[1][0])
print('when using【RBF kernel】 and seeting 【C as 1000】, the Accuracy is: %g%%'%predict4[1][0])
#