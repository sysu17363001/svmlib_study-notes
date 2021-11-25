import libsvm.svmutil as svm
import matplotlib.pyplot as plt

# label=0 ： label=1 == ratio

y_train,x_train=svm.svm_read_problem('dataset/svmguide1_train.txt',return_scipy=True) # 2000 label=1 1089 label=0
y_test, x_test = svm.svm_read_problem('dataset/svmguide1_test.txt',return_scipy=True) # 2000个 label=0  2000个label=1
result1=[]
result=[]
for i in range(1,6):
    ratio = 0.1*i
    cut_off_position = 2000 + int(2000 * ratio)
    y_train_clip = y_train[0:cut_off_position]
    x_train_clip = x_train[0:cut_off_position]
    model = svm.svm_train(y_train_clip, x_train_clip)
    model1 = svm.svm_train(y_train_clip,x_train_clip,'-w0 %g -w1 1'%(10*ratio))
    result2 = svm.svm_predict(y_test,x_test,model1)
    result3 = svm.svm_predict(y_test,x_test,model)

    result1.append(result2[1][0])
    result.append(result3[1][0])

index=[1,2,3,4,5]
Ratio_value=['0.1','0.2','0.3','0.4','0.5']
plt.plot(index,result,label='without parameter w_0',marker='o')
plt.plot(index,result1,label="setting the w_0 as 10*ratio",marker='*')
plt.title("the performance of different ratio",fontsize=20)
plt.xlabel("Ratio",fontsize=18)
plt.xticks(index,Ratio_value)
plt.ylabel("Accuracy(%)",fontsize=18)
plt.legend()




plt.show()

