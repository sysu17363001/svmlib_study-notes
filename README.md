# svmlib_study-notes
这是一份 《模式识别》--吴建鑫 P139_7.1习题的解答 
-------------------------------------------------
### /dataset 
里面是svmguide1数据

### /find_parameter 
里面是在区间范围内寻找最优的C和gamma的project
使用方法：首先需要下载gnuplot源代码，将下载得到的gnuplot文件夹拷贝 放入 find_parameter下面
然后就在命令模式下进入到find_parameter目录，再运行命令
python easy.py .\svmguide1_train.txt .\svmguide1_test.txt
，即可开始最优参数寻找，如果想更换数据集，把自己的数据集放入find_parameter下面
再把命令中的.\svmguide1_train.txt .\svmguide1_test.txt参数换成自己的就行

### imbalance_study.py
是一份探究不平衡训练数据对于svm性能的影响的相关python代码，以及使用参数w_i去对这种影响的消除

### libsvm_test.py
这是一份简单的libsvm函数的使用相关python代码
