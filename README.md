# text_matching

### 这个仓库主要是实现文本匹配方向的四个论文ABCNN,ESIM,BiMPM以及DRCN.数据集是常用的SNLI1.0和quora dataset，直接解压就行。

####　这个仓库包含四个文件夹，每个文件夹内各有独立的三个.py文件(main.py,data_process.py,model.py)


#### 每个文件夹内的main.py都可以直接运行，只需要注意文件路径即可。最后SNLI数据集上的实验结果如下
|模型|准确率  |
|--|--|
|  ESIM|82.4  |
|BiMPM|82.7|
|DRCN|80.5|
|ABCNN|74.6|

####　我没有用预训练的word2vec或者Glove，也没有考虑字符嵌入

我在CSDN上对每一个模型也做了说明，有兴趣的欢迎评论.
[DRCN](https://blog.csdn.net/m0_45478865/article/details/105880429)
[BiMPM](https://blog.csdn.net/m0_45478865/article/details/105806104)
[ESIM](https://blog.csdn.net/m0_45478865/article/details/105784839)
[ABCNN](https://blog.csdn.net/m0_45478865/article/details/105763451)
