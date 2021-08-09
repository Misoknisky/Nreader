### 2018百度机器阅读理解 (3RD)  

1.  下载数据集Dureader1.0  
2.  执行数据预处理
```
python run.py --prepare
```

3.  Train Model(BIDAF,QANET,RNET，DSQA)  
```
python run.py \
      --train \
      --algo=QANET
```