# machine-learning-basic
机器学习相关知识点

# 模型
1. 常见的传统机器学习模型
    1. LR
    2. Softmax
    3. SVM
    4. AdaBoostTree
    5. 决策树（c45,ID3） 
    6. 随机森林
    7. gbdt
    8. xgboost
    9. lightGBM
    10. ensemble集成模型（bagging，boosting，stacking）
    11. 关联规则挖掘（apriori, fp-growth）
    12. 半监督模型，基于图的模型
    13. 协同过滤
    
2. 神经网络模型
    1. MLP
    2. CNN
    3. RNN（LSTM）
    4. Attention
    5. GAN
    6. word2vector(层次霍夫曼树，负采样)， doc2vector
    7. fasttext
    8. ctr: LR, gbdt+LR, fm, ffm, PNN, Deep&Wide, DeepFm
    9. rank: point, pair(rankSVM), list, 基于文本生成的方案。
    10. online learning: FTRL(Follow-the-regularized-Leader)
    
3. 聚类模型
    1. kmeans
    2. 层次聚类
    
4. 主题模型
    1. plsa 
    2. LDA


# 优化策略
1. 对数损失函数
2. 平方损失函数
3. 指数损失函数
4. Hinge损失函数
5. 交叉熵损失函数

# 优化算法
1. 梯度下降（批量，随机，小批量）：自动调整学习率的（Adagrad，RMSprop，Adam）
2. 牛顿法，拟牛顿法
3. 共轭梯度法
4. 拉格朗日乘数法（约束优化问题）
5. SMO优化算法（SVM）

# 其他理论知识点
  1. 正则化，归一化
  2. 欠拟合与过拟合
  3. 特征选择（卡方检验）
  4. 距离计算方法
  5. BN，droupout
  
# NLP理论
  1. 词法，句法，语法
  2. CRF，HMM
  
# NLP应用
  1. 分词， 词性标注，命名实体识别
  2. 分类
  3. 聚类
  4. 搜索+排序
  5. 匹配（相似匹配，问答匹配）
  6. 推荐
  7. 摘要
  8. 文本生成
  9. 翻译
  10. 问答
  11. 推理
