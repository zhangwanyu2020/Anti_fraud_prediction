## Anti_fraud_prediction

##百度常规赛/反欺诈预测/xgboost

该项目来自百度常规赛：点击反欺诈预测。比赛详情请见官网：https://aistudio.baidu.com/aistudio/competition/detail/52

模型进行了特征工程和模型训练，模型最终采用xgboost。在此之前，有尝试逻辑回归/adaboost/lightGBM等，也尝试过模型融合，最终xgboost是表现最好的，其实不止13个version,至少20个以上，中间有些版本没有保存。

最终得分89.02，但version_13.py训练后预测不一定能达到此分数，但非常接近了，89.02的那一版被update掉了。

模型中采用了网格搜索，确定了最优的参数后又重新训练的，由于训练时长太久，只对n_estimators和max_depth进行了搜索。






