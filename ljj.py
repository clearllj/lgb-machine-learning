import pandas as pd
from tqdm import tqdm
import warnings
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
import json
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# 警告过滤器用于控制警告消息的行为。 忽略警告信息
warnings.filterwarnings('ignore')
DATA_PATH = 'data/'

# 读取train数据集
train = pd.read_csv(DATA_PATH + 'train_dataset.csv', sep='\t')

# 随机过采样
ros = RandomOverSampler(random_state=0)
train, train['risk_label'] = ros.fit_resample(train, train['risk_label'])

# 读取test训练集 从样本少的类别中随机抽样，再将抽样得来的样本添加到数据集 缺点：重复采样往往会导致严重的过拟合
test = pd.read_csv(DATA_PATH + 'test_dataset.csv', sep='\t')
# 将train和test数据集合并
data = pd.concat([train, test])

# 提取session_id中的数字作为ii
data['ii'] = data['session_id'].apply(lambda x: int(x[-7:-5]))

# location列转成多列
data['location_first_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['first_lvl'])
data['location_sec_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['sec_lvl'])
data['location_third_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['third_lvl'])

# 删除客户端类型和浏览器来源的无效数据
data.drop(['client_type', 'browser_source'], axis=1, inplace=True)
# 对于首次认证方式为空的数据进行填充
data['auth_type'].fillna('1', inplace=True)

# 标签编码LabelEncoder
for col in tqdm(['user_name', 'action', 'auth_type', 'ip',
                 'ip_location_type_keyword', 'ip_risk_level', 'location', 'device_model',
                 'os_type', 'os_version', 'browser_type', 'browser_version',
                 'bus_system_code', 'op_target', 'location_first_lvl', 'location_sec_lvl',
                 'location_third_lvl']):
    # 获取一个LabelEncoder
    lbl = LabelEncoder()
    # fit_transform训练LabelEncoder并使用训练好的LabelEncoder进行编码
    data[col] = lbl.fit_transform(data[col])

# 日期数据处理
data['op_date'] = pd.to_datetime(data['op_date'])
data['year'] = data['op_date'].dt.year
data['month'] = data['op_date'].dt.month
data['day'] = data['op_date'].dt.day
data['hour'] = data['op_date'].dt.hour
# 10 ** 9 转化为纳秒
data['op_ts'] = data["op_date"].values.astype(np.int64) // 10 ** 9
# 重排数据
data = data.sort_values(by=['user_name', 'op_ts']).reset_index(drop=True)
# 向下挪动一位
data['last_ts'] = data.groupby(['user_name'])['op_ts'].shift(1)
# ts_diff1列的值为操作时间减去结束时间
data['ts_diff1'] = data['op_ts'] - data['last_ts']

# 通过对各列的遍历，获取数据中不同的值
for f in ['ip', 'location', 'device_model', 'os_version', 'browser_version']:
    data[f'user_{f}_nunique'] = data.groupby(['user_name'])[f].transform('nunique')
# 双重循环遍历获取不同的方法值
for method in ['mean', 'max', 'min', 'std', 'sum', 'median']:
    for col in ['user_name', 'ip', 'location', 'device_model', 'os_version', 'browser_version']:
        data[f'ts_diff1_{method}_' + str(col)] = data.groupby(col)['ts_diff1'].transform(method)

group_list = ['user_name', 'ip', 'location', 'device_model', 'os_version', 'browser_version', 'op_target']

num_feature_list = ['ts_diff1']
# 特征工程
# 增加特征值
for group in group_list:
    for feature in num_feature_list:
        tmp = data.groupby(group)[feature].agg([sum, min, max, np.mean]).reset_index()
        tmp = pd.merge(data, tmp, on=group, how='left')
        data['{}-mean_gb_{}'.format(feature, group)] = data[feature] - tmp['mean']
        data['{}-min_gb_{}'.format(feature, group)] = data[feature] - tmp['min']
        data['{}-max_gb_{}'.format(feature, group)] = data[feature] - tmp['max']
        data['{}/sum_gb_{}'.format(feature, group)] = data[feature] / tmp['sum']

cat_cols = ['action', 'auth_type', 'browser_type',
            'browser_version', 'bus_system_code', 'device_model',
            'ip', 'ip_location_type_keyword', 'ip_risk_level', 'location', 'op_target',
            'os_type', 'os_version', 'user_name'
            ]

# 根据risk_label字段将训练集与测试集分开
train = data[~data['risk_label'].isna()].reset_index(drop=True)
test = data[data['risk_label'].isna()].reset_index(drop=True)
# lgb模型训练并预测

features = [i for i in train.columns if i not in ['risk_label', 'session_id', 'op_date', 'last_ts']]
y = train['risk_label']
# KFold是用于生成交叉验证的数据集的，而StratifiedKFold则是在KFold的基础上，
# 加入了分层抽样的思想，使得测试集和训练集有相同的数据分布，因此表现在算法上，
# StratifiedKFold需要同时输入数据和标签，便于统一训练集和测试集的分布
# 将数据分成5份 打乱顺序 随机数种子个数为2046
# StratifiedKFold能确保训练集，测试集中各类别样本的比例与原始数据集中相同。
KF = StratifiedKFold(n_splits=5, random_state=2022, shuffle=True)
# 模型参数
params = {
    # 目标（函数）这里用的是binary 二进制
    'objective': 'binary',
    'boosting_type': 'gbdt',  # 提升方法 选择的是 gbdt 表示传统的梯度提升决策树
    'metric': 'auc',    # 模型度量标准
    'n_jobs': -1,
    'learning_rate': 0.05,  # 学习率为0.05
    'num_leaves': 2 ** 6,    # 树的最大叶子节点数2的六次方。
    'max_depth': 8,   # 树的最大深度，控制过拟合的有效手段。
    'tree_learner': 'serial',    # 单个machine tree 学习器
    'colsample_bytree': 0.8,    # 在每棵树训练之前选择80% 的特征来训练。
    'subsample_freq': 1,    # 每1次执行bagging
    'subsample': 0.8,   # 在每棵树训练之前选择80% 的样本（非重复采样）来训练。
    'num_boost_round': 5000,  # boosting的迭代次数
    'max_bin': 255,     # 一个整数，表示最大的桶的数量。默认值为 255。lightgbm 会根据它来自动压缩内存。如max_bin=255 时，则lightgbm 将使用uint8 来表示特征的每一个值。
    'verbose': -1,  # 一个整数，表示是否输出中间信息 小于0，则仅仅输出critical 信息
    'seed': 2021,   # 随机数种子
    'bagging_seed': 2021,    # 表示bagging 的随机数种子
    'feature_fraction_seed': 2021,   # feature_fraction 的随机数种子
    'early_stopping_rounds': 100,   # 如果一个验证集的度量在100个 循环中没有提升，则停止训练。
}
# 初始化
oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros((len(test)))
# 5折交叉验证遍历
for fold_, (trn_idx, val_idx) in enumerate(KF.split(train.values, y.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=y.iloc[trn_idx])   # 测试集
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=y.iloc[val_idx])   # 验证集
    num_round = 3000
    clf = lgb.train(
        params,     # 用于训练的参数
        trn_data,   # 要训练的数据
        num_round,            # 提升迭代次数
        valid_sets=[trn_data, val_data],     # 数据集列表
        verbose_eval=100,      # 有效集上的评估指标在每100个提升阶段打印。
        early_stopping_rounds=50,     # 如果在50轮间指标不提升，那就提前停止
        categorical_feature=cat_cols    # 类别特征
    )
# 模型预测
    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)  # 预测值选择最好的结果
    predictions_lgb[:] += clf.predict(test[features], num_iteration=clf.best_iteration) / 5  # 取平均值
# 模型评估
# 准确率
print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
# 精确率  精确率是 tp / (tp + fp)的比例，其中tp是真正性的数量，fp是假正性的数量.
# 精确率直观地可以说是分类器不将负样本标记为正样本的能力
print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
# 召回率 实际为正的样本中被预测为正的样本所占实际为正的样本的比例
print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
# 精确率和召回率的加权平均值 F1 score越高 模型越稳健
print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))

# 将结果写入ljj.csv中 id从1到10000 后面跟的是预测值
test['ret'] = predictions_lgb
test['id'] = range(1, 10001)
test[['id', 'ret']].to_csv('ans/ljj.csv', index=False)
