import numpy as np
import pandas as pd
from numpy import linalg as la
from tqdm import tqdm

# 相似度计算函数
# 欧式距离
def euclidSim(x, y):
    return 1. / (1. + la.norm(x - y))


# 皮尔逊相关系数
def pearsSim(x, y):
    if len(x) < 3: return 1.
    return 0.5 + 0.5 * np.corrcoef(x, y, rowvar=0)[0][1]


# 余弦相似度
def cosSim(x, y):
    num = float(x.T * y)
    denom = la.norm(x) * la.norm(y)
    return 0.5 + 0.5 * (num / denom)


class CallaborativeFiltering:
    def __init__(self, data, is_matrix, sim_meas, users, rank=3, use_svd=False, n_sv=100):
        self.data = data
        self.sim_meas = sim_meas
        self.users = users
        self.rank = rank
        self.use_svd = use_svd
        self.n_sv = n_sv
        self.is_matrix = is_matrix

        if isinstance(self.data, np.ndarray): pass
        elif isinstance(self.data, list): self.data = np.asarray(self.data)
        else: raise ValueError('数据类型必须为list或numpy.ndarray!')

        if self.sim_meas not in [euclidSim, pearsSim, cosSim]:
            raise ValueError('相似度计算函数错误！')

        if self.use_svd and not self.is_matrix:
            raise ValueError('数据必须为矩阵才能使用SVD分解！')

        if self.is_matrix:
            self.num_users = int(self.data.shape[0])
            self.num_items = int(self.data.shape[1])
        else:
            self.num_users = int(max(self.data[:, 0]))
            self.num_items = int(max(self.data[:, 1]))
            self.data = self._get_data()
    
    def _get_data(self):
        num_data = self.data.shape[0]
        data_dic = {}
        for i in range(num_data):
            userId = int(self.data[i][0])
            itemId = int(self.data[i][1])
            rating = self.data[i][2]
            if userId not in data_dic:
                data_dic[userId] = [[itemId], [rating]]
            else:
                data_dic[userId][0].append(itemId)
                data_dic[userId][1].append(rating)
        return data_dic

    def _est(self, user, un_rated_item):
        """
        标准估计分数函数

        Parameters
        ----------
        user: 用户ID
        un_rated_item: 未评分过的物品ID

        Returns
        ----------
        score: user对un_rated_item的评分
        """
        if not self.is_matrix:
            rated_items = self.data[user][0]
            rating = self.data[user][1]

        if self.use_svd:
            U, sigma, VT = la.svd(self.data)
            Sig_n_sv = np.matrix(np.eye(n_sv) * Sigma[:n_sv])
            # 将原始矩阵映射为SVD降维后的新空间。item的个数=新矩阵的行数
            transformedItems = (dataMat.T * U[:, :n_sv] * Sig_n_sv.I).T

        sim_total = 0.
        rat_sim_total = 0.
        # 计算item与每个物品的相似度
        for i in range(self.num_items):
            if self.is_matrix:
                user_rating = self.data[user - 1, i]
                if user_rating == 0: continue
                if self.use_svd:
                    item_vec = transformedItems[:, un_rated_item - 1]
                    i_vec = transformedItems[:, i]
                # 获取同时给i商品和item商品评分的uesr索引值
                else:
                    both = np.nonzero(np.logical_and(self.data[:, un_rated_item - 1] > 0, self.data[:, i] > 0))[0]
                    item_vec = self.data[both, un_rated_item - 1]
                    i_vec = self.data[both, i]
            else:
                item_id = i + 1
                if item_id not in rated_items: continue
                user_rating = rating[rated_items.index(item_id)]    
                item_vec, i_vec = [], []
                for k, v in self.data.items():
                    if un_rated_item in v[0] and item_id in v[0]: 
                        item_vec.append(v[1][v[0].index(un_rated_item)])
                        i_vec.append(v[1][v[0].index(item_id)])
                item_vec, i_vec = np.asarray(item_vec), np.asarray(i_vec)
            if len(item_vec) == 0 or len(i_vec) == 0: similarity = 0
            # 否则计算两物品重合部分的相似度
            else:
                similarity = self.sim_meas(item_vec, i_vec)
            sim_total += similarity  # 累加相似度
            rat_sim_total += similarity * user_rating  # 累加（用户评分和相似度的乘积）
        if sim_total == 0:
            return 0
        else:
            return rat_sim_total / sim_total

    def _recommand_for_one_user(self, user):
        # 用户未评分的物品索引
        if self.is_matrix: un_rated_items = np.nonzero(self.data[user - 1, :] == 0)[0]
        else: un_rated_items = [i for i in range(self.num_items) if i + 1 not in self.data[user][0]]

        if len(un_rated_items) == 0:
            return

        item_scores = []
        for item in tqdm(un_rated_items):
            estimated_score = self._est(user, item)
            item_scores.append((item, estimated_score))
        recomm = sorted(item_scores, reverse = True)[:self.rank]
        return recomm

    def recommand(self):
        recomm_res = {}
        for user in self.users:
            recomm = self._recommand_for_one_user(user)
            recomm_res[user] = recomm
        return recomm_res



if __name__ == '__main__':
    # 读取数据
    data_path = 'data/ratings.csv'
    df = pd.read_csv(data_path, usecols=['userId', 'movieId', 'rating'])
    data = np.asarray(df.values)
    is_matrix = False
    
    # 相似度计算函数
    sim_meas = euclidSim

    # 待推荐用户
    users = [1, 2, 3, 4, 5]

    # 推荐个数
    rank = 3

    # 开始推荐
    cf = CallaborativeFiltering(data, is_matrix, sim_meas, users, rank)
    recomm_res = cf.recommand()
    print(recomm_res)