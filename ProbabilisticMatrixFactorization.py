# -*- coding: utf-8 -*-
import numpy as np

class PMF(object):
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000):
        self.num_feat = num_feat  
        self.epsilon = epsilon  
        self._lambda = _lambda  
        self.momentum = momentum  
        self.maxepoch = maxepoch  
        self.num_batches = num_batches  
        self.batch_size = batch_size  

        self.w_Item = None  
        self.w_User = None  

        self.mae_train = []
        self.mae_test = []

    def fit(self, train_vec, test_vec):
        self.mean_inv = np.mean(train_vec[:, 2])  

        pairs_train = train_vec.shape[0]  
        pairs_test = test_vec.shape[0]  

        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(test_vec[:, 0]))) + 1  
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1  

        incremental = False  
        if ((not incremental) or (self.w_Item is None)):
            self.epoch = 0
            self.w_Item = 0.1 * np.random.randn(num_item, self.num_feat)  
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  

            self.w_Item_inc = np.zeros((num_item, self.num_feat))  
            self.w_User_inc = np.zeros((num_user, self.num_feat))  

        while self.epoch < self.maxepoch:  
            self.epoch += 1

            shuffled_order = np.arange(train_vec.shape[0])  
            np.random.shuffle(shuffled_order)  

            for batch in range(self.num_batches):  
                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  

                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                pred_out = np.sum(np.multiply(self.w_User[batch_UserID, :], self.w_Item[batch_ItemID, :]), axis=1)  
                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv  

                Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) + self._lambda * self.w_User[batch_UserID, :]
                Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], self.w_User[batch_UserID, :]) + self._lambda * (self.w_Item[batch_ItemID, :])  

                dw_Item = np.zeros((num_item, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat))

                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] += Ix_User[i, :]

                self.w_Item_inc = self.momentum * self.w_Item_inc + self.epsilon * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.epsilon * dw_User / self.batch_size

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc

                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(train_vec[:, 0], dtype='int32'), :], 
                                                  self.w_Item[np.array(train_vec[:, 1], dtype='int32'), :]), axis=1)  
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    self.mae_train.append(np.mean(np.abs(rawErr)))  

                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(test_vec[:, 0], dtype='int32'), :], 
                                                  self.w_Item[np.array(test_vec[:, 1], dtype='int32'), :]), axis=1)  
                    rawErr = pred_out - test_vec[:, 2] + self.mean_inv
                    self.mae_test.append(np.mean(np.abs(rawErr)))  

                    print('Training MAE: %f, Test MAE %f' % (self.mae_train[-1], self.mae_test[-1]))

    def predict(self, invID):
        return np.dot(self.w_Item, self.w_User[int(invID), :]) + self.mean_inv  

    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

    def topK(self, test_vec, k=10):
        inv_lst = np.unique(test_vec[:, 0])
        pred = {}
        for inv in inv_lst:
            if pred.get(inv, None) is None:
                pred[inv] = np.argsort(self.predict(inv))[-k:]  

        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]:
                intersection_cnt[test_vec[i, 0]] = intersection_cnt.get(test_vec[i, 0], 0) + 1
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32'))

        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k)
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

        return precision_acc / len(inv_lst), recall_acc / len(inv_lst)
