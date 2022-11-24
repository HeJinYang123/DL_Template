import numpy as np
from torch.utils.data import Dataset
from config import ModelParams


class UNiLABDataset(Dataset):
    def __init__(self, args: ModelParams, sunshine, temp, wind_dir, wind_spd, mode='train'):
        self.args = args
        self.sunshine = sunshine
        self.temp = temp
        self.wind_dir = wind_dir
        self.wind_spd = wind_spd
        self.mode = mode
        self.feature = []  # (channels=5, 1, days)
        for i in range(len(self.sunshine)):
            # 小时 气温 风向 风速 光照
            hour = (i - 1) % 24 + 1
            items = np.array([[hour], [self.temp[i]], [self.wind_dir[i]], [self.wind_spd[i]],
                              [self.sunshine[i]]], dtype=np.float32)
            self.feature.append(items)

    def __getitem__(self, index):
        # feature[index]: (channels=5, 1, index ~ index+look_back*24)
        if self.mode == 'train':
            feature = self.feature[index]
            for i in range(index+1, index+self.args.look_back_len*24+1):
                feature = np.append(feature, self.feature[i], axis=1)
            label = self.sunshine[index + self.args.look_back_len*24]
            # 当日辐射强度置零
            feature[-1][-1] = 0

            return feature, label
        if self.mode == 'predict':
            start = self.args.infer_day_range[0]*24 - self.args.look_back_len*24 + index
            feature = self.feature[start]
            for i in range(start+1, start + self.args.look_back_len * 24+1):
                feature = np.append(feature, self.feature[i], axis=1)
            label = self.sunshine[index + self.args.look_back_len * 24]
            feature[-1][-1] = 0
            return feature, label

    def __len__(self):
        if self.mode == 'train':
            return (self.args.train_day_range[1]-self.args.train_day_range[0])*24 - self.args.look_back_len*24
        elif self.mode == 'val':
            return (self.args.val_day_range[1]-self.args.val_day_range[0])*24
        elif self.mode == 'predict':
            return (self.args.infer_day_range[1]-self.args.infer_day_range[0])*24


