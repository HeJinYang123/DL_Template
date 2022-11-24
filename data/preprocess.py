from math import ceil
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from config import ModelParams
from dataset import UNiLABDataset


def fill_nan(data):
    '''前后插值法填充缺失值'''
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                if i > 0 and i < data.shape[0] - 1:
                    if not np.isnan(data[i + 1, j]):
                        data[i, j] = (data[i - 1, j] + data[i + 1, j]) / 2
                    else:
                        data[i, j] = data[i - 1, j]
                elif i == 0 and not np.isnan(data[i + 1, j]):
                    data[i, j] = data[i + 1, j]
                elif i == data.shape[0] - 1 and not np.isnan(data[i - 1, j]):
                    data[i, j] = data[i - 1, j]
                else:
                    data[i, j] = -1
    return data


class Data_Processor:
    def __init__(self, args: ModelParams, mode='train'):
        self.args = args
        self.mode = mode
        self.sunshine = []
        self.temp = []
        self.wind_dir = []
        self.wind_spd = []

    def process(self):
        sunshine = pd.read_csv(self.args.sunshine_path).values.tolist()
        temp = pd.read_csv(self.args.temp_path).values.tolist()
        wind = pd.read_csv(self.args.wind_path, float_precision='high').values.tolist()

        day_range = None
        if self.mode == 'train':
            day_range = self.args.train_day_range
        elif self.mode == 'val':
            day_range = self.args.val_day_range
        elif self.mode == 'predict':
            day_range = self.args.infer_day_range
        else:
            raise ValueError('mode should be train or val')

        # index = 24 * day + (hour - 1)
        # hour: 1 ~ 24
        for day in range(day_range[1]):
            for hour in range(1, 25):
                self.temp.append(temp[day*24+hour-1][2])
                self.wind_dir.append(wind[day*24+hour-1][2])
                self.wind_spd.append(wind[day*24+hour-1][3])
                if 6 <= hour <= 20 and day < 300:
                    try:
                        self.sunshine.append(sunshine[day * 15 + hour-6][2])
                    except:
                        raise ValueError('day: {}, hour: {}'.format(day, hour))
                else:
                    self.sunshine.append(0)

        # Fill blank
        self.sunshine = fill_nan(np.array(self.sunshine, dtype=np.float64).reshape(-1, 1))
        self.temp = fill_nan(np.array(self.temp, dtype=np.float64).reshape(-1, 1))
        self.wind_dir = fill_nan(np.array(self.wind_dir, dtype=np.float64).reshape(-1, 1))
        self.wind_spd = fill_nan(np.array(self.wind_spd, dtype=np.float64).reshape(-1, 1))

        return self.sunshine, self.temp, self.wind_dir, self.wind_spd











