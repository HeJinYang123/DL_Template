import torch
import time
import os
from tqdm import tqdm
import pandas as pd

from data.dataloader import UNiLABDataset
from data.preprocess import Data_Processor
from modules.model import MyNet
from config import ModelParams
from modules.loss import ComputeLoss
from Logger.logger import Logger

global TIME
global logger

TIME = time.strftime("%Y-%m-%d-%H-%M")
logger = Logger(TIME).get_logger(__name__)

class Task:

    print(TIME)
    def __init__(self):
        self.logger = logger

        self.args = ModelParams()
        self.model = MyNet(self.args).to(self.args.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        loss_func = torch.nn.MSELoss()
        self.compute_loss = ComputeLoss(loss_func)

        self._make_dir()
        self._load_model(self.args.model_path)


    def _make_dir(self):
        current_dir = os.path.abspath(".")
        self.exp_dir = current_dir + "/results/exp_{}/".format(TIME)
        self.model_dir = self.exp_dir + "models/"

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def _load_model(self, path):
        if path is not None:
            self.model.load_state_dict(torch.load(path, map_location=self.args.device))
            logger.info('load model from {}'.format(path))
        else:
            return

    def train(self):
        self.logger.info('---------------Start training---------------')
        # drop
        self.logger.info("Data processing...")
        dataProcessor = Data_Processor(self.args)
        sunshine, temp, wind_dir, wind_spd = dataProcessor.process()

        # make dataset
        train_dataset = UNiLABDataset(self.args, sunshine, temp, wind_dir, wind_spd)
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            pin_memory=True,
        )
        dataProcessor = Data_Processor(self.args, mode='val')
        sunshine, temp, wind_dir, wind_spd = dataProcessor.process()
        val_dataset = UNiLABDataset(self.args, sunshine, temp, wind_dir, wind_spd)
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            pin_memory=True
        )

        logger.info(self.model)
        for i_epoch in range(self.args.epochs):
            logger.info('-------------------Epoch: {}-------------------'.format(i_epoch + 1))
            pbar = enumerate(train_data_loader)
            pbar = tqdm(pbar, total=len(train_data_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

            loss = 0
            for i, (feature, label) in pbar:
                feature = feature.to(self.args.device).unsqueeze(2)
                label = label.to(self.args.device)

                # Forward
                out = self.model(feature)
                pred = out
                total_loss = self.compute_loss(out, label)

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                # Update loss
                loss += total_loss

            logger.info(f"loss: {loss/(24*(self.args.train_day_range[1]-self.args.train_day_range[0]))}")
            if i_epoch % 10 == 0:
                self.validate(val_data_loader)
                torch.save(self.model.state_dict(), self.model_dir + f"model_{i_epoch}.pth")
            torch.save(self.model.state_dict(), self.exp_dir + f"last_model.pth")
        logger.info('-------------------Training finished-------------------')

        torch.save(
            self.model.state_dict(), "last_model.pth"
        )

    @torch.no_grad()
    def validate(self, val_data_loader):
        self.logger.info('Validating...')
        pbar = enumerate(val_data_loader)
        pbar = tqdm(pbar, total=len(val_data_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        loss = 0
        for i, (feature, label) in pbar:
            feature = feature.to(self.args.device).unsqueeze(2)
            label = label.to(self.args.device)
            # Forward
            out = self.model(feature)
            # Calculate accuracy
            loss += self.compute_loss(out, label)
        logger.info(f"loss: {loss/(24*(self.args.val_day_range[1]-self.args.val_day_range[0]))}")

    def predict(self):
        self.logger.info('---------------Start predicting---------------')
        self.logger.info("Data Loading...")
        dataProcessor = Data_Processor(self.args, mode='predict')
        sunshine, temp, wind_dir, wind_spd = dataProcessor.process()
        dataset = UNiLABDataset(self.args, sunshine, temp, wind_dir, wind_spd, mode='predict')
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            pin_memory=True
        )
        self.logger.info("Data Loading finished")
        self.logger.info("Predicting...")
        pbar = enumerate(dataloader)
        pbar = tqdm(pbar, total=len(dataloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        ans = ['Radiation']
        ans_with_time = [['Time', 'Radiation']]
        for i, (feature, label) in pbar:
            feature = feature.to(self.args.device).unsqueeze(2)  # torch.Size([1, 5, 1, 121])
            label = label.to(self.args.device)
            now_hour = feature[0][0][0][-1].item()
            # Forward
            out = self.model(feature)
            ans_with_time.append([now_hour, out.item()])
            if 6 <= now_hour <= 20:
                ans.append(out.item())

            # print(f'now_hour: {now_hour}, out: {out.item()}')

        self.logger.info("Predicting finished")
        pd.DataFrame(ans).to_csv(self.exp_dir + "sunshine_pred.csv", index=False, header=False)
        pd.DataFrame(ans_with_time).to_csv(self.exp_dir + "sunshine_pred_with_time.csv", index=False, header=False)
