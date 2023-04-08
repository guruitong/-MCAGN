import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import ViTImageProcessor
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Feeder(torch.utils.data.Dataset):
    def __init__(self,
                 data_folder='MSED_clone',
                 train=True):
        super().__init__()
        self.PATH = data_folder
        self.train = train
        self.mod = 0
        if train:
            self.txt_df = self.load_txt_train()
            self.image_folder = os.path.join(self.PATH, 'train', 'images')
            self.image_folder2 = os.path.join(self.PATH, 'dev', 'images')
        else:
            self.txt_df = self.load_txt_test()
            self.image_folder = os.path.join(self.PATH, 'test', 'images')
        self.image_processor = ViTImageProcessor.from_pretrained("google/"
                                                                 "vit-base-patch16-224-in21k")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.le = LabelEncoder()
        self.txt_df['Sentiment'] = self.le.fit_transform(self.txt_df['Sentiment'])
        self.txt_df['Emotion'] = self.le.fit_transform(self.txt_df['Emotion'])
        self.txt_df['Desire'] = self.le.fit_transform(self.txt_df['Desire'])

    def load_txt_test(self):
        TXT_PATH = os.path.join(self.PATH, 'test', 'test.csv')
        return pd.read_csv(TXT_PATH)

    def load_txt_train(self):
        TXT_PATH = os.path.join(self.PATH, 'train', 'train.csv')
        TXT_PATH_DEV = os.path.join(self.PATH, 'dev', 'dev.csv')
        df1 = pd.read_csv(TXT_PATH)
        df2 = pd.read_csv(TXT_PATH_DEV)
        self.mod = len(df1)
        return pd.concat([df1, df2], ignore_index=True, axis=0)

    def __getitem__(self, idx):
        txt_inputs = []
        sentiments = []
        emotions = []
        desires = []
        for index in idx:
            txt_line = self.txt_df.iloc[index]
            sep = self.tokenizer.sep_token
            txt_input = txt_line['Caption'] + sep + txt_line['Title']
            txt_inputs.append(txt_input)
            sentiments.append(txt_line['Sentiment'])
            emotions.append(txt_line['Emotion'])
            desires.append(txt_line['Desire'])
        txt_inputs = self.tokenizer(txt_inputs, return_tensors="pt", padding=True).to(device)

        sentiments = np.array(sentiments)
        desires = np.array(desires)
        emotions = np.array(emotions)

        images = []
        for index in idx:
            if self.train and (index >= self.mod):
                index_zerod = index - self.mod
                im_path = os.path.join(self.image_folder2, str(index_zerod + 1) + '.jpg')
            else:
                im_path = os.path.join(self.image_folder, str(index + 1) + '.jpg')
            image = Image.open(im_path)
            image = np.array(image)
            images.append(image)
        images = self.image_processor.preprocess(images, return_tensors='pt').to(device)
        images = images.pixel_values
        return txt_inputs, images, sentiments, emotions, desires

    def __len__(self):
        return len(self.txt_df)