import os
import pandas as pd
import pickle
from torch.utils.data import Dataset,DataLoader
class Depression(Dataset):
    def __init__(self,data_dir:str,selected:bool=False,psm:bool=False):
        pms_root = 'data/VFT/PSM.xlsx'
        pms_res = pd.read_excel(pms_root)['Number'].to_list()


        self.data_list = []
        for file in os.listdir(data_dir):
            if psm and file[:-4] not in pms_res:
                continue
            with open(os.path.join(data_dir, file),'rb') as f:
                data = pickle.load(f)
                if selected:
                    if ((data['y'] == 0) or (data['y'] == 2) or (data['y'] == 3)):
                        self.data_list.append(data)
                else:
                    self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]