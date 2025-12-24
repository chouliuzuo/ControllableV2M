import numpy as np
from torch.utils.data import Dataset
import os
import torch
import torch.nn.functional as F

class MultiModalDataset(Dataset):
    def __init__(self, folder_path, music_feature_path, music_target_path, shot = 10, object = 5, second = 35):
        super().__init__()
        self.folder_path = folder_path
        self.music_feature_path = music_feature_path
        self.music_target_path = music_target_path
        self.data_files = [file for file in os.listdir(folder_path)]
        self.shot = shot
        self.object = object
        self.second = second
 
    def __len__(self):
        return len(self.data_files)
 
    def __getitem__(self, idx):
        data_file = os.path.join(self.folder_path, self.data_files[idx])
        shot_number = int(len(os.listdir(data_file)) / 2)
        if shot_number > self.shot:
            shot_number = self.shot
        semantics = []
        colors = []
        starts = []
        moves = []
        areas = []
        pre_second = 0
        for i in range(int(shot_number)):
            file = os.path.join(data_file, str(i)+'.pt')
            sem = torch.load(file)
            object_padding_num = self.object - sem.shape[0]
            object_padding1 = (0,0,0,object_padding_num)
            semantics.append(F.pad(sem, object_padding1, mode='constant', value=0))
            with np.load(file.replace('pt','npz')) as data:
                object_padding = (0,0,0,0,0,object_padding_num)
                time_padding_num = self.second - pre_second - data['position.npy'].shape[2]
                time_padding = (pre_second,time_padding_num,0,0,0,object_padding_num)
                colors.append(F.pad(torch.from_numpy(data['hist.npy']), object_padding,'constant',0))
                starts.append(F.pad(torch.from_numpy(data['s_pos.npy']), object_padding,'constant',0))
                moves.append(F.pad(torch.from_numpy(data['position.npy']), time_padding,'constant',0))
                areas.append(F.pad(torch.from_numpy(data['area.npy']), object_padding1,'constant',0))
                pre_second += data['position.npy'].shape[2]
        colors = torch.stack(colors)
        starts = torch.stack(starts)
        semantics = torch.stack(semantics)
        areas = torch.stack(areas)   
        moves = torch.stack(moves)
        # moves = torch.cat(moves, dim=-1).permute(2,0,1)
        
        shot_padding_num = self.shot - shot_number
        # time_padding_num = self.second - moves.shape[0]
        shot_padding = (0,0,0,0,0,shot_padding_num)
        # time_padding = (0,0,0,0,0,time_padding_num)
        shot_padding1 = (0,0,0,0,0,0,0,shot_padding_num)
        
        semantics = F.pad(semantics, shot_padding, mode='constant', value=0)
        colors = F.pad(colors, shot_padding1, mode='constant', value=0).reshape(-1,3,256)
        starts = F.pad(starts, shot_padding1, mode='constant', value=0).reshape(-1,2)
        moves = F.pad(moves, shot_padding1, mode='constant', value=0).reshape(-1,2,self.second).permute(2,0,1)
        areas = F.pad(areas, shot_padding, mode='constant', value=0).reshape(-1, 1)
        
        music_target_file = os.path.join(self.music_target_path, self.data_files[idx]+ '.pt')
        music_target = torch.load(music_target_file)
        if torch.rand(1).item() < 0.5:
            n = torch.randint(1, 201, (1,)).item()
            # randomly truncate the beginning part
            truncated_tensor = music_target[:, :, n:]
            music_target = F.pad(truncated_tensor, pad=(0,n), mode='constant', value=0)
        music_feature_file = os.path.join(self.music_feature_path, self.data_files[idx]+ '.npz')
        with np.load(music_feature_file) as data:
            loudness = torch.from_numpy(data['loudness.npy'])
            pitch = torch.from_numpy(data['pitch.npy'])
            chroma = torch.from_numpy(data['chroma.npy'])
            spectral_centroid = torch.from_numpy(data['spectral_centroid.npy'])
        return (semantics, colors, starts, moves, areas),  (pitch, chroma, loudness, spectral_centroid, music_target)
