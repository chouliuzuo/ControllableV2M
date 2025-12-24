import os
import numpy as np
import torch
from PIL import Image

input_dir = "./feature"
output_dir = "./tmp"
object_num = 5

entries = sorted(os.listdir(input_dir))
for files in entries:
    files_path = os.path.join(input_dir, files)
    if os.path.exists(os.path.join(output_dir, files)):
        continue
    if os.path.isdir(files_path):
        entries2  = sorted(os.listdir(files_path))
        for files2 in entries2:
            files_path2 = os.path.join(files_path, files2)
            if os.path.isdir(files_path2):
                hist_cat = None
                pos_cat = None
                pt_cat = None
                area_cat = None
                for i in range(object_num):
                    if not os.path.exists(os.path.join(files_path2, str(i)+".npz")):
                        break
                    npz_data = np.load(os.path.join(files_path2, str(i)+".npz"))
                    hist_b = npz_data["hist_b"]
                    hist_b = hist_b.T
                    hist_g = npz_data["hist_g"]
                    hist_g = hist_g.T
                    hist_r = npz_data["hist_r"]
                    hist_r = hist_r.T
                    hist = np.concatenate((hist_b, hist_g, hist_r), axis=0).reshape(1, 3, 256)
                    position_x = npz_data["position_x"]
                    position_y = npz_data["position_y"]
                    position = np.vstack((position_x, position_y))
                    position = position.reshape(1, position.shape[0], position.shape[1])
                    area = np.array(npz_data["area"]).reshape(1, 1)
                    if hist_cat is None:
                        hist_cat = hist
                    else:
                        hist_cat = np.concatenate((hist_cat, hist), axis=0)
                    if pos_cat is None:
                        pos_cat = position
                    else:
                        pos_cat = np.concatenate((pos_cat, position), axis=0)
                    if area_cat is None:
                        area_cat = area
                    else:
                        area_cat = np.concatenate((area_cat, area), axis=0)
                for i in range(object_num):
                    if not os.path.exists(os.path.join(files_path2, str(i)+".pt")):
                        break
                    pt_data = torch.load(os.path.join(files_path2, str(i)+".pt"))
                    if pt_cat is None:
                        pt_cat = pt_data
                    else:
                        pt_cat = torch.concat([pt_cat, pt_data], axis=0)
                print(hist_cat.shape, pos_cat.shape, pt_cat.shape)
                os.makedirs(os.path.join(output_dir, files), exist_ok=True)
                total_area = area_cat.sum()
                area_cat = area_cat / total_area
                position = pos_cat
                if position.shape[-1] == 0:
                    continue
                img_path = os.path.join(files_path, '0', '0.png')
                print(img_path)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"无法打开图片 {img_path}: {e}")
                tmp = np.array([height, width])
                s_pos = (position[:,:,0] / tmp - 0.5) * 2
                s_pos = s_pos.reshape(position.shape[0], position.shape[1], 1)
                position = position - position[:,:,0].reshape(position.shape[0], position.shape[1], 1)
                np.savez(
                    os.path.join(output_dir, files, files2 +  ".npz"),
                    **{"area": area_cat, "hist": hist_cat, "s_pos": s_pos, "position": position},
                )
                torch.save(pt_cat, os.path.join(output_dir, files, files2 +  ".pt"))