import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from models.GT import GT_Model
from models.GR import GR_Model
from models.encoder_build import encoder
from utils import index

class TestDataset(Dataset):
    def __init__(self, blended_dir, trans_dir=None):
        self.to_tensor = transforms.ToTensor()
        self.blended_files = sorted(os.listdir(blended_dir))
        self.blended_dir = blended_dir
        self.trans_dir = trans_dir

    def __getitem__(self, index):
        fname = self.blended_files[index]
        b_img = cv2.imread(os.path.join(self.blended_dir, fname))
        t_img = cv2.imread(os.path.join(self.trans_dir, fname)) if self.trans_dir else b_img.copy()
        return self.to_tensor(b_img.astype(np.float32) / 255.0), self.to_tensor(t_img.astype(np.float32) / 255.0), fname

    def __len__(self): return len(self.blended_files)


def evaluate_dataset(dataset_name, b_dir, t_dir, gt_model, gr_model, save_dir="./result"):
    loader = torch.utils.data.DataLoader(TestDataset(b_dir, t_dir), batch_size=1, shuffle=False)
    psnr_sum, ssim_sum, count = 0, 0, 0
    os.makedirs(os.path.join(save_dir, dataset_name), exist_ok=True)

    with torch.no_grad():
        for image, gt, fname in loader:
            image, gt = image.cuda(), gt.cuda()
            pred_r = gr_model(image)
            output_t, *_ = gt_model(image, pred_r)

            out_np, gt_np = index.tensor2im(output_t), index.tensor2im(gt)
            cv2.imwrite(os.path.join(save_dir, dataset_name, fname[0]), out_np)

            if t_dir:
                res, p, s = index.quality_assess(out_np, gt_np)
                psnr_sum += p; ssim_sum += s; count += 1

    if count > 0:
        print(f"[{dataset_name}] PSNR: {psnr_sum/count:.2f} dB | SSIM: {ssim_sum/count:.4f}")

if __name__ == '__main__':
    root = r"D:\PBL6\reflection-removal\test"
    gt_model, gr_model = GT_Model(encoder().cuda(), encoder().cuda()).cuda(), GR_Model(encoder().cuda()).cuda()
    
    ckpt = torch.load('./checkpoint/epoch_150.pth') # Chọn checkpoint model của em
    gt_model.load_state_dict(ckpt['GT_state']); gr_model.load_state_dict(ckpt['GR_state'])
    gt_model.eval(); gr_model.eval()

    # Đánh giá toàn bộ các tập test theo cấu trúc thư mục của em
    evaluate_dataset("Nature", os.path.join(root, "Nature", "blended"), os.path.join(root, "Nature", "transmission_layer"), gt_model, gr_model)
    evaluate_dataset("real20_420", os.path.join(root, "real20_420", "blended"), os.path.join(root, "real20_420", "transmission_layer"), gt_model, gr_model)
    evaluate_dataset("real45", os.path.join(root, "real45"), None, gt_model, gr_model)
    for sir2_sub in ["PostcardDataset", "SolidObjectDataset", "WildSceneDataset"]:
        sub_path = os.path.join(root, "SIR2", sir2_sub)
        if os.path.exists(sub_path):
            evaluate_dataset(sir2_sub, os.path.join(sub_path, "blended"), os.path.join(sub_path, "transmission_layer"), gt_model, gr_model)