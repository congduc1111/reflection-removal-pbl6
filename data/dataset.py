import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.utils import is_image_file, srgb_to_linear, linear_to_srgb

def random_crop_pair(img1, img2, crop_size=256):
    h, w = img1.shape[:2]
    top = 0 if h <= crop_size else random.randint(0, h - crop_size)
    left = 0 if w <= crop_size else random.randint(0, w - crop_size)
    return img1[top:top+crop_size, left:left+crop_size], img2[top:top+crop_size, left:left+crop_size]

def resize_short_side(img, target_short):
    h, w = img.shape[:2]
    if h < w:
        new_h, new_w = target_short, max(target_short, int(round(w * (target_short / h))))
    else:
        new_w, new_h = target_short, max(target_short, int(round(h * (target_short / w))))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


class RealPairDataset(Dataset):
    """Quản lý train/real và train/nature (chỉ có GT T, KHÔNG có GT R)."""
    def __init__(self, root_dirs, crop_size=256):
        super().__init__()
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()
        self.pairs = []

        for d in root_dirs:
            b_dir, t_dir = os.path.join(d, 'blended'), os.path.join(d, 'transmission_layer')
            if not os.path.exists(b_dir) or not os.path.exists(t_dir): continue
            t_files = set(os.listdir(t_dir))
            for fname in sorted(os.listdir(b_dir)):
                if is_image_file(fname) and fname in t_files:
                    self.pairs.append((os.path.join(b_dir, fname), os.path.join(t_dir, fname)))
        print(f"[RealPairDataset] Đã nạp {len(self.pairs)} cặp I-T thực tế.")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, index):
        b_path, t_path = self.pairs[index % len(self.pairs)]
        b_img, t_img = cv2.imread(b_path), cv2.imread(t_path)

        short_side = random.randrange(self.crop_size, 448, 2)
        b_img = resize_short_side(b_img, short_side).astype(np.float32) / 255.0
        t_img = resize_short_side(t_img, short_side).astype(np.float32) / 255.0
        b_img, t_img = random_crop_pair(b_img, t_img, self.crop_size)

        if random.random() < 0.5: b_img, t_img = np.fliplr(b_img), np.fliplr(t_img)
        if random.random() < 0.3: b_img, t_img = np.flipud(b_img), np.flipud(t_img)
        k_rot = random.randint(0, 3)
        if k_rot > 0: b_img, t_img = np.rot90(b_img, k_rot), np.rot90(t_img, k_rot)

        r_dummy = torch.zeros((3, self.crop_size, self.crop_size), dtype=torch.float32)
        return self.to_tensor(np.ascontiguousarray(b_img)), self.to_tensor(np.ascontiguousarray(t_img)), r_dummy, False


class VocCurriculumSynDataset(Dataset):
    """Sinh ảnh Synthetic từ VOC2012 với Curriculum Learning và lọc quang học chống ghép ngẫu nhiên thô."""
    def __init__(self, voc_png_dir, crop_size=256):
        super().__init__()
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()
        self.img_paths = [os.path.join(voc_png_dir, f) for f in sorted(os.listdir(voc_png_dir)) if is_image_file(f)]
        self.difficulty_phase = 1
        print(f"[VocCurriculumSynDataset] Tìm thấy {len(self.img_paths)} ảnh nguồn VOC2012.")

    def set_curriculum_phase(self, phase): self.difficulty_phase = phase

    def _sample_patch(self):
        r_path = random.choice(self.img_paths)
        r_raw = cv2.imread(r_path)
        if r_raw is None: return np.zeros((self.crop_size, self.crop_size, 3), dtype=np.float32)
        scale_r = random.randrange(max(360, self.crop_size + 64), 540, 2)
        r_resized = resize_short_side(r_raw, scale_r).astype(np.float32) / 255.0
        h, w = r_resized.shape[:2]
        top = random.randint(0, max(0, h - self.crop_size))
        left = random.randint(0, max(0, w - self.crop_size))
        return r_resized[top:top+self.crop_size, left:left+self.crop_size]

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, index):
        t_path = self.img_paths[index % len(self.img_paths)]
        t_raw = cv2.imread(t_path)
        while t_raw is None: t_raw = cv2.imread(random.choice(self.img_paths))

        short_t = random.randrange(self.crop_size, 448, 2)
        t_resized = resize_short_side(t_raw, short_t).astype(np.float32) / 255.0
        t_patch, _ = random_crop_pair(t_resized, t_resized, self.crop_size)
        r_patch = self._sample_patch()

        # Pha trộn quang học Linear
        sigma = 3.5 if self.difficulty_phase == 1 else (2.5 if self.difficulty_phase == 2 else 1.5)
        alpha = random.uniform(0.2, 0.4) if self.difficulty_phase == 1 else random.uniform(0.3, 0.8)
        
        ksize = int(2 * round(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r_patch, (ksize | 1, ksize | 1), sigma)

        t_lin, r_lin = srgb_to_linear(t_patch), srgb_to_linear(r_blur)
        r_lin_scaled = alpha * np.clip((r_lin - 0.1) / 0.9, 0.0, 1.0)
        i_lin = np.clip(t_lin + r_lin_scaled, 0.0, 1.0)

        return self.to_tensor(linear_to_srgb(i_lin)), self.to_tensor(linear_to_srgb(t_lin)), self.to_tensor(linear_to_srgb(r_lin_scaled)), True


class CurriculumFusionDataset(Dataset):
    def __init__(self, syn_dataset, real_dataset, steps_per_epoch=3000, syn_ratio=0.8):
        self.syn_dataset, self.real_dataset = syn_dataset, real_dataset
        self.steps_per_epoch, self.syn_ratio = steps_per_epoch, syn_ratio

    def update_schedule(self, epoch):
        if epoch < 30: self.syn_dataset.set_curriculum_phase(1); self.syn_ratio = 0.80
        elif epoch < 80: self.syn_dataset.set_curriculum_phase(2); self.syn_ratio = 0.65
        else: self.syn_dataset.set_curriculum_phase(3); self.syn_ratio = 0.50

    def __len__(self): return self.steps_per_epoch
    def __getitem__(self, index):
        if random.random() < self.syn_ratio:
            return self.syn_dataset[random.randint(0, len(self.syn_dataset) - 1)]
        return self.real_dataset[random.randint(0, len(self.real_dataset) - 1)]