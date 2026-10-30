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
    """Quản lý các tập ảnh thực tế (train/real, train/nature - chỉ có GT T, KHÔNG có GT R)."""
    def __init__(self, root_dirs, crop_size=256):
        super().__init__()
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()
        self.pairs = []

        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        for d in root_dirs:
            if not d or not os.path.exists(d):
                continue
            b_dir, t_dir = os.path.join(d, 'blended'), os.path.join(d, 'transmission_layer')
            if not os.path.exists(b_dir) or not os.path.exists(t_dir):
                continue
            t_files = set(os.listdir(t_dir))
            for fname in sorted(os.listdir(b_dir)):
                if is_image_file(fname) and fname in t_files:
                    self.pairs.append((os.path.join(b_dir, fname), os.path.join(t_dir, fname)))
        print(f"[RealPairDataset] Đã nạp {len(self.pairs)} cặp I-T thực tế.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        if len(self.pairs) == 0:
            dummy = torch.zeros((3, self.crop_size, self.crop_size), dtype=torch.float32)
            return dummy, dummy, dummy, False

        b_path, t_path = self.pairs[index % len(self.pairs)]
        b_img, t_img = cv2.imread(b_path), cv2.imread(t_path)
        while b_img is None or t_img is None:
            b_path, t_path = random.choice(self.pairs)
            b_img, t_img = cv2.imread(b_path), cv2.imread(t_path)

        short_side = random.randrange(self.crop_size, 448, 2)
        b_img = resize_short_side(b_img, short_side).astype(np.float32) / 255.0
        t_img = resize_short_side(t_img, short_side).astype(np.float32) / 255.0
        b_img, t_img = random_crop_pair(b_img, t_img, self.crop_size)

        if random.random() < 0.5:
            b_img, t_img = np.fliplr(b_img), np.fliplr(t_img)
        if random.random() < 0.3:
            b_img, t_img = np.flipud(b_img), np.flipud(t_img)
        k_rot = random.randint(0, 3)
        if k_rot > 0:
            b_img, t_img = np.rot90(b_img, k_rot), np.rot90(t_img, k_rot)

        r_dummy = torch.zeros((3, self.crop_size, self.crop_size), dtype=torch.float32)
        return (
            self.to_tensor(np.ascontiguousarray(b_img)),
            self.to_tensor(np.ascontiguousarray(t_img)),
            r_dummy,
            False
        )


class CurriculumSynDataset(Dataset):
    """
    Sinh ảnh Synthetic với Curriculum Learning và mô hình quang học phi tuyến (sRGB <-> Linear RGB).
    Tương thích cả dataset mới dataset500 (chứa transmission_layer & reflection_layer) lẫn VOC2012 cũ.
    """
    def __init__(self, syn_dir, crop_size=256):
        super().__init__()
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()
        self.difficulty_phase = 1
        self.t_paths = []
        self.r_paths = []

        if syn_dir and os.path.exists(syn_dir):
            t_sub = os.path.join(syn_dir, 'transmission_layer')
            r_sub = os.path.join(syn_dir, 'reflection_layer')
            if os.path.isdir(t_sub) and os.path.isdir(r_sub):
                self.t_paths = [os.path.join(t_sub, f) for f in sorted(os.listdir(t_sub)) if is_image_file(f)]
                self.r_paths = [os.path.join(r_sub, f) for f in sorted(os.listdir(r_sub)) if is_image_file(f)]
                print(f"[CurriculumSynDataset] Chế độ 2 thư mục: {len(self.t_paths)} ảnh T, {len(self.r_paths)} ảnh R.")
            else:
                all_imgs = [os.path.join(syn_dir, f) for f in sorted(os.listdir(syn_dir)) if is_image_file(f)]
                self.t_paths = all_imgs
                self.r_paths = all_imgs
                print(f"[CurriculumSynDataset] Chế độ VOC/đơn thư mục: {len(all_imgs)} ảnh nguồn.")
        else:
            print(f"[CurriculumSynDataset] Cảnh báo: Thư mục synthetic không tồn tại: {syn_dir}")

    def set_curriculum_phase(self, phase):
        self.difficulty_phase = phase

    def _sample_patch(self):
        if not self.r_paths:
            return np.zeros((self.crop_size, self.crop_size, 3), dtype=np.float32)
        r_path = random.choice(self.r_paths)
        r_raw = cv2.imread(r_path)
        while r_raw is None and len(self.r_paths) > 1:
            r_path = random.choice(self.r_paths)
            r_raw = cv2.imread(r_path)
        if r_raw is None:
            return np.zeros((self.crop_size, self.crop_size, 3), dtype=np.float32)

        scale_r = random.randrange(max(360, self.crop_size + 64), 540, 2)
        r_resized = resize_short_side(r_raw, scale_r).astype(np.float32) / 255.0
        h, w = r_resized.shape[:2]
        top = random.randint(0, max(0, h - self.crop_size))
        left = random.randint(0, max(0, w - self.crop_size))
        patch = r_resized[top:top+self.crop_size, left:left+self.crop_size]

        if random.random() < 0.5:
            patch = np.fliplr(patch)
        if random.random() < 0.3:
            patch = np.flipud(patch)
        k_rot = random.randint(0, 3)
        if k_rot > 0:
            patch = np.rot90(patch, k_rot)
        return patch

    def __len__(self):
        return len(self.t_paths)

    def __getitem__(self, index):
        if not self.t_paths:
            dummy = torch.zeros((3, self.crop_size, self.crop_size), dtype=torch.float32)
            return dummy, dummy, dummy, True

        t_path = self.t_paths[index % len(self.t_paths)]
        t_raw = cv2.imread(t_path)
        while t_raw is None:
            t_path = random.choice(self.t_paths)
            t_raw = cv2.imread(t_path)

        short_t = random.randrange(self.crop_size, 448, 2)
        t_resized = resize_short_side(t_raw, short_t).astype(np.float32) / 255.0
        t_patch, _ = random_crop_pair(t_resized, t_resized, self.crop_size)

        if random.random() < 0.5:
            t_patch = np.fliplr(t_patch)
        if random.random() < 0.3:
            t_patch = np.flipud(t_patch)
        k_rot = random.randint(0, 3)
        if k_rot > 0:
            t_patch = np.rot90(t_patch, k_rot)

        r_patch = self._sample_patch()

        # Pha trộn quang học Linear và Curriculum Learning qua 3 mức độ
        sigma = 3.5 if self.difficulty_phase == 1 else (2.5 if self.difficulty_phase == 2 else 1.5)
        alpha = random.uniform(0.2, 0.4) if self.difficulty_phase == 1 else random.uniform(0.3, 0.8)

        ksize = int(2 * round(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r_patch, (ksize | 1, ksize | 1), sigma)

        t_lin, r_lin = srgb_to_linear(t_patch), srgb_to_linear(r_blur)
        r_lin_scaled = alpha * np.clip((r_lin - 0.1) / 0.9, 0.0, 1.0)
        i_lin = np.clip(t_lin + r_lin_scaled, 0.0, 1.0)

        i_srgb = linear_to_srgb(i_lin)
        t_srgb = linear_to_srgb(t_lin)
        r_srgb = linear_to_srgb(r_lin_scaled)

        return (
            self.to_tensor(np.ascontiguousarray(i_srgb)),
            self.to_tensor(np.ascontiguousarray(t_srgb)),
            self.to_tensor(np.ascontiguousarray(r_srgb)),
            True
        )

# Giữ tên cũ VocCurriculumSynDataset để tương thích 100% với cấu trúc cũ
VocCurriculumSynDataset = CurriculumSynDataset


class CurriculumFusionDataset(Dataset):
    def __init__(self, syn_dataset, real_dataset, steps_per_epoch=500, syn_ratio=0.8):
        self.syn_dataset = syn_dataset
        self.real_dataset = real_dataset
        self.steps_per_epoch = steps_per_epoch
        self.syn_ratio = syn_ratio

    def update_schedule(self, epoch):
        if self.syn_dataset is not None:
            if epoch < 30:
                self.syn_dataset.set_curriculum_phase(1)
                self.syn_ratio = 0.80
            elif epoch < 80:
                self.syn_dataset.set_curriculum_phase(2)
                self.syn_ratio = 0.65
            else:
                self.syn_dataset.set_curriculum_phase(3)
                self.syn_ratio = 0.50

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        has_syn = self.syn_dataset is not None and len(self.syn_dataset) > 0
        has_real = self.real_dataset is not None and len(self.real_dataset) > 0

        if has_syn and (not has_real or random.random() < self.syn_ratio):
            return self.syn_dataset[random.randint(0, len(self.syn_dataset) - 1)]
        elif has_real:
            return self.real_dataset[random.randint(0, len(self.real_dataset) - 1)]
        else:
            raise RuntimeError("Cả synthetic dataset và real dataset đều không có dữ liệu!")


class TestDataset(Dataset):
    """Dataset phục vụ đánh giá (Validation/Test) trên các tập ảnh thực tế và benchmark."""
    def __init__(self, blended_dir, trans_dir=None):
        super().__init__()
        self.to_tensor = transforms.ToTensor()
        self.blended_dir = blended_dir
        self.trans_dir = trans_dir
        self.blended_files = []
        if blended_dir and os.path.exists(blended_dir):
            self.blended_files = [f for f in sorted(os.listdir(blended_dir)) if is_image_file(f)]

    def __getitem__(self, index):
        fname = self.blended_files[index]
        b_path = os.path.join(self.blended_dir, fname)
        b_img = cv2.imread(b_path)
        if b_img is None:
            b_img = np.zeros((256, 256, 3), dtype=np.uint8)

        orig_h, orig_w = b_img.shape[:2]
        max_side = max(orig_h, orig_w)

        # Điều chỉnh kích thước an toàn cho VRAM (max side <= 512, chia hết cho 32)
        if max_side > 512:
            scale = 512.0 / max_side
            eval_h = max(32, int(round(orig_h * scale / 32.0)) * 32)
            eval_w = max(32, int(round(orig_w * scale / 32.0)) * 32)
            b_eval = cv2.resize(b_img, (eval_w, eval_h), interpolation=cv2.INTER_AREA)
        else:
            eval_h = max(32, (orig_h // 32) * 32)
            eval_w = max(32, (orig_w // 32) * 32)
            if eval_h != orig_h or eval_w != orig_w:
                b_eval = cv2.resize(b_img, (eval_w, eval_h), interpolation=cv2.INTER_AREA)
            else:
                b_eval = b_img

        b_tensor = self.to_tensor(np.ascontiguousarray(b_eval.astype(np.float32) / 255.0))
        return b_tensor, fname, orig_h, orig_w

    def __len__(self):
        return len(self.blended_files)

