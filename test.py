import os
import argparse
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from models.GT import GT_Model
from models.GR import GR_Model
from models.encoder_build import encoder
from data.dataset import TestDataset
from utils import index

def evaluate_dataset(dataset_name, b_dir, t_dir, gt_model, gr_model, device, save_dir="./result"):
    if not b_dir or not os.path.exists(b_dir):
        return None, None, 0
    dataset = TestDataset(b_dir, t_dir)
    if len(dataset) == 0:
        return None, None, 0

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    psnr_sum, ssim_sum, count = 0.0, 0.0, 0
    out_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for image, gt, fname in loader:
            image = image.to(device)
            pred_r = gr_model(image)
            output_t, *_ = gt_model(image, pred_r)

            out_np = index.tensor2im(output_t)
            cv2.imwrite(os.path.join(out_dir, fname[0]), out_np)

            if t_dir and os.path.exists(t_dir):
                gt = gt.to(device)
                gt_np = index.tensor2im(gt)
                _, p, s = index.quality_assess(out_np, gt_np)
                psnr_sum += p
                ssim_sum += s
                count += 1

    if count > 0:
        avg_p = psnr_sum / count
        avg_s = ssim_sum / count
        print(f"[{dataset_name:<20}] PSNR: {avg_p:6.2f} dB | SSIM: {avg_s:.4f} (Đã test và lưu {count} ảnh -> {out_dir})", flush=True)
        return avg_p, avg_s, count
    else:
        print(f"[{dataset_name:<20}] Đã lưu kết quả suy luận ({len(dataset)} ảnh) vào: {out_dir}", flush=True)
        return None, None, len(dataset)

def get_default_test_root():
    candidates = [
        "../dataset500/testsets",
        "./dataset500/testsets",
        "dataset500/testsets",
        "../dataset500",
        "./dataset500",
        "dataset500",
        "/kaggle/input/datasets/congduc1122/pbl6-reflection-removal-duc/reflection-removal/test",
        "/kaggle/input/datasets/congduc1122/pbl6-reflection-removal-duc/reflection-removal/testsets",
        "/kaggle/input/datasets/congduc1122/pbl6-reflection-removal-duc/reflection-removal",
        "/kaggle/input/reflection-removal/testsets",
        "/kaggle/input/reflection-removal/test",
        "/kaggle/input/reflection-removal"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

def get_default_save_dir():
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/result"
    return "./result"

def get_default_ckpt():
    candidates = [
        "./checkpoint/best.pth",
        "/kaggle/working/checkpoint/best.pth",
        "../checkpoint/best.pth",
        "./checkpoint/epoch_150.pth",
        "/kaggle/working/checkpoint/epoch_150.pth",
        "./checkpoint/epoch_100.pth",
        "./checkpoint/epoch_050.pth",
        "./checkpoint/latest.pth",
        "/kaggle/working/checkpoint/latest.pth"
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PBL6 Reflection Removal Testing')
    parser.add_argument('--test_root', default=get_default_test_root(), help="Thư mục test")
    parser.add_argument('--ckpt_path', default=get_default_ckpt(), help="Đường dẫn file checkpoint (mặc định: best.pth)")
    parser.add_argument('--save_dir', default=get_default_save_dir(), help="Thư mục lưu ảnh kết quả")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("================================================================================")
    print("🚀 BẮT ĐẦU CHƯƠNG TRÌNH KIỂM THỬ (TEST)")
    print("================================================================================")
    print(f" • Thiết bị      : {device}")
    print(f" • Test root     : {args.test_root}")
    print(f" • Checkpoint    : {args.ckpt_path}")
    print(f" • Save dir      : {args.save_dir}")
    print("================================================================================")

    base_test_dir = args.test_root
    if os.path.isdir(os.path.join(args.test_root, "testsets")):
        base_test_dir = os.path.join(args.test_root, "testsets")
    elif os.path.isdir(os.path.join(args.test_root, "test")):
        base_test_dir = os.path.join(args.test_root, "test")

    gt_model = GT_Model(encoder().to(device), encoder().to(device)).to(device)
    gr_model = GR_Model(encoder().to(device)).to(device)

    if os.path.isfile(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location=device)
        gt_model.load_state_dict(ckpt['GT_state'])
        gr_model.load_state_dict(ckpt['GR_state'])
        ep_info = f" (Epoch {ckpt.get('best_epoch', ckpt.get('epoch', '?'))})"
        print(f"✅ Đã nạp checkpoint thành công từ: {args.ckpt_path}{ep_info}")
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy checkpoint tại {args.ckpt_path}, đánh giá với trọng số khởi tạo ban đầu.")

    gt_model.eval()
    gr_model.eval()

    all_psnr, all_ssim = [], []

    # 1. Dataset Nature (nếu có)
    nature_path = os.path.join(base_test_dir, "Nature")
    if os.path.exists(nature_path):
        p, s, _ = evaluate_dataset("Nature", os.path.join(nature_path, "blended"), os.path.join(nature_path, "transmission_layer"), gt_model, gr_model, device, args.save_dir)
        if p is not None: all_psnr.append(p); all_ssim.append(s)

    # 2. Dataset real20 (hoặc real20_420)
    for r20_name in ["real20", "real20_420"]:
        r20_path = os.path.join(base_test_dir, r20_name)
        if os.path.exists(r20_path):
            p, s, _ = evaluate_dataset(r20_name, os.path.join(r20_path, "blended"), os.path.join(r20_path, "transmission_layer"), gt_model, gr_model, device, args.save_dir)
            if p is not None: all_psnr.append(p); all_ssim.append(s)
            break

    # 3. Dataset real45 (ảnh thực tế không GT)
    real45_path = os.path.join(base_test_dir, "real45")
    if os.path.exists(real45_path):
        evaluate_dataset("real45", real45_path, None, gt_model, gr_model, device, args.save_dir)

    # 4. SIR2 Benchmark (Postcard, SolidObject, WildScene)
    for sir2_sub in ["PostcardDataset", "SolidObjectDataset", "WildSceneDataset"]:
        sub_candidates = [
            os.path.join(base_test_dir, sir2_sub),
            os.path.join(base_test_dir, "SIR2", sir2_sub)
        ]
        for sub_path in sub_candidates:
            if os.path.exists(sub_path):
                b_dir = os.path.join(sub_path, "blended")
                t_dir = os.path.join(sub_path, "transmission_layer")
                p, s, _ = evaluate_dataset(sir2_sub, b_dir, t_dir, gt_model, gr_model, device, args.save_dir)
                if p is not None: all_psnr.append(p); all_ssim.append(s)
                break

    if all_psnr:
        print("--------------------------------------------------------------------------------")
        print(f"⭐ TRUNG BÌNH TOÀN BỘ BENCHMARK (GT): PSNR = {np.mean(all_psnr):6.2f} dB | SSIM = {np.mean(all_ssim):.4f}")
        print("================================================================================")
