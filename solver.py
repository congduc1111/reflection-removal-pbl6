import os
import glob
import time
import json
import csv
import torch
import cv2
import numpy as np
import torchvision.models as models
import torch.nn.functional as F

if os.name != 'nt':
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')  # Headless mode cho Kaggle / Linux server
import matplotlib.pyplot as plt

from models.GT import Discriminator, GT_Model
from models.GR import GR_Model
from models.encoder_build import encoder
from data.dataset import RealPairDataset, VocCurriculumSynDataset, CurriculumFusionDataset, TestDataset
from utils import index

EPS = 1e-12

def plot_training_curves(metrics_history, save_paths):
    """Vẽ biểu đồ huấn luyện và kiểm thử hoàn chỉnh 3 khung (Loss, PSNR, SSIM)."""
    epochs = metrics_history.get('epochs', [])
    train_loss = metrics_history.get('train_loss', [])
    eval_epochs = metrics_history.get('eval_epochs', [])
    mean_psnr = metrics_history.get('mean_psnr', [])
    mean_ssim = metrics_history.get('mean_ssim', [])
    datasets_data = metrics_history.get('datasets', {})
    best_epoch = metrics_history.get('best_epoch', None)
    best_psnr = metrics_history.get('best_psnr', None)
    best_ssim = metrics_history.get('best_ssim', None)

    if not epochs and not eval_epochs:
        return

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    # 1. Đồ thị Loss qua các Epochs
    ax = axes[0]
    if epochs and train_loss:
        ax.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=2)
        if max(epochs) >= 50:
            ax.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
            y_mid = (max(train_loss) + min(train_loss)) / 2 if train_loss else 0.5
            ax.text(25, y_mid, 'Stage 1\n(GR only)', ha='center', va='center', fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))
            ax.text(min(max(epochs), 100), y_mid, 'Stage 2\n(GR + GT)', ha='center', va='center', fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))
    ax.set_title('Training Loss vs Epochs', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    # 2. Đồ thị PSNR (dB)
    ax = axes[1]
    if eval_epochs and mean_psnr:
        colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']
        for idx, (ds_name, ds_vals) in enumerate(datasets_data.items()):
            ds_psnr = ds_vals.get('psnr', [])
            if len(ds_psnr) == len(eval_epochs):
                c = colors[idx % len(colors)]
                ax.plot(eval_epochs, ds_psnr, linestyle=':', alpha=0.6, label=ds_name, color=c)

        ax.plot(eval_epochs, mean_psnr, label='Mean PSNR', color='#2ca02c', linewidth=2.5, marker='o')

        if best_epoch and best_psnr is not None and best_epoch in eval_epochs:
            ax.scatter([best_epoch], [best_psnr], color='#d62728', s=160, zorder=5, marker='*')
            ax.annotate(f"Best: Ep {best_epoch}\n{best_psnr:.2f} dB",
                        xy=(best_epoch, best_psnr),
                        xytext=(max(1, best_epoch - 25), max(min(mean_psnr), best_psnr - 2.5)),
                        arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
                        fontsize=9, fontweight='bold', color='#8b0000',
                        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#d62728", lw=1))
    else:
        ax.text(0.5, 0.5, 'Waiting for evaluation...\n(every 5 epochs)', ha='center', va='center', transform=ax.transAxes, color='gray')
    ax.set_title('Validation PSNR (dB) vs Epochs', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('PSNR (dB)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=8)

    # 3. Đồ thị SSIM
    ax = axes[2]
    if eval_epochs and mean_ssim:
        colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']
        for idx, (ds_name, ds_vals) in enumerate(datasets_data.items()):
            ds_ssim = ds_vals.get('ssim', [])
            if len(ds_ssim) == len(eval_epochs):
                c = colors[idx % len(colors)]
                ax.plot(eval_epochs, ds_ssim, linestyle=':', alpha=0.6, label=ds_name, color=c)

        ax.plot(eval_epochs, mean_ssim, label='Mean SSIM', color='#e377c2', linewidth=2.5, marker='s')

        if best_epoch and best_ssim is not None and best_epoch in eval_epochs:
            ax.scatter([best_epoch], [best_ssim], color='#d62728', s=160, zorder=5, marker='*')
            ax.annotate(f"Best: Ep {best_epoch}\n{best_ssim:.4f}",
                        xy=(best_epoch, best_ssim),
                        xytext=(max(1, best_epoch - 25), max(min(mean_ssim), best_ssim - 0.05)),
                        arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
                        fontsize=9, fontweight='bold', color='#8b0000',
                        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#d62728", lw=1))
    else:
        ax.text(0.5, 0.5, 'Waiting for evaluation...\n(every 5 epochs)', ha='center', va='center', transform=ax.transAxes, color='gray')
    ax.set_title('Validation SSIM vs Epochs', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('SSIM', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    for p in save_paths:
        if p:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)

def save_metrics_to_csv(metrics_history, csv_path):
    """Lưu lịch sử metrics ra file CSV để dễ dàng theo dõi bằng Excel / Google Sheets."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    epochs = metrics_history.get('epochs', [])
    train_loss = metrics_history.get('train_loss', [])
    eval_epochs = metrics_history.get('eval_epochs', [])
    mean_psnr = metrics_history.get('mean_psnr', [])
    mean_ssim = metrics_history.get('mean_ssim', [])
    datasets_data = metrics_history.get('datasets', {})

    eval_dict = {}
    for i, ep in enumerate(eval_epochs):
        eval_dict[ep] = {
            'mean_psnr': mean_psnr[i] if i < len(mean_psnr) else '',
            'mean_ssim': mean_ssim[i] if i < len(mean_ssim) else '',
        }
        for ds_name, ds_vals in datasets_data.items():
            p_list = ds_vals.get('psnr', [])
            s_list = ds_vals.get('ssim', [])
            eval_dict[ep][f"{ds_name}_psnr"] = p_list[i] if i < len(p_list) else ''
            eval_dict[ep][f"{ds_name}_ssim"] = s_list[i] if i < len(s_list) else ''

    ds_names = list(datasets_data.keys())
    fieldnames = ['epoch', 'train_loss', 'is_eval', 'mean_psnr', 'mean_ssim']
    for ds_name in ds_names:
        fieldnames.extend([f"{ds_name}_psnr", f"{ds_name}_ssim"])

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, ep in enumerate(epochs):
            row = {
                'epoch': ep,
                'train_loss': f"{train_loss[i]:.6f}" if i < len(train_loss) else '',
                'is_eval': 1 if ep in eval_dict else 0,
                'mean_psnr': f"{eval_dict[ep]['mean_psnr']:.4f}" if ep in eval_dict and eval_dict[ep]['mean_psnr'] != '' else '',
                'mean_ssim': f"{eval_dict[ep]['mean_ssim']:.4f}" if ep in eval_dict and eval_dict[ep]['mean_ssim'] != '' else '',
            }
            for ds_name in ds_names:
                p_val = eval_dict.get(ep, {}).get(f"{ds_name}_psnr", '')
                s_val = eval_dict.get(ep, {}).get(f"{ds_name}_ssim", '')
                row[f"{ds_name}_psnr"] = f"{p_val:.4f}" if p_val != '' else ''
                row[f"{ds_name}_ssim"] = f"{s_val:.4f}" if s_val != '' else ''
            writer.writerow(row)


class Solver:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.max_time_seconds = getattr(self.args, 'max_hours', 10.0) * 3600.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Solver] Thiết bị đang sử dụng: {self.device}")

        self.eval_freq = getattr(self.args, 'eval_freq', 5)
        self.checkpoint_dir = getattr(self.args, 'checkpoint_dir', './checkpoint')
        self.save_dir = getattr(self.args, 'save_dir', './result')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        self.test_root = self._find_test_root(getattr(self.args, 'test_root', None))
        print(f"[Solver] Thư mục kiểm thử (Test Root): {self.test_root}")

        # Quản lý lịch sử chỉ số đánh giá
        self.metrics_history = {
            'epochs': [],
            'train_loss': [],
            'eval_epochs': [],
            'mean_psnr': [],
            'mean_ssim': [],
            'datasets': {},
            'best_epoch': 0,
            'best_psnr': -1.0,
            'best_ssim': -1.0
        }
        self.best_epoch = 0
        self.best_psnr = -1.0
        self.best_ssim = -1.0

        # Nạp lại metrics_history nếu tiếp tục train từ checkpoint
        self._load_metrics_history()

        self.encoder_I = encoder().to(self.device)
        self.encoder_R = encoder().to(self.device)
        self.encoder = encoder().to(self.device)

        self.vgg_feature = models.vgg19()
        vgg_path = self._find_vgg_ckpt(self.args.vgg_ckpt)
        if vgg_path and os.path.exists(vgg_path):
            print(f"[Solver] Nạp trọng số VGG-19 từ checkpoint cục bộ: {vgg_path}")
            vgg_state = torch.load(vgg_path, map_location='cpu')
            self.vgg_feature.load_state_dict(vgg_state)
        else:
            print("[Solver] Tải trực tiếp VGG-19 pretrained từ torchvision...")
            self.vgg_feature = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        self.vgg_feature.to(self.device).eval()
        for p in self.vgg_feature.parameters():
            p.requires_grad = False

    def _find_test_root(self, specified_path):
        candidates = [
            specified_path,
            "/kaggle/input/datasets/congduc1122/pbl6-dataset500/dataset500/dataset500/testsets",
            "/kaggle/input/pbl6-dataset500/dataset500/dataset500/testsets",
            "/kaggle/input/datasets/congduc1122/pbl6-dataset500/testsets",
            "/kaggle/input/pbl6-dataset500/testsets",
            os.path.join(getattr(self.args, 'data_root', ''), 'testsets'),
            os.path.join(getattr(self.args, 'data_root', ''), 'test'),
            os.path.join(os.path.dirname(getattr(self.args, 'data_root', '')), 'testsets'),
            os.path.join(os.path.dirname(getattr(self.args, 'data_root', '')), 'test'),
            "../dataset500/testsets",
            "./dataset500/testsets",
            "dataset500/testsets",
            "/kaggle/input/datasets/congduc1122/pbl6-reflection-removal-duc/reflection-removal/test",
            "/kaggle/input/datasets/congduc1122/pbl6-reflection-removal-duc/reflection-removal/testsets",
            "/kaggle/input/reflection-removal/testsets",
            "/kaggle/input/reflection-removal/test"
        ]
        for c in candidates:
            if c and os.path.isdir(c):
                return os.path.abspath(c)
        if os.path.isdir("/kaggle/input"):
            for root, dirs, _ in os.walk("/kaggle/input"):
                if "testsets" in dirs or ("real20" in dirs and "PostcardDataset" in dirs):
                    return os.path.abspath(os.path.join(root, "testsets") if "testsets" in dirs else root)
        return specified_path if specified_path else "./dataset500/testsets"

    def _find_vgg_ckpt(self, specified_path):
        candidates = [
            specified_path,
            "/kaggle/input/datasets/congduc1122/pbl6-dataset500/vgg19-dcbb9e9d.pth",
            "/kaggle/input/pbl6-dataset500/vgg19-dcbb9e9d.pth",
            './checkpoints/vgg19-dcbb9e9d.pth',
            '../checkpoints/vgg19-dcbb9e9d.pth',
            './checkpoint/vgg19-dcbb9e9d.pth',
            '../checkpoint/vgg19-dcbb9e9d.pth',
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'vgg19-dcbb9e9d.pth'),
            os.path.join(os.path.dirname(__file__), 'checkpoint', 'vgg19-dcbb9e9d.pth')
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        if os.path.isdir("/kaggle/input"):
            for root, _, files in os.walk("/kaggle/input"):
                for f in files:
                    if "vgg19" in f and f.endswith(".pth"):
                        return os.path.abspath(os.path.join(root, f))
        return None

    def _load_metrics_history(self):
        for candidate in [
            os.path.join(self.checkpoint_dir, 'metrics_history.json'),
            os.path.join(self.save_dir, 'metrics_history.json')
        ]:
            if os.path.isfile(candidate):
                try:
                    with open(candidate, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.metrics_history = data
                    self.best_epoch = data.get('best_epoch', 0)
                    self.best_psnr = data.get('best_psnr', -1.0)
                    self.best_ssim = data.get('best_ssim', -1.0)
                    print(f"📈 [Resume] Đã nạp lại lịch sử metrics: Best Epoch {self.best_epoch} (PSNR: {self.best_psnr:.2f} dB, SSIM: {self.best_ssim:.4f})")
                    break
                except Exception as e:
                    print(f"⚠️ Không thể đọc {candidate}: {e}")

    def get_test_configs(self):
        """Tự động phát hiện tất cả các tập dữ liệu test có trong test_root."""
        base_dir = self.test_root
        for candidate in [
            self.test_root,
            os.path.join(self.test_root, "testsets"),
            os.path.join(self.test_root, "test"),
        ]:
            if os.path.isdir(candidate):
                base_dir = candidate
                break

        configs = []

        # 1. Nature
        nature_p = os.path.join(base_dir, "Nature")
        if os.path.isdir(nature_p):
            configs.append({
                'name': 'Nature',
                'b_dir': os.path.join(nature_p, 'blended'),
                't_dir': os.path.join(nature_p, 'transmission_layer'),
                'has_gt': True
            })

        # 2. real20 / real20_420
        for r20 in ['real20', 'real20_420']:
            r20_p = os.path.join(base_dir, r20)
            if os.path.isdir(r20_p):
                configs.append({
                    'name': r20,
                    'b_dir': os.path.join(r20_p, 'blended'),
                    't_dir': os.path.join(r20_p, 'transmission_layer'),
                    'has_gt': True
                })
                break

        # 3. SIR2 Benchmark (Postcard, SolidObject, WildScene)
        for sir2 in ['PostcardDataset', 'SolidObjectDataset', 'WildSceneDataset']:
            for candidate in [os.path.join(base_dir, sir2), os.path.join(base_dir, 'SIR2', sir2)]:
                if os.path.isdir(candidate):
                    configs.append({
                        'name': sir2,
                        'b_dir': os.path.join(candidate, 'blended'),
                        't_dir': os.path.join(candidate, 'transmission_layer'),
                        'has_gt': True
                    })
                    break

        # 4. real45 (Tập ảnh thực tế không có Ground Truth)
        real45_p = os.path.join(base_dir, 'real45')
        if os.path.isdir(real45_p):
            configs.append({
                'name': 'real45',
                'b_dir': real45_p,
                't_dir': None,
                'has_gt': False
            })

        # 5. Tự động bổ sung bất kỳ thư mục con nào khác
        known = [c['name'] for c in configs]
        if os.path.isdir(base_dir):
            for entry in sorted(os.listdir(base_dir)):
                full_entry = os.path.join(base_dir, entry)
                if os.path.isdir(full_entry) and entry not in known and entry not in ['SIR2', 'train']:
                    b_cand = os.path.join(full_entry, 'blended')
                    t_cand = os.path.join(full_entry, 'transmission_layer')
                    if os.path.isdir(b_cand) and os.path.isdir(t_cand):
                        configs.append({'name': entry, 'b_dir': b_cand, 't_dir': t_cand, 'has_gt': True})
                    elif os.path.isdir(b_cand):
                        configs.append({'name': entry, 'b_dir': b_cand, 't_dir': None, 'has_gt': False})
                    else:
                        configs.append({'name': entry, 'b_dir': full_entry, 't_dir': None, 'has_gt': False})

        return configs

    def evaluate_single_dataset(self, b_dir, t_dir, save_images=False, out_dir=None):
        """Đánh giá PSNR và SSIM trên 1 tập dữ liệu, có tùy chọn lưu ảnh kết quả."""
        if not b_dir or not os.path.exists(b_dir):
            return None, None, 0
        dataset = TestDataset(b_dir, t_dir)
        if len(dataset) == 0:
            return None, None, 0

        if save_images and out_dir:
            os.makedirs(out_dir, exist_ok=True)

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        psnr_sum, ssim_sum, count = 0.0, 0.0, 0

        with torch.no_grad():
            for image, gt, fname in loader:
                image = image.to(self.device)
                pred_r = self.gr_model(image)
                output_t, *_ = self.gt_model(image, pred_r)

                out_np = index.tensor2im(output_t)
                if save_images and out_dir:
                    cv2.imwrite(os.path.join(out_dir, fname[0]), out_np)

                if t_dir and os.path.exists(t_dir):
                    gt = gt.to(self.device)
                    gt_np = index.tensor2im(gt)
                    _, p, s = index.quality_assess(out_np, gt_np)
                    psnr_sum += p
                    ssim_sum += s
                    count += 1

        avg_psnr = (psnr_sum / count) if count > 0 else None
        avg_ssim = (ssim_sum / count) if count > 0 else None
        return avg_psnr, avg_ssim, len(dataset)

    def run_eval_epoch(self, epoch):
        """Đánh giá định kỳ mỗi X epochs (mặc định 5 epochs), cập nhật đồ thị và lưu model tốt nhất."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.gt_model.eval()
        self.gr_model.eval()

        configs = self.get_test_configs()
        gt_configs = [c for c in configs if c['has_gt']]

        if not gt_configs:
            print(f"⚠️ [Epoch {epoch}] Không tìm thấy tập test có Ground Truth để tính PSNR/SSIM trong: {self.test_root}")
            return

        print(f"\n================================================================================")
        print(f"📊 [KIỂM THỬ ĐỊNH KỲ - EPOCH {epoch}/{self.args.num_epochs}]")
        print(f"--------------------------------------------------------------------------------")

        cur_psnrs, cur_ssims = [], []

        for cfg in gt_configs:
            ds_name = cfg['name']
            p, s, cnt = self.evaluate_single_dataset(cfg['b_dir'], cfg['t_dir'], save_images=False)
            if p is not None and s is not None:
                cur_psnrs.append(p)
                cur_ssims.append(s)
                if ds_name not in self.metrics_history['datasets']:
                    self.metrics_history['datasets'][ds_name] = {'psnr': [], 'ssim': []}
                self.metrics_history['datasets'][ds_name]['psnr'].append(p)
                self.metrics_history['datasets'][ds_name]['ssim'].append(s)
                print(f" • {ds_name:<20}: PSNR = {p:6.2f} dB | SSIM = {s:.4f} ({cnt} ảnh)")

        if cur_psnrs:
            mean_psnr = float(np.mean(cur_psnrs))
            mean_ssim = float(np.mean(cur_ssims))

            self.metrics_history['eval_epochs'].append(epoch)
            self.metrics_history['mean_psnr'].append(mean_psnr)
            self.metrics_history['mean_ssim'].append(mean_ssim)

            print(f"--------------------------------------------------------------------------------")
            print(f"⭐ TRUNG BÌNH TESTSET  : PSNR = {mean_psnr:6.2f} dB | SSIM = {mean_ssim:.4f}")

            # Kiểm tra và cập nhật mô hình tốt nhất
            if mean_psnr > self.best_psnr:
                self.best_psnr = mean_psnr
                self.best_ssim = mean_ssim
                self.best_epoch = epoch
                self.metrics_history['best_epoch'] = self.best_epoch
                self.metrics_history['best_psnr'] = self.best_psnr
                self.metrics_history['best_ssim'] = self.best_ssim

                best_save = {
                    'epoch': epoch,
                    'best_epoch': self.best_epoch,
                    'best_psnr': self.best_psnr,
                    'best_ssim': self.best_ssim,
                    'GT_state': self.gt_model.state_dict(),
                    'GR_state': self.gr_model.state_dict(),
                    'D_state': self.discriminator.state_dict(),
                    'GR_opt': self.GR_opt.state_dict(),
                    'G_opt': self.G_opt.state_dict(),
                    'D_opt': self.D_opt.state_dict()
                }
                best_ckpt_path = os.path.join(self.checkpoint_dir, 'best.pth')
                torch.save(best_save, best_ckpt_path)
                print(f"🏆 [KỶ LỤC MỚI] Đã lưu model tốt nhất tại Epoch {epoch} vào: {best_ckpt_path}")
            else:
                print(f"ℹ️ Model tốt nhất hiện tại: Epoch {self.best_epoch} (PSNR: {self.best_psnr:.2f} dB, SSIM: {self.best_ssim:.4f})")

            # Lưu file json và csv lịch sử
            self._save_metrics_files()

            # Vẽ và cập nhật đồ thị hoàn chỉnh
            self.plot_curves()
            print(f"📈 Đã cập nhật đồ thị: {os.path.join(self.save_dir, 'training_curves.png')}")
            print(f"================================================================================\n", flush=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _save_metrics_files(self):
        json_paths = [
            os.path.join(self.save_dir, 'metrics_history.json'),
            os.path.join(self.checkpoint_dir, 'metrics_history.json')
        ]
        for p in json_paths:
            try:
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
            except Exception as e:
                pass

        csv_paths = [
            os.path.join(self.save_dir, 'metrics_history.csv'),
            os.path.join(self.checkpoint_dir, 'metrics_history.csv')
        ]
        for p in csv_paths:
            try:
                save_metrics_to_csv(self.metrics_history, p)
            except Exception as e:
                pass

    def plot_curves(self):
        save_paths = [
            os.path.join(self.save_dir, 'training_curves.png'),
            os.path.join(self.checkpoint_dir, 'training_curves.png'),
            './training_curves.png'
        ]
        try:
            plot_training_curves(self.metrics_history, save_paths)
        except Exception as e:
            print(f"⚠️ Không thể vẽ biểu đồ: {e}")

    def run_final_test_and_save_results(self):
        """Tự động chọn Epoch tốt nhất, nạp trọng số, suy luận toàn bộ testset và lưu ảnh vào result/."""
        print(f"\n================================================================================")
        print(f"🎯 BẮT ĐẦU QUÁ TRÌNH KIỂM THỬ CUỐI CÙNG VỚI MÔ HÌNH TỐT NHẤT")
        print(f"================================================================================")

        best_ckpt_path = os.path.join(self.checkpoint_dir, 'best.pth')
        if os.path.isfile(best_ckpt_path):
            print(f"🌟 Nạp trọng số mô hình tốt nhất từ: {best_ckpt_path} (Epoch {self.best_epoch})")
            checkpoint = torch.load(best_ckpt_path, map_location=self.device)
            self.gt_model.load_state_dict(checkpoint['GT_state'])
            self.gr_model.load_state_dict(checkpoint['GR_state'])
        else:
            print(f"⚠️ Không tìm thấy file best.pth, sử dụng trọng số hiện tại của mô hình.")

        self.gt_model.eval()
        self.gr_model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        configs = self.get_test_configs()
        results_summary = []

        for cfg in configs:
            ds_name = cfg['name']
            b_dir = cfg['b_dir']
            t_dir = cfg['t_dir']
            out_dir = os.path.join(self.save_dir, ds_name)

            print(f"🚀 Đang suy luận và xuất ảnh cho tập [{ds_name}]...")
            p, s, cnt = self.evaluate_single_dataset(b_dir, t_dir, save_images=True, out_dir=out_dir)

            results_summary.append({
                'name': ds_name,
                'count': cnt,
                'psnr': p,
                'ssim': s,
                'out_dir': out_dir
            })

        # In bảng tổng kết cuối cùng
        report_lines = []
        report_lines.append("=" * 95)
        report_lines.append(f"🏆 BẢNG TỔNG KẾT KẾT QUẢ KIỂM THỬ CUỐI CÙNG (MODEL TỐT NHẤT - EPOCH {self.best_epoch})")
        report_lines.append("=" * 95)
        report_lines.append(f"{'Tên Tập Dữ Liệu':<22} | {'Số Ảnh':<8} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Thư Mục Lưu Ảnh Kết Quả'}")
        report_lines.append("-" * 95)

        valid_psnrs, valid_ssims, total_cnt = [], [], 0
        for r in results_summary:
            total_cnt += r['count']
            p_str = f"{r['psnr']:6.2f}" if r['psnr'] is not None else "  N/A "
            s_str = f"{r['ssim']:.4f}" if r['ssim'] is not None else "  N/A "
            if r['psnr'] is not None:
                valid_psnrs.append(r['psnr'])
                valid_ssims.append(r['ssim'])
            report_lines.append(f"{r['name']:<22} | {r['count']:<8} | {p_str:<10} | {s_str:<8} | {r['out_dir']}")

        report_lines.append("-" * 95)
        if valid_psnrs:
            m_p = float(np.mean(valid_psnrs))
            m_s = float(np.mean(valid_ssims))
            report_lines.append(f"{'⭐ TRUNG BÌNH (GT)':<22} | {sum(r['count'] for r in results_summary if r['psnr'] is not None):<8} | {m_p:6.2f}     | {m_s:.4f}   |")
        report_lines.append("=" * 95)
        report_lines.append(f"📁 Toàn bộ ảnh kết quả đã được xuất đầy đủ ra: {os.path.abspath(self.save_dir)}")
        report_lines.append(f"📈 Biểu đồ hoàn chỉnh đã được cập nhật tại: {os.path.abspath(os.path.join(self.save_dir, 'training_curves.png'))}")
        report_lines.append(f"✅ BẠN KHÔNG CẦN PHẢI CHẠY LẠI FILE test.py NỮA!")
        report_lines.append("=" * 95)

        report_text = "\n".join(report_lines)
        print(f"\n{report_text}\n", flush=True)

        report_file = os.path.join(self.save_dir, 'final_evaluation_report.txt')
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📝 Đã lưu báo cáo văn bản vào: {report_file}")
        except Exception as e:
            pass

        # Cập nhật đồ thị lần cuối
        self.plot_curves()

    def feat_loss(self, output_t, gt_t):
        feat_loss = 0
        conv_list = [2, 7, 12, 21, 30]
        sigma_list = [1/2.6, 1/4.8, 1/3.7, 1/5.6, 10/1.5]
        i = 0
        for index in range(31):
            output_t = self.vgg_feature.features[index](output_t)
            gt_t = self.vgg_feature.features[index](gt_t)
            if index in conv_list:
                feat_loss += self.l1_loss(output_t, gt_t) * sigma_list[i]
                i += 1
        return feat_loss

    def fft_loss(self, pred_t, gt_t):
        fft_pred = torch.fft.rfft2(pred_t, norm='backward')
        fft_gt = torch.fft.rfft2(gt_t, norm='backward')
        return self.l1_loss(torch.abs(fft_pred), torch.abs(fft_gt)) + 0.1 * self.l1_loss(torch.angle(fft_pred), torch.angle(fft_gt))

    def exclusion_loss(self, img_T, img_R, level=3):
        grad_x_loss, grad_y_loss = [], []
        for l in range(level):
            gx_T, gy_T = img_T[:,:,1:,:] - img_T[:,:,:-1,:], img_T[:,:,:,1:] - img_T[:,:,:,:-1]
            gx_R, gy_R = img_R[:,:,1:,:] - img_R[:,:,:-1,:], img_R[:,:,:,1:] - img_R[:,:,:,:-1]
            alphax = 2.0 * torch.mean(torch.abs(gx_T)) / (torch.mean(torch.abs(gx_R)) + 1e-6)
            alphay = 2.0 * torch.mean(torch.abs(gy_T)) / (torch.mean(torch.abs(gy_R)) + 1e-6)

            gradx1_s, grady1_s = (torch.sigmoid(gx_T) * 2) - 1, (torch.sigmoid(gy_T) * 2) - 1
            gradx2_s, grady2_s = (torch.sigmoid(gx_R * alphax) * 2) - 1, (torch.sigmoid(gy_R * alphay) * 2) - 1

            grad_x_loss.append(torch.mean(torch.mul(gradx1_s.pow(2), gradx2_s.pow(2))) ** 0.25)
            grad_y_loss.append(torch.mean(torch.mul(grady1_s.pow(2), grady2_s.pow(2))) ** 0.25)
            img_T = F.interpolate(img_T, scale_factor=0.5, mode='bilinear', align_corners=False)
            img_R = F.interpolate(img_R, scale_factor=0.5, mode='bilinear', align_corners=False)
        return (torch.sum(sum(grad_x_loss)/3) + torch.sum(sum(grad_y_loss)/3)) / 2

    def l1_loss(self, input, output):
        return torch.mean(torch.abs(input - output))

    def convert_L(self, img):
        return 0.114 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.299 * img[:, 2:3, :, :]

    def thr_loss(self, list_map, list_R, list_mask, list_mask_encoder, list_map_encoder):
        mask_loss, mask_encoder_loss = 0, 0
        for i in range(4):
            mask_loss += self.l1_loss(list_map[i] * list_mask[i], list_map[i] * 1)
            mask_encoder_loss += self.l1_loss(list_map_encoder[i] * list_mask_encoder[i], list_map_encoder[i] * 0)
        return mask_loss + mask_encoder_loss

    def _find_train_dirs(self):
        real_dirs = []
        syn_dir = None

        search_roots = [
            self.args.data_root,
            os.path.join(getattr(self.args, 'data_root', ''), "dataset500"),
            os.path.join(getattr(self.args, 'data_root', ''), "dataset500", "dataset500"),
            "/kaggle/input/datasets/congduc1122/pbl6-dataset500/dataset500/dataset500",
            "/kaggle/input/pbl6-dataset500/dataset500/dataset500",
            "/kaggle/input/datasets/congduc1122/pbl6-dataset500",
            "/kaggle/input/pbl6-dataset500",
            "../dataset500",
            "./dataset500",
            "dataset500",
        ]

        # 1. Tìm real_dirs
        for root in search_roots:
            if not root or not os.path.isdir(root):
                continue
            for cand in [
                os.path.join(root, "train", "real"),
                os.path.join(root, "real"),
                os.path.join(root, "train", "nature"),
                os.path.join(root, "nature")
            ]:
                if os.path.isdir(cand) and cand not in real_dirs:
                    if os.path.isdir(os.path.join(cand, 'blended')) or any(os.path.isdir(os.path.join(cand, sub)) for sub in ['transmission_layer']):
                        real_dirs.append(cand)

        # 2. Tìm syn_dir
        for root in search_roots:
            if not root or not os.path.isdir(root):
                continue
            for cand in [
                os.path.join(root, "train", "synthetic"),
                os.path.join(root, "synthetic"),
                os.path.join(root, "train", "VOCdevkit", "VOC2012", "PNGImages"),
                os.path.join(root, "VOCdevkit", "VOC2012", "PNGImages")
            ]:
                if os.path.isdir(cand):
                    syn_dir = cand
                    break
            if syn_dir:
                break

        # 3. Quét tự động trong /kaggle/input nếu vẫn chưa thấy
        if not real_dirs or not syn_dir:
            if os.path.isdir("/kaggle/input"):
                for cur_root, dirs, _ in os.walk("/kaggle/input"):
                    if not real_dirs and "real" in dirs:
                        cand_real = os.path.join(cur_root, "real")
                        if os.path.isdir(os.path.join(cand_real, "blended")):
                            real_dirs.append(cand_real)
                    if not syn_dir and "synthetic" in dirs:
                        cand_syn = os.path.join(cur_root, "synthetic")
                        if os.path.isdir(os.path.join(cand_syn, "reflection_layer")) or os.path.isdir(os.path.join(cand_syn, "transmission_layer")):
                            syn_dir = cand_syn

        return real_dirs, syn_dir

    def train_model(self):
        self.gr_model = GR_Model(self.encoder).to(self.device)
        self.gt_model = GT_Model(self.encoder_I, self.encoder_R).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.GR_opt = torch.optim.Adam(self.gr_model.parameters(), lr=self.args.lr)
        self.G_opt = torch.optim.Adam(list(self.gt_model.parameters()) + list(self.gr_model.parameters()), lr=self.args.lr)
        self.D_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr)

        # Xử lý Resume Checkpoint
        if self.args.resume and os.path.exists(self.args.resume):
            print(f"🔄 Đang nạp lại checkpoint từ: {self.args.resume}", flush=True)
            checkpoint = torch.load(self.args.resume, map_location=self.device)
            self.gt_model.load_state_dict(checkpoint['GT_state'])
            self.gr_model.load_state_dict(checkpoint['GR_state'])
            if 'D_state' in checkpoint:
                self.discriminator.load_state_dict(checkpoint['D_state'])
            if 'G_opt' in checkpoint:
                self.G_opt.load_state_dict(checkpoint['G_opt'])
            if 'GR_opt' in checkpoint:
                self.GR_opt.load_state_dict(checkpoint['GR_opt'])
            if 'D_opt' in checkpoint:
                self.D_opt.load_state_dict(checkpoint['D_opt'])
            self.start_epoch = checkpoint.get('epoch', 0)
            print(f"✅ Đã nạp thành công! Tiếp tục train từ Epoch {self.start_epoch + 1}", flush=True)

        # Tự động tìm kiếm thư mục dữ liệu train
        real_dirs, syn_dir = self._find_train_dirs()

        print(f"[Solver] Thư mục ảnh thực tế tìm thấy: {real_dirs}")
        print(f"[Solver] Thư mục ảnh tổng hợp tìm thấy: {syn_dir}")

        if not real_dirs and not syn_dir:
            raise RuntimeError(
                f"❌ LỖI DỮ LIỆU: Không tìm thấy thư mục 'train/real' hoặc 'train/synthetic'!\n"
                f"Data Root được truyền: '{self.args.data_root}'\n"
                f"Vui lòng kiểm tra lại đường dẫn dataset trong /kaggle/input."
            )

        train_real_dataset = RealPairDataset(real_dirs, crop_size=256) if real_dirs else None
        train_syn_dataset = VocCurriculumSynDataset(syn_dir, crop_size=256) if syn_dir else None

        steps_per_epoch = getattr(self.args, 'steps_per_epoch', 500)
        self.train_fusion_dataset = CurriculumFusionDataset(train_syn_dataset, train_real_dataset, steps_per_epoch=steps_per_epoch, syn_ratio=0.8)

        num_workers = 2 if os.name != 'nt' else 0
        train_loader = torch.utils.data.DataLoader(
            self.train_fusion_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda')
        )

        start_time = time.time()
        print(f"⏱️ Bắt đầu huấn luyện ({steps_per_epoch} ảnh/epoch, tổng {self.args.num_epochs} epochs). Giới hạn an toàn: {self.args.max_hours} tiếng.", flush=True)

        try:
            for epoch in range(self.start_epoch, self.args.num_epochs):
                if (time.time() - start_time) >= self.max_time_seconds:
                    print(f"⏰ ĐÃ ĐẠT GIỚI HẠN {self.args.max_hours} TIẾNG! Dừng an toàn tại Epoch {epoch}.", flush=True)
                    break

                self.train_fusion_dataset.update_schedule(epoch)
                epoch_loss = 0.0

                # GIAI ĐOẠN 1: 50 Epoch đầu chỉ train GR
                if epoch < 50:
                    self.gr_model.train()
                    for index, (input_b, gt_t, gt_r, has_gt_r) in enumerate(train_loader):
                        if not bool(has_gt_r[0]):
                            continue
                        input_b = input_b.to(self.device)
                        gt_r = gt_r.to(self.device)
                        self.GR_opt.zero_grad()
                        pred_R = self.gr_model(input_b)

                        r_rec = self.l1_loss(pred_R, gt_r)
                        r_feat = self.feat_loss(pred_R, gt_r)
                        loss_gr = r_rec + 0.2 * r_feat

                        loss_gr.backward()
                        self.GR_opt.step()
                        epoch_loss += loss_gr.item()

                        if (index + 1) % 100 == 0:
                            print(f"[Stage 1: Epoch {epoch+1}/50] Step [{index+1}/{len(train_loader)}] | GR_Loss: {loss_gr.item():.4f} | R_L1: {r_rec.item():.4f}", flush=True)

                    avg_epoch_loss = epoch_loss / max(len(train_loader), 1)
                    print(f"==> Kết thúc Epoch {epoch+1}/{self.args.num_epochs} (Stage 1) | Avg GR_Loss: {avg_epoch_loss:.4f}", flush=True)

                # GIAI ĐOẠN 2: 100 Epoch sau train kết hợp cả GR và GT
                else:
                    self.gt_model.train(); self.discriminator.train(); self.gr_model.train()
                    for index, (input_b, gt_t, gt_r, has_gt_r) in enumerate(train_loader):
                        input_b = input_b.to(self.device)
                        gt_t = gt_t.to(self.device)
                        valid_r = bool(has_gt_r[0])

                        if valid_r:
                            gt_r = gt_r.to(self.device)
                            gray_r = self.convert_L(gt_r)
                            map_thr, map_encoder = (gray_r < 0.01).float(), (gray_r > 0.30).float()
                        else:
                            map_thr = torch.zeros((input_b.size(0), 1, 256, 256), device=input_b.device)
                            map_encoder = torch.zeros((input_b.size(0), 1, 256, 256), device=input_b.device)

                        # Train Discriminator
                        if index % 2 == 0:
                            self.D_opt.zero_grad()
                            with torch.no_grad():
                                pred_R = self.gr_model(input_b)
                                output_t, *_ = self.gt_model(input_b, pred_R, map_thr, map_encoder)
                            d_loss = 0.5 * torch.mean(-(torch.log(self.discriminator(input_b, gt_t) + EPS) + torch.log(1.0 - self.discriminator(input_b, output_t) + EPS)))
                            d_loss.backward(); self.D_opt.step()

                        # Train Generator (GR + GT)
                        self.G_opt.zero_grad()
                        pred_R = self.gr_model(input_b)
                        output_t, list_map, list_R, list_mask, list_mask_enc, list_map_enc = self.gt_model(input_b, pred_R, map_thr, map_encoder)

                        t_rec = self.l1_loss(output_t, gt_t)
                        f_loss_t = self.feat_loss(output_t, gt_t)
                        freq_t = self.fft_loss(output_t, gt_t)
                        a_loss = torch.mean(-torch.log(self.discriminator(input_b, output_t) + EPS))
                        e_loss = self.exclusion_loss(output_t, pred_R)

                        if valid_r:
                            g_loss = (t_rec + self.l1_loss(pred_R, gt_r)) + 1.0 * (f_loss_t + self.feat_loss(pred_R, gt_r)) + 0.1 * freq_t + 0.2 * e_loss + self.thr_loss(list_map, list_R, list_mask, list_mask_enc, list_map_enc) + 0.01 * a_loss
                        else:
                            g_loss = t_rec + 1.0 * f_loss_t + 0.1 * freq_t + 0.2 * e_loss + 0.01 * a_loss

                        g_loss.backward(); self.G_opt.step()
                        epoch_loss += g_loss.item()

                        if (index + 1) % 100 == 0:
                            print(f"[Stage 2: Epoch {epoch+1}/{self.args.num_epochs}] Step [{index+1}/{len(train_loader)}] | G_Loss: {g_loss.item():.4f} | T_L1: {t_rec.item():.4f}", flush=True)

                    avg_epoch_loss = epoch_loss / max(len(train_loader), 1)
                    print(f"==> Kết thúc Epoch {epoch+1}/{self.args.num_epochs} (Stage 2) | Avg G_Loss: {avg_epoch_loss:.4f}", flush=True)

                # Cập nhật lịch sử loss
                self.metrics_history['epochs'].append(epoch + 1)
                self.metrics_history['train_loss'].append(avg_epoch_loss)

                # ===============================================================
                # LƯU CHECKPOINT TỐI ƯU BỘ NHỚ ĐĨA
                # ===============================================================
                save_data = {
                    'epoch': epoch + 1,
                    'GT_state': self.gt_model.state_dict(),
                    'GR_state': self.gr_model.state_dict(),
                    'D_state': self.discriminator.state_dict(),
                    'GR_opt': self.GR_opt.state_dict(),
                    'G_opt': self.G_opt.state_dict(),
                    'D_opt': self.D_opt.state_dict()
                }

                # Ghi đè file latest.pth sau mỗi epoch
                latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
                torch.save(save_data, latest_path)
                print(f"💾 [Latest] Đã cập nhật {latest_path} cho Epoch {epoch+1}", flush=True)

                # Lưu snapshot ở các mốc quan trọng (mỗi 10 epoch hoặc mốc 50, 100, 150)
                cur_ep = epoch + 1
                if cur_ep % 10 == 0 or cur_ep in [50, 100, 150]:
                    ckpt_path = os.path.join(self.checkpoint_dir, f'epoch_{cur_ep:03d}.pth')
                    torch.save(save_data, ckpt_path)
                    print(f"💾 [Snapshot] Đã lưu mốc quan trọng: {ckpt_path}", flush=True)

                # Dọn dẹp snapshot cũ để giải phóng dung lượng đĩa
                all_snaps = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'epoch_*.pth')))
                if len(all_snaps) > 3:
                    for old_f in all_snaps[:-3]:
                        if not any(k in old_f for k in ['050.pth', '100.pth', '150.pth']):
                            try:
                                os.remove(old_f)
                                print(f"🧹 Dọn bớt snapshot cũ để giải phóng đĩa: {old_f}", flush=True)
                            except:
                                pass

                # ===============================================================
                # ĐÁNH GIÁ CHU KỲ (MỖI 5 EPOCH HOẶC EPOCH CUỐI CÙNG)
                # ===============================================================
                if (cur_ep % self.eval_freq == 0) or (cur_ep == self.args.num_epochs):
                    self.run_eval_epoch(cur_ep)

        except KeyboardInterrupt:
            print("\n⚠️ Quá trình huấn luyện bị tạm dừng bởi người dùng (KeyboardInterrupt)!", flush=True)

        # ===============================================================
        # TỰ ĐỘNG CHỌN EPOCH TỐT NHẤT VÀ XUẤT TOÀN BỘ ẢNH RA THƯ MỤC result/
        # Chỉ thực hiện khi đã huấn luyện ít nhất 1 epoch hoặc đã có best.pth
        # ===============================================================
        best_ckpt_path = os.path.join(self.checkpoint_dir, 'best.pth')
        if self.best_epoch > 0 or os.path.isfile(best_ckpt_path):
            self.run_final_test_and_save_results()
        else:
            print("\nℹ️ Quá trình train dừng trước khi hoàn thành epoch nào, bỏ qua xuất ảnh kết quả cuối cùng.")
