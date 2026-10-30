import os
import argparse
from solver import Solver

def get_default_data_root():
    candidates = [
        "/kaggle/input/datasets/congduc1122/pbl6-dataset500/dataset500/dataset500",
        "/kaggle/input/pbl6-dataset500/dataset500/dataset500",
        "/kaggle/input/datasets/congduc1122/pbl6-dataset500",
        "/kaggle/input/pbl6-dataset500",
        "../dataset500",
        "./dataset500",
        "dataset500",
        "/kaggle/input/datasets/congduc1122/pbl6-reflection-removal-duc/reflection-removal",
        "/kaggle/input/reflection-removal"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Quét tự động trong /kaggle/input
    if os.path.isdir("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            if "train" in dirs and ("real" in os.listdir(os.path.join(root, "train")) or "synthetic" in os.listdir(os.path.join(root, "train"))):
                return root
    return candidates[0]

def get_default_test_root():
    candidates = [
        "/kaggle/input/datasets/congduc1122/pbl6-dataset500/dataset500/dataset500/testsets",
        "/kaggle/input/pbl6-dataset500/dataset500/dataset500/testsets",
        "/kaggle/input/datasets/congduc1122/pbl6-dataset500/testsets",
        "/kaggle/input/pbl6-dataset500/testsets",
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
    # Quét tự động trong /kaggle/input
    if os.path.isdir("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            if "testsets" in dirs or ("real20" in dirs and "PostcardDataset" in dirs):
                return os.path.join(root, "testsets") if "testsets" in dirs else root
    return candidates[0]

def get_default_vgg_ckpt():
    candidates = [
        "/kaggle/input/datasets/congduc1122/pbl6-dataset500/vgg19-dcbb9e9d.pth",
        "/kaggle/input/pbl6-dataset500/vgg19-dcbb9e9d.pth",
        "./checkpoints/vgg19-dcbb9e9d.pth",
        "../checkpoints/vgg19-dcbb9e9d.pth",
        "./checkpoint/vgg19-dcbb9e9d.pth",
        "../checkpoint/vgg19-dcbb9e9d.pth",
        "/kaggle/input/datasets/congduc1122/pbl6-reflection-removal-duc/vgg19-dcbb9e9d.pth",
        "/kaggle/input/reflection-removal/vgg19-dcbb9e9d.pth"
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Quét tự động trong /kaggle/input
    if os.path.isdir("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if "vgg19" in f and f.endswith(".pth"):
                    return os.path.join(root, f)
    return candidates[0]

def get_default_save_dir():
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/result"
    return "./result"

def get_default_checkpoint_dir():
    if os.path.exists("/kaggle/working"):
        return "/kaggle/working/checkpoint"
    return "./checkpoint"

def main():
    parser = argparse.ArgumentParser('PBL6 Reflection Removal Training & Auto-Evaluation')
    parser.add_argument('--data_root', default=get_default_data_root(), help="Thư mục gốc dataset train (chứa real, synthetic)")
    parser.add_argument('--test_root', default=get_default_test_root(), help="Thư mục gốc testset (chứa real20, real45, SIR2...)")
    parser.add_argument('--vgg_ckpt', default=get_default_vgg_ckpt(), help='Đường dẫn VGG-19 pretrained checkpoint')
    parser.add_argument('--eval_freq', default=5, type=int, help="Kiểm thử PSNR/SSIM và vẽ lại đồ thị mỗi X epochs (mặc định: 5)")
    parser.add_argument('--save_model_freq', default=1, type=int, help="Lưu model mỗi X epochs")
    parser.add_argument('--checkpoint_dir', default=get_default_checkpoint_dir(), help="Thư mục lưu checkpoint")
    parser.add_argument('--save_dir', default=get_default_save_dir(), help="Thư mục lưu ảnh kết quả test và đồ thị")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=1, type=int, help="Batch size")
    parser.add_argument('--num_epochs', default=150, type=int, help="Tổng số epochs (mặc định: 150)")
    parser.add_argument('--steps_per_epoch', default=500, type=int, help="Số ảnh/bước train mỗi epoch (500 ảnh)")
    parser.add_argument('--max_hours', default=10.0, type=float, help="Thời gian tối đa trước khi tự dừng an toàn (10 tiếng)")
    parser.add_argument('--resume', default="", type=str, help="Đường dẫn file checkpoint để train tiếp (vd: ./checkpoint/latest.pth)")

    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    print("================================================================================")
    print("🚀 KHỞI ĐỘNG HUẤN LUYỆN PBL6 REFLECTION REMOVAL (150 EPOCHS)")
    print("================================================================================")
    print(f" • Data Root      : {args.data_root}")
    print(f" • Test Root      : {args.test_root}")
    print(f" • VGG Checkpoint : {args.vgg_ckpt}")
    print(f" • Chu kỳ test    : Mỗi {args.eval_freq} epochs tự động test PSNR/SSIM và vẽ lại đồ thị")
    print(f" • Checkpoint Dir : {args.checkpoint_dir}")
    print(f" • Result Dir     : {args.save_dir} (Xuất toàn bộ ảnh kết quả model tốt nhất)")
    print(f" • Tổng Epochs    : {args.num_epochs} (Giai đoạn 1: 1-50 GR | Giai đoạn 2: 51-150 GR+GT)")
    print("================================================================================")

    solver = Solver(args)
    solver.train_model()

if __name__ == '__main__':
    main()
