import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser('PBL6 Reflection Removal Training')
    parser.add_argument('--data_root', default=r"D:\PBL6\reflection-removal", help="Thư mục gốc dataset")
    parser.add_argument('--vgg_ckpt', default='./checkpoint/vgg19-dcbb9e9d.pth', help='Đường dẫn VGG-19 pretrained checkpoint')
    parser.add_argument('--save_model_freq', default=5, type=int, help="Lưu model mỗi X epochs")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=1, type=int, help="Batch size (Khuyên dùng 1 cho ảnh 256x256)")
    parser.add_argument('--num_epochs', default=150, type=int, help="Tổng số epochs")
    
    args = parser.parse_args()
    os.makedirs('./checkpoint', exist_ok=True)
    solver = Solver(args)
    solver.train_model()

if __name__ == '__main__':
    main()