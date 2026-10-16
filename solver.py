import os
import torch
import torchvision.models as models
import torch.nn.functional as F
from models.GT import Discriminator, GT_Model
from models.GR import GR_Model
from models.encoder_build import encoder
from data.dataset import RealPairDataset, VocCurriculumSynDataset, CurriculumFusionDataset

EPS = 1e-12

class Solver:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.global_step = 0
        
        self.encoder_I, self.encoder_R, self.encoder = encoder().cuda(), encoder().cuda(), encoder().cuda()
        
        self.vgg_feature = models.vgg19()
        vgg_state = torch.load(self.args.vgg_ckpt)
        self.vgg_feature.load_state_dict(vgg_state)
        self.vgg_feature.cuda().eval()

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
        """Cải tiến: FFT Frequency Loss giữ chi tiết biên và tần số cao."""
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
            img_T, img_R = F.interpolate(img_T, scale_factor=0.5, mode='bilinear'), F.interpolate(img_R, scale_factor=0.5, mode='bilinear')
        return (torch.sum(sum(grad_x_loss)/3) + torch.sum(sum(grad_y_loss)/3)) / 2

    def l1_loss(self, input, output): return torch.mean(torch.abs(input - output))
    def convert_L(self, img): return 0.114 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.299 * img[:, 2:3, :, :]
    
    def thr_loss(self, list_map, list_R, list_mask, list_mask_encoder, list_map_encoder):
        mask_loss, mask_encoder_loss = 0, 0
        for i in range(4):
            mask_loss += self.l1_loss(list_map[i] * list_mask[i], list_map[i] * 1)
            mask_encoder_loss += self.l1_loss(list_map_encoder[i] * list_mask_encoder[i], list_map_encoder[i] * 0)
        return mask_loss + mask_encoder_loss

    def train_model(self):
        self.gr_model, self.gt_model, self.discriminator = GR_Model(self.encoder).cuda(), GT_Model(self.encoder_I, self.encoder_R).cuda(), Discriminator().cuda()
        self.G_opt = torch.optim.Adam(list(self.gt_model.parameters()) + list(self.gr_model.parameters()), lr=self.args.lr)
        self.D_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr)

        real_dirs = [os.path.join(self.args.data_root, "train", "real"), os.path.join(self.args.data_root, "train", "nature")]
        voc_dir = os.path.join(self.args.data_root, "train", "VOCdevkit", "VOC2012", "PNGImages")

        train_real_dataset = RealPairDataset(real_dirs, crop_size=256)
        train_syn_dataset = VocCurriculumSynDataset(voc_dir, crop_size=256)
        self.train_fusion_dataset = CurriculumFusionDataset(train_syn_dataset, train_real_dataset, steps_per_epoch=2000, syn_ratio=0.8)

        train_loader = torch.utils.data.DataLoader(self.train_fusion_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        for epoch in range(self.start_epoch, self.args.num_epochs):
            self.train_fusion_dataset.update_schedule(epoch)
            self.gt_model.train(); self.discriminator.train(); self.gr_model.train()
            
            for index, (input_b, gt_t, gt_r, has_gt_r) in enumerate(train_loader):
                input_b, gt_t = input_b.cuda(), gt_t.cuda()
                valid_r = bool(has_gt_r[0])

                if valid_r:
                    gt_r = gt_r.cuda()
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

                # Train Generator
                self.G_opt.zero_grad()
                pred_R = self.gr_model(input_b)
                output_t, list_map, list_R, list_mask, list_mask_enc, list_map_enc = self.gt_model(input_b, pred_R, map_thr, map_encoder)
                
                t_rec, f_loss_t, freq_t, a_loss, e_loss = self.l1_loss(output_t, gt_t), self.feat_loss(output_t, gt_t), self.fft_loss(output_t, gt_t), torch.mean(-torch.log(self.discriminator(input_b, output_t) + EPS)), self.exclusion_loss(output_t, pred_R)

                if valid_r:
                    g_loss = (t_rec + self.l1_loss(pred_R, gt_r)) + 0.2 * (f_loss_t + self.feat_loss(pred_R, gt_r)) + 0.1 * freq_t + e_loss + self.thr_loss(list_map, list_R, list_mask, list_mask_enc, list_map_enc) + 0.01 * a_loss
                else:
                    g_loss = t_rec + 0.2 * f_loss_t + 0.1 * freq_t + e_loss + 0.01 * a_loss

                g_loss.backward(); self.G_opt.step()

            if (epoch + 1) % self.args.save_model_freq == 0:
                torch.save({'epoch': epoch + 1, 'GT_state': self.gt_model.state_dict(), 'GR_state': self.gr_model.state_dict()}, f'./checkpoint/epoch_{epoch+1:03d}.pth')
                print(f"Đã lưu checkpoint tại Epoch {epoch+1}")