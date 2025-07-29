from utils.utils import *
from utils.FRSGM.condrefinenet_fourier import CondRefineNetDilated

class ADDL_V3:
    def __init__(self,
                 max_iter=100,
                 level=2):

        print('=====================================')
        print('============== ADDL V3 ==============')
        print('=====================================')

        import main
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.T_ksfull = torch.from_numpy(main.ksfull).to(device)
        self.T_ksdata = torch.from_numpy(main.ksdata).to(device)
        self.T_mask = torch.from_numpy(main.mask).to(device)
        self.T_Ker = torch.from_numpy(main.Ker).to(device)
        self.T_Ker_Tra = torch.from_numpy(main.Ker_Tra).to(device)

        self.max_iter, self.level, self.save_path = max_iter, level, main.save_path

        self.scoreNet = CondRefineNetDilated(2, 2, 64).to(device)
        self.scoreNet.load_state_dict(torch.load('./utils/checkpoint/net.pth')['weights'])
        self.scoreNet.eval()
        self.gamma = 1.3
        self.lam = 0.006

        print('Estimating the Lip')
        self.Lip_C = (1 + main.Lip_C) ** 2
        self.PD3O_gamma = 1.999 / self.Lip_C
        self.PD3O_delta = 0.999 / self.PD3O_gamma

        self.Thr, self.T_AveKer = 0, torch.ones((3, 3), dtype=torch.float64, device=device) / 9
        self.start_time = time.time() - main.t_kernel - main.t_lip
        self.A = lambda x: VidHaarDec3S(IFFT2_3D_N((1 - self.T_mask) * x), self.level)
        self.At = lambda x: (1 - self.T_mask) * FFT2_3D_N(VidHaarRec3S(x, self.level))


    def process(self):
        y = self.T_ksdata.clone()
        c = VidHaarDec3S(IFFT2_3D_N(self.T_ksdata), self.level)
        s = torch.zeros_like(c)
        ref = sos(IFFT2_3D_N(self.T_ksfull))
        gamma_At_sk = 0

        for iter in range(self.max_iter):
            if iter == 0:
                print(f"At the begin, err = {torch.norm(torch.abs(self.T_ksfull - y)):.6f}")
            elif (iter + 1) % int(self.max_iter / 5) == 0:
                print(f"At iteration {iter + 1}, err = {torch.norm(torch.abs(self.T_ksfull - y)):.6f}")

            u = (1 - self.T_mask) * y + self.T_ksdata
            grad_f = Kernel_Rec_ks_C_I_Pro(u, self.T_Ker, 1)
            grad_f = Kernel_Rec_ks_C_I_Pro(grad_f, self.T_Ker_Tra, 1)
            gamma_grad_f = self.PD3O_gamma * grad_f
            gamma_At_sk_uk_gamma_grad_f = gamma_At_sk - (2 * u - y - gamma_grad_f)
            t = s - self.PD3O_delta * self.A(gamma_At_sk_uk_gamma_grad_f) + self.PD3O_delta * c
            s[..., [1, 2, 3, 4, 6, 7, 8, 9]] = t[..., [1, 2, 3, 4, 6, 7, 8, 9]] - self.SGM(iter, y, t)
            gamma_At_sk = self.PD3O_gamma * self.At(s)
            y = u - gamma_grad_f - gamma_At_sk


        res = sos(IFFT2_3D_N((1 - self.T_mask) * y + self.T_ksdata))
        useTime = time.time() - self.start_time
        print(f"ADDL-V3 Elapsed Time: {useTime:.2f} seconds")
        return PSNR_SSIM_HaarPSI(ref, res, 'ADDL-V3', self.save_path, useTime)

    def SGM(self, iter, uk, xk_delta_z, step=1):
        if iter in range(0, 50, 5):
            coil_image = DenoiseByDiffusion(self, IFFT2_3D_N((1 - self.T_mask) * uk + self.T_ksdata), step=step)
            y = torch.abs(VidHaarDec3S(coil_image))
            Thr = imfilter_symmetric_4D(y[..., [1, 2, 3, 4, 6, 7, 8, 9]], self.T_AveKer)
            self.Thr = EnergyScaling_4D(1 / Thr, xk_delta_z[..., [1, 2, 3, 4, 6, 7, 8, 9]])

        xk_delta_z[..., [1, 2, 3, 4, 6, 7, 8, 9]] = wthresh(xk_delta_z[..., [1, 2, 3, 4, 6, 7, 8, 9]], self.Thr)
        return xk_delta_z[..., [1, 2, 3, 4, 6, 7, 8, 9]]

