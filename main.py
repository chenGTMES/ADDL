import csv

from utils.utils import *
from algorithm.ADDL import ADDL_V3


ksfull, ksdata, mask, save_path, sensitivity, Ker, Ker_Tra, Lip_C = None, None, None, None, None, None, None, None
sensitivityLi, t_sense, t_kernel, t_lip = None, None, None, None
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(filename, maskfilename):
    indexs = []
    save_root = load_data_and_mask(filename, maskfilename)

    indexs.append(ADDL_V3(max_iter=100).process())

    torch.cuda.empty_cache()

    df = pd.DataFrame(indexs, columns=["Model Name", "PSNR", "SSIM", "HaarPSI", "DISTS", "time"])
    df.to_csv(f'{save_root}/results_{filename}_{maskfilename}.csv', index=False)
    df["PSNR"], df["SSIM"], df["HaarPSI"], df["DISTS"], df["time"] = df["PSNR"].round(4), df["SSIM"].round(4), df["HaarPSI"].round(4), df["DISTS"].round(4), df["time"].round(4)
    print(df.to_string(index=False))

if __name__ == "__main__":
    

    filename = '05_t2_tse_tra_512_s33_3mm_29'
    maskfilename = 'mask_random_512_512_SR_30_AC_27'
    main(filename, maskfilename)
