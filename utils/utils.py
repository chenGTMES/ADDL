import os
import cv2
import datetime
import torch
import time
import math
import warnings
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from scipy.ndimage import rotate
from scipy.io import savemat
from scipy.io import loadmat
from scipy.ndimage import label
from scipy.ndimage import convolve
from scipy.ndimage import binary_dilation
from PIL import Image
from skimage.util import img_as_ubyte
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from utils.HaarPSI import HaarPSI
from utils.DISTS.DISTS import compute_DISTS
from utils.ESPIRiT.espirit import espirit

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

def load_data_and_mask(filename, maskfilename, Ker_Size=(7,7), normalizedCoeByEnergy=True):
    import main
    now = datetime.datetime.now()
    year_month = now.strftime('%Y_%m')
    year_month_day = now.strftime('%Y_%m_%d')
    dt = now.strftime('%Y_%m_%d_%H_%M_%S')
    dt = f"{dt}_{filename}"
    save_path = os.path.join('./result', year_month, year_month_day, dt)
    os.makedirs(save_path, exist_ok=True)

    data_file = f'./data/{filename}.mat'
    data = loadmat(data_file)
    ksfull = data['ksfull']

    row, col, coils = ksfull.shape

    tokens = maskfilename.split('_')
    random_or_uniform = tokens[1]

    size_str = f"{row}x{col}"
    maskfilepath = os.path.join('mask', size_str, random_or_uniform, f"{maskfilename}.png")

    mask = Image.open(maskfilepath)
    mask = np.array(mask).astype(np.int)
    mask = rotate(mask, angle=90, reshape=True)
    io.imsave(os.path.join(save_path, f"{maskfilename}.png"), img_as_ubyte(mask / np.max(mask)))

    mask = np.tile(mask[:, :, np.newaxis], (1, 1, coils))

    if normalizedCoeByEnergy:
        ksfull = NormalizedCoeByEnergy(ksfull)


    ref = sos(IFFT2_3D_N(ksfull))
    io.imsave(os.path.join(save_path, '1-reference.png'), img_as_ubyte(ref / np.max(ref)))

    ksdata = mask * ksfull
    un_image = sos(IFFT2_3D_N(ksdata))
    io.imsave(os.path.join(save_path, 'aliasing.png'), img_as_ubyte(un_image / np.max(un_image)))

    main.ksfull, main.ksdata, main.mask, main.save_path = ksfull, ksdata, mask, save_path

    start_time = time.time()
    sensitivity, sensitivityLi, ACS = get_sensitivity(ksdata, mask)
    # sensitivity, sensitivityLi, ACS = get_ESPIRiT_sensitivity(ksdata, mask)
    main.t_sense = time.time() - start_time
    start_time = time.time()
    Ker, Ker_Tra = Kernel_Estimation(ksdata, ACS, Ker_Size)
    main.t_kernel = time.time() - start_time
    start_time = time.time()
    Lip_C = Est_Lip_Ker_Mat_C_GPU(torch.from_numpy(ksdata).cuda(), torch.from_numpy(Ker).cuda(), Ker_Size)
    main.t_lip = time.time() - start_time
    main.sensitivity, main.sensitivityLi, main.Ker, main.Ker_Tra, main.Lip_C = sensitivity, sensitivityLi, Ker, Ker_Tra, Lip_C

    return save_path

def Est_Lip_Ker_Mat_C_GPU(ful_kspace, ker, ker_size=(5, 5)):
    def compute_Lip(ful_kspace, ker, ker_size):
        row, col, coi = ful_kspace.shape
        ker_matrix = torch.zeros((row, col), dtype=torch.float32, device=ful_kspace.device)
        ker_matrix_circ_fft = torch.zeros((row, col, coi, coi), dtype=torch.complex128, device=ful_kspace.device)
        for coi_num in range(coi):
            for i in range(coi):
                ker_matrix[:ker_size[0], :ker_size[1]] = ker[:, :, i, coi_num]
                ker_matrix_circ = torch.roll(ker_matrix, shifts=(-(ker_size[0] // 2), -(ker_size[1] // 2)), dims=(0, 1))
                ker_matrix_circ_fft[:, :, i, coi_num] = torch.fft.fft2(ker_matrix_circ)
        eig_vals = np.linalg.norm(ker_matrix_circ_fft.cpu().numpy().reshape(row * col, coi, coi), ord=2, axis=(1, 2))
        lip = eig_vals.max()
        return lip
    if ker.dim() == 4:
        return compute_Lip(ful_kspace, ker, ker_size)
    elif ker.dim() == 5:
        Lip = -1
        for i in range(ker.shape[-1]):
            Lip = max(Lip, compute_Lip(ful_kspace, ker[..., i], ker_size))
        return Lip
    else:
        return None

def Est_Lip_Ker_Mat_C(ful_kspace, ker, Ker_size=(5, 5)):
    row, col, coi = ful_kspace.shape
    ker_matrix = np.zeros((row, col), dtype=np.float32)
    ker_matrix_circ_fft = np.zeros((row, col, coi, coi), dtype=np.complex128)

    for coi_num in range(coi):
        for i in range(coi):
            ker_matrix[:Ker_size[0], :Ker_size[1]] = ker[:, :, i, coi_num]
            ker_matrix_circ = np.roll(ker_matrix, shift=(-(Ker_size[0] // 2), -(Ker_size[1] // 2)), axis=(0, 1))
            ker_matrix_circ_fft[:, :, i, coi_num] = np.fft.fft2(ker_matrix_circ)

    row, col, coi, _ = ker_matrix_circ_fft.shape
    eig_vals = np.linalg.norm(ker_matrix_circ_fft.reshape(row * col, coi, coi), ord=2, axis=(1, 2))
    lip = eig_vals.max()

    return lip

def Kernel_Estimation_Real_Imag(Full_kspace, ACS_Line, Ker_Size=[7, 7], Itr=500):
    center = Full_kspace.shape[0] // 2
    start = center - ACS_Line // 2
    end = center + ACS_Line // 2
    ACS_Data = Full_kspace[start:end, :, :]
    Mat_Data, Tag_Vec = Spirit_Kernel(Ker_Size, ACS_Data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Mat_Data = torch.tensor(Mat_Data, dtype=torch.complex128, device=device)
    Tag_Vec = torch.tensor(Tag_Vec, dtype=torch.complex128, device=device)

    Mat_Data_r = Mat_Data.real
    Mat_Data_i = Mat_Data.imag
    Tag_Vec_r = Tag_Vec.real
    Tag_Vec_i = Tag_Vec.imag
    w_r = np.zeros((Mat_Data.shape[1], Mat_Data.shape[2]), dtype=np.float64)
    w_i = np.zeros((Mat_Data.shape[1], Mat_Data.shape[2]), dtype=np.float64)

    for i in range(Mat_Data.shape[2]):
        w_r[:, i] = solve_linear_constrain(Mat_Data_r[:, :, i], Tag_Vec_r[:, i], Itr)
        w_i[:, i] = solve_linear_constrain(Mat_Data_i[:, :, i], Tag_Vec_i[:, i], Itr)

    Ker_out = None
    Ker_Tra_out = None

    for w in [w_r, w_i]:
        Con_w = np.sum(w ** 2, axis=0)
        Con_w_Max = np.max(Con_w)
        IndDel = int(np.floor((Ker_Size[0] - 1) / 2) * Ker_Size[1] + np.ceil(Ker_Size[1] / 2)) - 1

        # 初始化 Ker_w 和 Ker
        Ker_w = np.zeros((w.shape[0] + 1, w.shape[1]), dtype=np.float64)
        Ker = np.zeros((Ker_Size[0], Ker_Size[1], w.shape[1], w.shape[1]), dtype=np.float64)
        Ker_Tra = np.zeros((Ker_Size[0], Ker_Size[1], w.shape[1], w.shape[1]), dtype=np.float64)

        # 计算 Ker_Size 的积
        Ker_Size_prod = np.prod(Ker_Size)

        # 循环处理每一列
        for i in range(Ker_w.shape[1]):
            Target_loc = i * Ker_Size_prod + IndDel
            Ind_List = list(range(0, Target_loc)) + list(range(Target_loc + 1, Ker_Size_prod * Ker_w.shape[1]))

            # 更新 Ker_w
            Ker_w[Ind_List, i] = w[:, i]

            # 更新 Ker
            reshaped_Ker_w = Ker_w[:, i].reshape(Ker_w.shape[1], Ker_Size[0], Ker_Size[1])
            Ker[:, :, :, i] = np.transpose(reshaped_Ker_w, (2, 1, 0))
            RecFraOpe = transpose_filter(Ker[:, :, :, i])
            Ker_Tra[:, :, :, i] = RecFraOpe

        Trans_Ker = np.zeros_like(Ker_Tra)

        for coil in range(Ker_w.shape[1]):
            for i in range(Ker_w.shape[1]):
                Trans_Ker[:, :, i, coil] = Ker_Tra[:, :, coil, i]

        Ker_Tra = Trans_Ker
        KerW = np.sum(Ker ** 2, axis=(0, 1, 2))
        Ker_TraW = np.sum(Ker_Tra ** 2, axis=(0, 1, 2))

        Ker_W_Max = np.max(KerW * Ker_TraW)

        if Ker_W_Max > 1:
            Ker_Con = np.sqrt(Ker_W_Max + 1e-8)  # 使用 1e-8 作为 eps
            Ker = Ker / Ker_Con
            Ker_Tra = Ker_Tra / Ker_Con


        Ker_out = Ker if Ker_out is None else Ker_out + 1j * Ker
        Ker_Tra_out = Ker_Tra if Ker_Tra_out is None else Ker_Tra_out + 1j * Ker_Tra

    return Ker_out, Ker_Tra_out

def CCN_Kernel_Estimation_D2(Full_KSpace, ACS_Line, Ker_Size=[7, 7]):
    h, w, c = Full_KSpace.shape
    center_h = h // 2
    start_h = center_h - ACS_Line // 2
    end_h = center_h + ACS_Line // 2
    center_w = w // 2
    start_w = center_w - ACS_Line // 2
    end_w = center_w + ACS_Line // 2

    Ker = np.zeros((Ker_Size[0], Ker_Size[1], c, c, 2))
    Ker_Tra = np.zeros((Ker_Size[0], Ker_Size[1], c, c, 2))

    Data1 = Full_KSpace[start_h:end_h, start_w:end_w, :]
    Mat_Data, Tag_Vec = Spirit_Kernel(Ker_Size, Data1)
    Ker[..., 0], Ker_Tra[..., 0] = SPIRiT_Kernel_Estimation(Mat_Data, Tag_Vec, Ker_Size)

    Data2 = Full_KSpace[start_h:end_h, :start_w, :]
    Mat_Data1, Tag_Vec1 = Spirit_Kernel(Ker_Size, Data2)
    Data2 = Full_KSpace[center_h:end_h, end_w:, :]
    Mat_Data2, Tag_Vec2 = Spirit_Kernel(Ker_Size, Data2)
    Mat_Data = np.concatenate((Mat_Data1, Mat_Data2), axis=0)
    Tag_Vec = np.concatenate((Tag_Vec1, Tag_Vec2), axis=0)
    Ker[..., 1], Ker_Tra[..., 1] = SPIRiT_Kernel_Estimation(Mat_Data, Tag_Vec, Ker_Size)

    return Ker, Ker_Tra

def CCN_Kernel_Estimation_D3(Full_KSpace, ACS_Line, Ker_Size=[7, 7]):
    h, w, c = Full_KSpace.shape
    center_h = h // 2
    start_h = center_h - ACS_Line // 2
    end_h = center_h + ACS_Line // 2
    center_w = w // 2
    start_w = center_w - ACS_Line // 2
    end_w = center_w + ACS_Line // 2

    Ker = np.zeros((Ker_Size[0], Ker_Size[1], c, c, 3))
    Ker_Tra = np.zeros((Ker_Size[0], Ker_Size[1], c, c, 3))

    Data1 = Full_KSpace[start_h:end_h, start_w:end_w, :]
    Mat_Data, Tag_Vec = Spirit_Kernel(Ker_Size, Data1)
    Ker[..., 0], Ker_Tra[..., 0] = SPIRiT_Kernel_Estimation(Mat_Data, Tag_Vec, Ker_Size)

    Data2 = Full_KSpace[start_h:center_h, 0:start_w, :]
    Mat_Data1, Tag_Vec1 = Spirit_Kernel(Ker_Size, Data2)
    Data2 = Full_KSpace[center_h:end_h, end_w:w, :]
    Mat_Data2, Tag_Vec2 = Spirit_Kernel(Ker_Size, Data2)
    Mat_Data = np.concatenate((Mat_Data1, Mat_Data2), axis=0)
    Tag_Vec = np.concatenate((Tag_Vec1, Tag_Vec2), axis=0)
    Ker[..., 1], Ker_Tra[..., 1] = SPIRiT_Kernel_Estimation(Mat_Data, Tag_Vec, Ker_Size)

    Data3 = Full_KSpace[start_h:center_h, end_w:w, :]
    Mat_Data1, Tag_Vec1 = Spirit_Kernel(Ker_Size, Data3)
    Data3 = Full_KSpace[center_h:end_h, 0:start_w, :]
    Mat_Data2, Tag_Vec2 = Spirit_Kernel(Ker_Size, Data3)
    Mat_Data = np.concatenate((Mat_Data1, Mat_Data2), axis=0)
    Tag_Vec = np.concatenate((Tag_Vec1, Tag_Vec2), axis=0)
    Ker[..., 2], Ker_Tra[..., 2] = SPIRiT_Kernel_Estimation(Mat_Data, Tag_Vec, Ker_Size)

    return Ker, Ker_Tra

def Kernel_Estimation(Full_kspace, ACS_Line, Ker_Size=[7, 7]):
    center = Full_kspace.shape[0] // 2
    start = center - ACS_Line // 2
    end = center + ACS_Line // 2
    ACS_Data = Full_kspace[start:end, :, :]
    Mat_Data, Tag_Vec = Spirit_Kernel(Ker_Size, ACS_Data)
    return SPIRiT_Kernel_Estimation(Mat_Data, Tag_Vec, Ker_Size)

def SPIRiT_Kernel_Estimation(Mat_Data, Tag_Vec, Ker_Size, Itr=500):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Mat_Data = torch.tensor(Mat_Data, dtype=torch.complex128, device=device)
    Tag_Vec = torch.tensor(Tag_Vec, dtype=torch.complex128, device=device)

    Mat_Data_r = Mat_Data.real
    Mat_Data_i = Mat_Data.imag
    Tag_Vec_r = Tag_Vec.real
    Tag_Vec_i = Tag_Vec.imag
    w = np.zeros((Mat_Data.shape[1], Mat_Data.shape[2]), dtype=np.float64)

    for i in range(Mat_Data.shape[2]):
        # 构造 M 和 v
        M = torch.cat((Mat_Data_r[:, :, i], Mat_Data_i[:, :, i]), dim=0)
        v = torch.cat((Tag_Vec_r[:, i], Tag_Vec_i[:, i]), dim=0)
        w[:, i] = solve_linear_constrain(M, v, Itr)

    Con_w = np.sum(w ** 2, axis=0)
    Con_w_Max = np.max(Con_w)
    IndDel = int(np.floor((Ker_Size[0] - 1) / 2) * Ker_Size[1] + np.ceil(Ker_Size[1] / 2)) - 1

    # 初始化 Ker_w 和 Ker
    Ker_w = np.zeros((w.shape[0] + 1, w.shape[1]), dtype=np.float64)
    Ker = np.zeros((Ker_Size[0], Ker_Size[1], w.shape[1], w.shape[1]), dtype=np.float64)
    Ker_Tra = np.zeros((Ker_Size[0], Ker_Size[1], w.shape[1], w.shape[1]), dtype=np.float64)

    # 计算 Ker_Size 的积
    Ker_Size_prod = np.prod(Ker_Size)

    # 循环处理每一列
    for i in range(Ker_w.shape[1]):
        Target_loc = i * Ker_Size_prod + IndDel
        Ind_List = list(range(0, Target_loc)) + list(range(Target_loc + 1, Ker_Size_prod * Ker_w.shape[1]))

        # 更新 Ker_w
        Ker_w[Ind_List, i] = w[:, i]

        # 更新 Ker
        reshaped_Ker_w = Ker_w[:, i].reshape(Ker_w.shape[1], Ker_Size[0], Ker_Size[1])
        Ker[:, :, :, i] = np.transpose(reshaped_Ker_w, (2, 1, 0))
        RecFraOpe = transpose_filter(Ker[:, :, :, i])
        Ker_Tra[:, :, :, i] = RecFraOpe

    Trans_Ker = np.zeros_like(Ker_Tra)

    for coil in range(Ker_w.shape[1]):
        for i in range(Ker_w.shape[1]):
            Trans_Ker[:, :, i, coil] = Ker_Tra[:, :, coil, i]

    Ker_Tra = Trans_Ker
    KerW = np.sum(Ker ** 2, axis=(0, 1, 2))
    Ker_TraW = np.sum(Ker_Tra ** 2, axis=(0, 1, 2))

    Ker_W_Max = np.max(KerW * Ker_TraW)

    if Ker_W_Max > 1:
        Ker_Con = np.sqrt(Ker_W_Max + 1e-8)  # 使用 1e-8 作为 eps
        Ker = Ker / Ker_Con
        Ker_Tra = Ker_Tra / Ker_Con

    return Ker, Ker_Tra

def transpose_filter(FilOpe):
    RecFraOpe = np.zeros_like(FilOpe)
    _, _, DimFilOpe = FilOpe.shape
    for i in range(DimFilOpe):
        RecFraOpe[:, :, i] = np.flip(np.flip(FilOpe[:, :, i], axis=0), axis=1)

    return RecFraOpe

def solve_linear_constrain(A, b, Itr=500):
    # 初始化
    xk = torch.zeros((A.size(1), 1), dtype=A.dtype, device=A.device)
    y = torch.zeros((A.size(1), 1), dtype=A.dtype, device=A.device)
    AtA = torch.mm(A.T, A)
    Atb = torch.mm(A.T, b.reshape(-1, 1))
    tk = 1

    # 计算 Lipschitz 常数 L
    L = torch.max(torch.sum(torch.abs(AtA), dim=1))

    for _ in range(Itr):
        xk1 = y - (torch.mm(AtA, y) - Atb) / L
        x_Square = torch.norm(xk1, p=2)
        if x_Square > 0.99:
            xk1 = xk1 / (x_Square * 1.0102)

        tk1 = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        Acc_Wei = (tk - 1) / tk1
        tk = tk1

        y = xk1 + Acc_Wei * (xk1 - xk)
        xk = xk1

    return xk.flatten().cpu().numpy()

def Spirit_Kernel(Block, Acs):
    y, x, z = Acs.shape
    if Block[0] > y or Block[1] > x:
        raise ValueError("ACS area is too small!")
    elif Block[0] <= 1 or Block[1] <= 1:
        raise ValueError("Block error!")
    Cal_mat_row_length = (y - Block[0] + 1) * (x - Block[1] + 1)
    Cal_mat_col_length = Block[0] * Block[1]

    Cal_mat = np.zeros((Cal_mat_row_length, Cal_mat_col_length, z), dtype=Acs.dtype)
    count = 0
    for ii in range(Block[1]):  # x direction
        Block_x_head = ii
        Block_x_end = x - Block[1] + ii
        for jj in range(Block[0]):  # y direction
            Block_y_head = jj
            Block_y_end = y - Block[0] + jj
            # 从 Acs 中提取块并进行转置
            acs_block = Acs[Block_y_head:Block_y_end + 1, Block_x_head:Block_x_end + 1, :].transpose(1, 0, 2)
            Cal_mat[:, count, :] = acs_block.reshape(Cal_mat_row_length, z)
            count += 1
    acs_block = Acs[Block[0] // 2:y - (Block[0] - 1) // 2, Block[1] // 2:x - (Block[1] - 1) // 2, :].transpose(1, 0, 2)
    Target_vec = acs_block.reshape(Cal_mat_row_length, z)
    Temp_mat = Cal_mat.transpose(0, 2, 1).reshape(Cal_mat_row_length, Cal_mat_col_length * z)
    IndDel = (Block[0] - 1) // 2 * Block[1] + Block[1] // 2
    Cal_mat = np.zeros((Temp_mat.shape[0], Temp_mat.shape[1] - 1, z), dtype=Temp_mat.dtype)

    for i in range(z):
        Target_loc = i * Block[0] * Block[1] + IndDel
        list_ = list(range(Target_loc)) + list(range(Target_loc + 1, Block[0] * Block[1] * z))
        Cal_mat[:, :, i] = Temp_mat[:, list_]

    return Cal_mat, Target_vec

def get_sensitivity(ksfull, mask):
    x, y, z = ksfull.shape
    calib, AC_region = getCalibSize_1D_Edt(mask)
    ACS_line = np.abs(AC_region[1] - AC_region[0]) + 1
    Sense_Est_Mask = np.zeros((x, y), dtype=np.float32)
    Sense_Est_Mask[AC_region[0]:AC_region[1] + 1, :] = 1
    Sense_Est_Mask_3D = np.expand_dims(Sense_Est_Mask, axis=2)
    Sense_Est_Mask_3D = np.tile(Sense_Est_Mask_3D, (1, 1, z))
    kscenter = ksfull * Sense_Est_Mask_3D
    sence = Sensitivity_Compute(kscenter)
    # sence = Sensitivity_Compute(ksfull)
    return sence, sence, ACS_line

def get_sensitivity_MAINBODY(ksfull, mask):
    ali = sos(IFFT2_3D_N(ksfull))
    MAINBODY = FINDMAINBODY(ali)
    MAINBODY = convolve(MAINBODY, np.ones((25, 25)) / 81, mode='nearest')
    MAINBODY = (MAINBODY > 0).astype(int)

    sensitivity, ACS_line = get_sensitivity(ksfull, mask)

    x, y, z = ksfull.shape
    sense = np.zeros_like(ksfull)
    for i in range(z):
        sense[..., i] = pmri_poly_sensitivity(sensitivity[..., i], MAINBODY)

    temp = (np.sum(np.sqrt(np.abs(sense) ** 2), axis=2) + np.finfo(float).eps)
    sense /= np.expand_dims(temp, axis=-1)
    return sense, ACS_line

def pmri_poly_sensitivity(img, mask, order=2):
    valid_idx = np.where(mask.ravel())[0]
    yy, xx = np.meshgrid(np.arange(1, img.shape[1] + 1), np.arange(1, img.shape[0] + 1))
    x_all_idx = xx.ravel()  # 展平并转置
    y_all_idx = yy.ravel()  # 展平并转置
    rimg = img.T.ravel()  # 展平图像
    x_idx = x_all_idx[valid_idx]
    y_idx = y_all_idx[valid_idx]
    rimg = rimg[valid_idx]
    order_list = np.arange(order + 1)

    col = 0
    A = np.zeros((len(x_idx), len(order_list) * (len(order_list) + 1) // 2))
    A_all = np.zeros((len(x_all_idx), len(order_list) * (len(order_list) + 1) // 2))

    for i in range(len(order_list)):
        for j in range(len(order_list) - i):
            x_order = order_list[i]
            y_order = order_list[j]
            A[:, col] = (x_idx**x_order) * (y_idx**y_order)
            A_all[:, col] = (x_all_idx**x_order) * (y_all_idx**y_order)
            col += 1

    coeffs = np.linalg.pinv(A.T @ A) @ A.T @ rimg
    fit_all = A_all @ coeffs
    output = fit_all.reshape(img.shape)
    output = output * np.exp(np.sqrt(-1 + 0j) * (np.angle(img) * mask))

    return output

def get_ESPIRiT_sensitivity(ksfull, mask):
    sense, sense, ACS_line = get_sensitivity(ksfull, mask)
    ksdata = np.zeros_like(ksfull, dtype=np.complex)
    for i in range(ksfull.shape[2]):
        ksdata[:, :, i] = np.fft.fft2(np.fft.ifftshift(np.fft.ifft2(ksfull[:, :, i])))
    ksdata = np.expand_dims(ksdata, axis=0)
    esp = np.squeeze(espirit(ksdata, 6, min(ACS_line, 30), 0.01, 0.9925))
    sensitivity = esp[:, :, :, 0]
    return sensitivity, sense, ACS_line

def Sensitivity_Compute(kscenter):
    Img_zero_fill = IFFT2_3D_N(kscenter)
    # for i in range(Img_zero_fill.shape[-1]):
    #     ang = np.angle(Img_zero_fill[..., i])
    #     temp_real = (Img_zero_fill[..., i]).real
    #     temp_real[temp_real > (np.mean(temp_real) * 0.8)] = np.mean(temp_real) * 0.8
    #     temp_imag = (Img_zero_fill[..., i]).imag
    #     temp_imag[temp_imag > (np.mean(temp_imag) * 0.8)] = np.mean(temp_imag) * 0.8
    #     Img_zero_fill[..., i] = temp_real + 1j * temp_imag
    Img_clip_sos = sos(Img_zero_fill) + np.finfo(np.float32).eps
    Sense = Img_zero_fill / np.expand_dims(Img_clip_sos, axis=-1)

    # winSize = 5
    # aveKer = np.ones((winSize, winSize)) / (winSize * winSize)
    # for i in range(Img_zero_fill.shape[-1]):
    #     Img_zero_fill[..., i] = convolve(Img_zero_fill[..., i], aveKer, mode='nearest')
    #     # Img_zero_fill[..., i] = convolve(Img_zero_fill[..., i], aveKer, mode='nearest')
    #
    # Img_clip_sos = sos(Img_zero_fill) + np.finfo(np.float32).eps
    # Sense2 = Img_zero_fill / np.expand_dims(Img_clip_sos, axis=-1)

    return Sense

def getCalibSize_1D_Edt(mask):
    mask = mask[:, :, 0]
    mask_row, mask_col = mask.shape
    mask_row_center = mask_row // 2
    mask_col_center = mask_col // 2
    sx = 1
    sy = 1
    row_pos = mask_row_center - 1
    while row_pos >= 0:
        if mask[row_pos, mask_col_center] == 0:
            break
        else:
            sx += 1
        row_pos -= 1
    ACS_row_up_edge = row_pos + 1
    row_pos = mask_row_center + 1
    while row_pos < mask_row:
        if mask[row_pos, mask_col_center] == 0:
            break
        else:
            sx += 1
        row_pos += 1
    ACS_row_down_edge = row_pos - 1
    col_pos = mask_col_center - 1
    while col_pos >= 0:
        if mask[mask_row_center, col_pos] == 0:
            break
        else:
            sy += 1
        col_pos -= 1
    col_pos = mask_col_center + 1
    while col_pos < mask_col:
        if mask[mask_row_center, col_pos] == 0:
            break
        else:
            sy += 1
        col_pos += 1
    calib_size = [sx, sy]
    ACS_edge = [ACS_row_up_edge, ACS_row_down_edge]
    return calib_size, ACS_edge

def NormalizedCoeByEnergy(Coe):
    EngCoe = np.abs(Coe) ** 2
    SumEng = np.sum(np.sum(EngCoe, axis=1), axis=0)
    MaxEng = Coe.shape[0] * Coe.shape[1]
    WeiEng = MaxEng / SumEng
    NorCoe = np.empty_like(Coe)
    NorCoe = Coe * np.sqrt(WeiEng)[np.newaxis, np.newaxis, :]
    return NorCoe

def sos(x, pnorm=2):
    if isinstance(x, np.ndarray):
        sum_norm = np.sum(np.abs(x) ** pnorm, axis=-1)
        res = np.power(sum_norm, 1 / pnorm)
    if isinstance(x, torch.Tensor):
        sum_norm = torch.sum(torch.abs(x) ** pnorm, dim=-1)
        res = torch.pow(sum_norm, 1 / pnorm)
    return res

def sos_T(x, sense_map):
    return sense_map * x.unsqueeze(2).repeat(1, 1, sense_map.size(-1))

def sos_complex(x, sense_map):
    sense_map_conj = sense_map.real - 1j * sense_map.imag
    return (x * sense_map_conj).sum(dim=-1)

def sos_complex_T(x, sense_map):
    x_cpy = x.unsqueeze(2).repeat(1, 1, sense_map.size(-1))
    re = x_cpy.real * sense_map.real - x_cpy.imag * sense_map.imag
    im = x_cpy.real * sense_map.imag + x_cpy.imag * sense_map.real
    return re + 1j * im

def FFT2_3D_N(Img):
    if isinstance(Img, np.ndarray):
        fcoe = np.zeros_like(Img)
        for i in range(Img.shape[2]):
            fcoe[:, :, i] = np.fft.fftshift(np.fft.fft2(Img[:, :, i]))
    if isinstance(Img, torch.Tensor):
        fcoe = torch.zeros_like(Img)
        for i in range(Img.shape[2]):
            fcoe[:, :, i] = torch.fft.fftshift(torch.fft.fft2(Img[:, :, i]))

    fcoe /= np.sqrt(Img.shape[0] * Img.shape[1])

    return fcoe

def IFFT2_3D_N(Fcoe):
    if isinstance(Fcoe, np.ndarray):
        Img = np.zeros_like(Fcoe, dtype=np.complex)
        for i in range(Fcoe.shape[2]):
            Img[:, :, i] = np.fft.ifft2(np.fft.ifftshift(Fcoe[:, :, i]))
    if isinstance(Fcoe, torch.Tensor):
        Img = torch.zeros_like(Fcoe, dtype=torch.complex128)
        for i in range(Fcoe.shape[2]):
            Img[:, :, i] = torch.fft.ifft2(torch.fft.ifftshift(Fcoe[:, :, i]))

    Img *= np.sqrt(Fcoe.shape[0] * Fcoe.shape[1])

    return Img

def double2uint8(im, save_path):
    ref_uint8 = np.abs(im)
    ref_uint8 = ref_uint8 / np.max(ref_uint8)
    ref_uint8 = img_as_ubyte(ref_uint8)
    return ref_uint8

def DCT7Dec2(img, lev=1):
    img = img / 4
    img1_1 = torch.roll(img, shifts=lev, dims=0)
    img1_1 = torch.roll(img1_1, shifts=lev, dims=1)
    img0_1 = torch.roll(img, shifts=lev, dims=1)
    img1_0 = torch.roll(img, shifts=lev, dims=0)

    h, w = img.shape
    DecImg = torch.zeros((h, w, 5), dtype=img.dtype, device=img.device)
    DecImg[:, :, 0] = img + img1_1 + img0_1 + img1_0
    DecImg[:, :, 1] = img1_1 - img
    DecImg[:, :, 2] = img0_1 - img1_0
    DecImg[:, :, 3] = img1_1 - img1_0
    DecImg[:, :, 4] = img1_1 - img0_1
    return DecImg

def DCT7Rec2(DecImg, Filter):
    img1 = imfilter_circular(DecImg[:, :, 0], Filter[:, :, 0])
    img2 = imfilter_circular(DecImg[:, :, 1], Filter[:, :, 1])
    img3 = imfilter_circular(DecImg[:, :, 2], Filter[:, :, 2])
    img4 = imfilter_circular(DecImg[:, :, 3], Filter[:, :, 3] * 2)
    img5 = imfilter_circular(DecImg[:, :, 4], Filter[:, :, 4] * 2)
    img = img1 + img2 + img3 + img4 + img5
    return img

def ImgDecFram(img, Frame):
    h, w = img.shape
    DimFra = Frame.size(-1)
    ResImg = torch.zeros((h, w, DimFra), dtype=img.dtype, device=img.device)
    for i in range(DimFra):
        ResImg[:, :, i] = imfilter_circular(img, Frame[:, :, i])
    return ResImg

def ImgRecFram(Data, RecFrame):
    h, w, _ = Data.shape
    DimFra = RecFrame.size(-1)
    RecData = torch.zeros((h, w), dtype=Data.dtype, device=Data.device)
    for i in range(DimFra):
        RecData += imfilter_circular(Data[:, :, i], RecFrame[:, :, i])
    return RecData

def imfilter_circular(img, frame):
    # 确定 padding 以实现循环边界效果
    h, w = img.shape
    pad_h = frame.size(0) // 2
    pad_w = frame.size(1) // 2


    # 创建一个新的张量 img_pad，用于存放扩展后的结果
    img_pad = torch.cat((img[-pad_h:, :], img), dim=0)
    img_pad = torch.cat((img_pad, img[:pad_h, :]), dim=0)

    temp = img_pad[:, :pad_w]
    img_pad = torch.cat((img_pad[:, -pad_w:], img_pad), dim=1)
    img_pad = torch.cat((img_pad, temp), dim=1)

    img = img_pad

    # 处理输入，确保 img 和 frame 是 PyTorch 张量
    img = img.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
    frame = frame.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度


    # 使用 PyTorch 的卷积操作
    if torch.is_complex(img):
        conv_real = F.conv2d(img.real.to(dtype=torch.float64), frame.to(dtype=torch.float64), padding=0)
        conv_imag = F.conv2d(img.imag.to(dtype=torch.float64), frame.to(dtype=torch.float64), padding=0)
        filtered = torch.complex(conv_real, conv_imag)
    else:
        filtered = F.conv2d(img.to(dtype=torch.float64), frame.to(dtype=torch.float64), padding=0)

    return filtered.squeeze(0).squeeze(0)  # 移除 batch 和 channel 维度

def imfilter_symmetric(img, frame):
    # 确定 padding 以实现循环边界效果
    pad_h = frame.size(0) // 2
    pad_w = frame.size(1) // 2

    # 创建一个新的张量 img_pad，用于存放扩展后的结果
    temp = img[:pad_h, :]
    img_pad = torch.cat((torch.flip(temp, dims=[0]), img), dim=0)
    temp = img[-pad_h:, :]
    img_pad = torch.cat((img_pad, torch.flip(temp, dims=[0])), dim=0)
    temp = img_pad[:, :pad_w]
    img_pad = torch.cat((torch.flip(temp, dims=[1]), img_pad), dim=1)
    temp = img_pad[:, -pad_w:]
    img_pad = torch.cat((img_pad, torch.flip(temp, dims=[1])), dim=1)

    img = img_pad

    # 处理输入，确保 img 和 frame 是 PyTorch 张量
    img = img.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
    frame = frame.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度


    # 使用 PyTorch 的卷积操作
    if torch.is_complex(img):
        conv_real = F.conv2d(img.real.to(dtype=torch.float64), frame.to(dtype=torch.float64), padding=0)
        conv_imag = F.conv2d(img.imag.to(dtype=torch.float64), frame.to(dtype=torch.float64), padding=0)
        filtered = torch.complex(conv_real, conv_imag)
    else:
        filtered = F.conv2d(img.to(dtype=torch.float64), frame.to(dtype=torch.float64), padding=0)

    return filtered.squeeze(0).squeeze(0)  # 移除 batch 和 channel 维度

def imfilter_symmetric_3D(img, frame):
    _, _, subs = img.shape
    Sigma = torch.zeros_like(img)
    for sub in range(subs):
        Sigma[:, :, sub] = imfilter_symmetric(img[:, :, sub], frame)
    return Sigma

def imfilter_symmetric_4D(img, frame):
    _, _, coils, subs = img.shape
    Sigma = torch.zeros_like(img)
    for coil in range(coils):
        for sub in range(subs):
            Sigma[:, :, coil, sub] = imfilter_symmetric(img[:, :, coil, sub], frame)
    return Sigma

def wthresh(coef, thresh):
    if torch.is_complex(coef):
        res = coef * torch.maximum((torch.abs(coef) - thresh), torch.tensor(0.0)) / torch.abs(coef)
    else:
        res = torch.sgn(coef) * torch.maximum(torch.abs(coef) - thresh, torch.tensor(0.0))
    return res

def VidHaarDec3S(Vid, Lev=2):
    if Lev == 1:
        DecVid  = HaarDec3S(Vid, 1)
    elif Lev == 2:
        DecVid2 = HaarDec3S(Vid, 1)
        DecVid  = HaarDec3S(DecVid2[:,:,:, 0], 2)
        DecVid  = torch.cat((DecVid, DecVid2[:, :, :, 1:]), dim=3)
    else:
        raise ValueError('Lev is too big')

    return DecVid

def HaarDec3S(Vid, Lev=2):
    Vid0 = Vid.clone()  # Copy of Vid

    # Circular shifts using torch.roll
    shift_amount = 2 ** (Lev - 1)
    Vid1 = torch.roll(Vid0, shifts=[0, 0, shift_amount], dims=[0, 1, 2])  # 0 0 1
    Vid1m = torch.roll(Vid0, shifts=[0, 0, -shift_amount], dims=[0, 1, 2])  # 0 0 -1

    Vid2 = torch.roll(Vid0, shifts=[0, shift_amount, 0], dims=[0, 1, 2])  # 0 1 0

    Vid3 = torch.roll(Vid0, shifts=[0, shift_amount, shift_amount], dims=[0, 1, 2])  # 0 1 1
    Vid3m = torch.roll(Vid0, shifts=[0, -shift_amount, -shift_amount], dims=[0, 1, 2])  # 0 -1 -1

    Vid4 = torch.roll(Vid0, shifts=[shift_amount, 0, 0], dims=[0, 1, 2])  # 1 0 0

    Vid5 = torch.roll(Vid0, shifts=[shift_amount, 0, shift_amount], dims=[0, 1, 2])  # 1 0 1
    Vid5m = torch.roll(Vid0, shifts=[-shift_amount, 0, -shift_amount], dims=[0, 1, 2])  # -1 0 -1

    Vid6 = torch.roll(Vid0, shifts=[shift_amount, shift_amount, 0], dims=[0, 1, 2])  # 1 1 0

    Vid7 = torch.roll(Vid0, shifts=[shift_amount, shift_amount, shift_amount], dims=[0, 1, 2])  # 1 1 1
    Vid7m = torch.roll(Vid0, shifts=[-shift_amount, -shift_amount, -shift_amount], dims=[0, 1, 2])  # -1 -1 -1

    Vidp1p1m1 = torch.roll(Vid0, shifts=[shift_amount, shift_amount, -shift_amount], dims=[0, 1, 2])  # 1 1 -1
    Vidm1m1p1 = torch.roll(Vid0, shifts=[-shift_amount, -shift_amount, shift_amount], dims=[0, 1, 2])  # -1 -1 1

    Vidp1m1p1 = torch.roll(Vid0, shifts=[shift_amount, -shift_amount, shift_amount], dims=[0, 1, 2])  # 1 -1 1
    Vidm1p1m1 = torch.roll(Vid0, shifts=[-shift_amount, shift_amount, -shift_amount], dims=[0, 1, 2])  # -1 1 -1

    Vidm1p1p1 = torch.roll(Vid0, shifts=[-shift_amount, shift_amount, shift_amount], dims=[0, 1, 2])  # -1 1 1
    Vidp1m1m1 = torch.roll(Vid0, shifts=[shift_amount, -shift_amount, -shift_amount], dims=[0, 1, 2])  # 1 -1 -1

    Vidp10m1 = torch.roll(Vid0, shifts=[shift_amount, 0, -shift_amount], dims=[0, 1, 2])  # 1 0 -1
    Vidm10p1 = torch.roll(Vid0, shifts=[-shift_amount, 0, shift_amount], dims=[0, 1, 2])  # -1 0 1

    Vid0p1m1 = torch.roll(Vid0, shifts=[0, shift_amount, -shift_amount], dims=[0, 1, 2])  # 0 1 -1
    Vid0m1p1 = torch.roll(Vid0, shifts=[0, -shift_amount, shift_amount], dims=[0, 1, 2])  # 0 -1 1


    DecVid = torch.zeros([Vid.size(0), Vid.size(1), Vid.size(2), 6], dtype=Vid.dtype, device=Vid.device)


    # Compute decomposition
    DecVid[:, :, :, 0] = (Vid0 + Vid1 + Vid2 + Vid3 + Vid4 + Vid5 + Vid6 + Vid7) / 8  # low-pass filter

    DecVid[:, :, :, 1] = (Vid4 - Vid0) / 4  # (1,0,0) - (0,0,0) x-axis
    DecVid[:, :, :, 2] = (Vid2 - Vid0) / 4  # (0,1,0) - (0,0,0) y-axis

    DecVid[:, :, :, 3] = np.sqrt(2) / 8 * (Vid6 - Vid0)  # (1,1,0) - (0,0,0) xy-plane 1
    DecVid[:, :, :, 4] = np.sqrt(2) / 8 * (Vid4 - Vid2)  # (1,0,0) - (0,1,0) xy-plane 2

    DecVid[:, :, :, 5] = (1 / 2 * Vid0
                          - 1 / 16 * (Vid1 + Vid1m)
                          - 1 / 32 * (Vid3 + Vid3m)
                          - 1 / 32 * (Vid5 + Vid5m)
                          - 1 / 64 * (Vid7 + Vid7m)
                          - 1 / 64 * (Vidp1p1m1 + Vidm1m1p1)
                          - 1 / 64 * (Vidp1m1p1 + Vidm1p1m1)
                          - 1 / 64 * (Vidm1p1p1 + Vidp1m1m1)
                          - 1 / 32 * (Vidp10m1 + Vidm10p1)
                          - 1 / 32 * (Vid0p1m1 + Vid0m1p1))  # auxiliary filter

    return DecVid

def VidHaarRec3S(input, Lev=2):
    DecVid = input.clone()
    if Lev == 1:
        Vid                 = HaarRec3S(DecVid, 1)
    elif Lev == 2:
        DecVid[:, :, :, 5]  = HaarRec3S(DecVid[:, :, :, :6], 2)
        Vid                 = HaarRec3S(DecVid[:, :, :, 5:], 1)
    else:
        raise ValueError('Lev is too big')

    return Vid

def HaarRec3S(DecVid, Lev):
    Vid0 = DecVid[:, :, :, 0]

    # Perform circular shifts using torch.roll
    shift_amount = 2 ** (Lev - 1)
    Vid1 = torch.roll(Vid0, shifts=[0, 0, -shift_amount], dims=[0, 1, 2])  # 0 0 1
    Vid2 = torch.roll(Vid0, shifts=[0, -shift_amount, 0], dims=[0, 1, 2])  # 0 1 0
    Vid3 = torch.roll(Vid0, shifts=[0, -shift_amount, -shift_amount], dims=[0, 1, 2])  # 0 1 1
    Vid4 = torch.roll(Vid0, shifts=[-shift_amount, 0, 0], dims=[0, 1, 2])  # 1 0 0
    Vid5 = torch.roll(Vid0, shifts=[-shift_amount, 0, -shift_amount], dims=[0, 1, 2])  # 1 0 1
    Vid6 = torch.roll(Vid0, shifts=[-shift_amount, -shift_amount, 0], dims=[0, 1, 2])  # 1 1 0
    Vid7 = torch.roll(Vid0, shifts=[-shift_amount, -shift_amount, -shift_amount], dims=[0, 1, 2])  # 1 1 1

    # Initialize Vid
    Vid = (Vid0 + Vid1 + Vid2 + Vid3 + Vid4 + Vid5 + Vid6 + Vid7) / 8

    # Perform the reconstruction based on different filters

    # (1,0,0)-(0,0,0) x-axis
    Vid0 = DecVid[:, :, :, 1]
    Vid0 = torch.roll(Vid0, shifts=[-shift_amount, 0, 0], dims=[0, 1, 2]) - Vid0
    Vid += 1 / 4 * Vid0

    # (0,1,0)-(0,0,0) y-axis
    Vid0 = DecVid[:, :, :, 2]
    Vid0 = torch.roll(Vid0, shifts=[0, -shift_amount, 0], dims=[0, 1, 2]) - Vid0
    Vid += 1 / 4 * Vid0

    # (1,1,0)-(0,0,0) xy-plane 1
    Vid0 = DecVid[:, :, :, 3]
    Vid0 = torch.roll(Vid0, shifts=[-shift_amount, -shift_amount, 0], dims=[0, 1, 2]) - Vid0
    Vid += np.sqrt(2) / 8 * Vid0

    # (1,0,0)-(0,1,0) xy-plane 2
    Vid0 = DecVid[:, :, :, 4]
    Vid0 = torch.roll(Vid0, shifts=[-shift_amount, 0, 0], dims=[0, 1, 2]) - torch.roll(Vid0, shifts=[0, -shift_amount, 0], dims=[0, 1, 2])
    Vid += np.sqrt(2) / 8 * Vid0

    # Auxiliary filter
    Vid += DecVid[:, :, :, 5]

    return Vid

def Kernel_Rec_ks_C_I_Pro(ks_data, Ker, Pro):
    ks_Rec = torch.zeros_like(ks_data)
    for coi in range(ks_data.size(2)):
        ks_Rec[:, :, coi] = Pro * (ImgRecFram(ks_data, Ker[:, :, :, coi]) - ks_data[:, :, coi])

    return ks_Rec

def Kernel_Rec_ks_C_Pro(ks_data, Ker, Pro):
    ks_Rec = torch.zeros_like(ks_data)
    for coi in range(ks_data.size(2)):
        ks_Rec[:, :, coi] = Pro * ImgRecFram(ks_data, Ker[:, :, :, coi])

    return ks_Rec

def CCN_Kernel_Rec_ks_C_Pro_D2(ks_data, Ker, Pro, ACS_Line):
    h, w, c = ks_data.shape
    center_h = h // 2
    start_h = center_h - ACS_Line // 2
    end_h = center_h + ACS_Line // 2
    center_w = w // 2
    start_w = center_w - ACS_Line // 2
    end_w = center_w + ACS_Line // 2

    HFS = Ker.shape[0] // 2
    ks_Rec = torch.zeros_like(ks_data)

    tmp = torch.zeros_like(ks_data)
    for coi in range(ks_data.size(2)):
        ks_Rec[:, :, coi] = ImgRecFram(ks_data, Ker[:, :, :, coi, 1])
        tmp[start_h-HFS:end_h+HFS, start_w-HFS:end_w+HFS, coi] = ImgRecFram(ks_data[start_h-HFS:end_h+HFS, start_w-HFS:end_w+HFS, :], Ker[:, :, :, coi, 0])
    ks_Rec[start_h:end_h, start_w, :] = (ks_Rec[start_h:end_h, start_w, :] + tmp[start_h:end_h, start_w, :]) / 2
    ks_Rec[start_h:end_h, end_w-1, :] = (ks_Rec[start_h:end_h, end_w-1, :] + tmp[start_h:end_h, end_w-1, :]) / 2
    ks_Rec[start_h, start_w+1:end_w-1, :] = (ks_Rec[start_h, start_w+1:end_w-1, :] + tmp[start_h, start_w+1:end_w-1, :]) / 2
    ks_Rec[end_h-1, start_w+1:end_w-1, :] = (ks_Rec[end_h-1, start_w+1:end_w-1, :] + tmp[end_h-1, start_w+1:end_w-1, :]) / 2
    ks_Rec[start_h+1:end_h-1, start_w+1:end_w-1, :] = tmp[start_h+1:end_h-1, start_w+1:end_w-1, :]

    return Pro * ks_Rec

def CCN_Kernel_Rec_ks_C_Pro_D3(ks_data, Ker, Pro, ACS_Line):
    h, w, c = ks_data.shape
    center_h = h // 2
    start_h = center_h - ACS_Line // 2
    end_h = center_h + ACS_Line // 2
    center_w = w // 2
    start_w = center_w - ACS_Line // 2
    end_w = center_w + ACS_Line // 2

    HFS = Ker.shape[0] // 2


    ks_Rec = torch.zeros_like(ks_data)

    # for coi in range(ks_data.size(2)):
    #     tmp = ImgRecFram(ks_data[:center_h+HFS, :center_w+HFS, :], Ker[:, :, :, coi, 1])
    #     ks_Rec[:center_h, :center_w, coi] = tmp[:center_h, :center_w]
    #     tmp = ImgRecFram(ks_data[center_h-HFS:, center_w-HFS:, :], Ker[:, :, :, coi, 1])
    #     ks_Rec[center_h:, center_w:, coi] = tmp[HFS:, HFS:]
    #
    # for coi in range(ks_data.size(2)):
    #     tmp = ImgRecFram(ks_data[center_h-HFS:, :center_w+HFS, :], Ker[:, :, :, coi, 2])
    #     ks_Rec[center_h:, :center_w, coi] = tmp[HFS:, :center_w]
    #     tmp = ImgRecFram(ks_data[:center_h+HFS, center_w-HFS:, :], Ker[:, :, :, coi, 2])
    #     ks_Rec[:center_h, center_w:, coi] = tmp[:center_h, HFS:]


    for coi in range(ks_data.size(2)):
        ks_Rec[:, :, coi] = ImgRecFram(ks_data, Ker[:, :, :, coi, 1])
        ks_Rec[:, :, coi] = (ks_Rec[:, :, coi] + ImgRecFram(ks_data, Ker[:, :, :, coi, 2])) / 2
    tmp = torch.zeros_like(ks_data)
    for coi in range(ks_data.size(2)):
        tmp[start_h-HFS:end_h+HFS, start_w-HFS:end_w+HFS, coi] = ImgRecFram(ks_data[start_h-HFS:end_h+HFS, start_w-HFS:end_w+HFS, :], Ker[:, :, :, coi, 0])
    ks_Rec[start_h:end_h, start_w:end_w, :] = tmp[start_h:end_h, start_w:end_w, :]

    return Pro * ks_Rec

def EnergyScaling(Thr, cof):
    Thr_min = abs(Thr).min()
    Thr_max = abs(Thr).max()
    Coe_min = abs(cof).min()
    Coe_max = abs(cof).max()
    normalized_Thr = Coe_min + (Coe_max - Coe_min) * 1.0 * (Thr - Thr_min) / (Thr_max - Thr_min)
    return normalized_Thr

def EnergyScaling_3D(Thr, cof):
    n1, n2, n3 = Thr.shape
    min_Thr = Thr.view(-1, n3).max(dim=0, keepdim=True).values.view(1, 1, n3)
    maxCC = cof.abs().view(-1, n3).max(dim=0, keepdim=True).values.view(1, 1, n3)
    normalized_Thr = Thr / min_Thr * maxCC

    return normalized_Thr

def EnergyScaling_Haar(Thr, cof, uk):
    MAINBODY = FINDMAINBODY(abs(uk) / abs(uk).max())
    min_Thr = abs(Thr * MAINBODY.unsqueeze(-1).expand(-1, -1, Thr.shape[-1])).max()
    max_Cof = abs(cof).max()
    normalized_Thr = Thr / min_Thr * max_Cof
    normalized_Thr[:2] *= np.sqrt(2)

    return normalized_Thr

def EnergyScaling_4D(Thr, cof):
    _, _, _, n4 = Thr.shape
    min_Thr = Thr.contiguous().view(-1, n4).max(dim=0, keepdim=True).values.view(1, 1, 1, n4)
    maxCC = cof.abs().view(-1, n4).max(dim=0, keepdim=True).values.view(1, 1, 1, n4)
    normalized_Thr = Thr * (1 / min_Thr) * maxCC
    return normalized_Thr

def PSNR_SSIM_HaarPSI(ref, deg, model_name, save_path, utime=-1):
    image = Image.fromarray(img_as_ubyte(((abs(deg) / torch.max(abs(deg))).to(torch.float32)).cpu().numpy()))
    image.save(f'{save_path}/1-{model_name}.png')

    MAINBODY = FINDMAINBODY(ref)
    ref_pt = (ref / torch.max(ref)).to(torch.float32)
    deg_pt = (deg / torch.max(deg)).to(torch.float32)
    ref_np = crop_center((ref_pt * MAINBODY).cpu().numpy(), MAINBODY)
    deg_np = crop_center((deg_pt * MAINBODY).cpu().numpy(), MAINBODY)
    ref_pt_c = torch.from_numpy(ref_np).to(ref.device).to(torch.float32)
    deg_pt_c = torch.from_numpy(deg_np).to(deg.device).to(torch.float32)

    psnr = compare_psnr(ref_np, deg_np, data_range=1.)
    ssim = compare_ssim(ref_np, deg_np, multichannel=False, data_range=1.)
    haarpsi = HaarPSI(ref_pt_c, deg_pt_c)
    dists = compute_DISTS(ref_pt_c, deg_pt_c)
    print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, HaarPSI: {haarpsi[0].item():.4f}, DISTS: {dists.item():.4f}\n")

    ref_crop = crop_center(ref_pt.cpu().numpy(), MAINBODY)
    image = Image.fromarray(img_as_ubyte(ref_crop))
    image.save(f'{save_path}/2-reference-crop.png')
    deg_crop = crop_center(deg_pt.cpu().numpy(), MAINBODY)
    image = Image.fromarray(img_as_ubyte(deg_crop))
    image.save(f'{save_path}/2-{model_name}-crop.png')
    error_color_picture(np.abs(ref_crop-deg_crop), f'{save_path}/3-{model_name}-error.png')

    # image = image.convert("RGB")
    # draw = ImageDraw.Draw(image)
    # draw.text((2, 2), f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, HaarPSI: {haarpsi[0].item():.4f}, DISTS: {dists.item():.4f}", fill=(255, 0, 0))
    # image.save(f'{save_path}/Index-{model_name}.png')
    return [model_name, psnr, ssim, haarpsi[0].item(), dists.item(), utime]

def compute_HaarPSI(ref, deg):
    ref_ = ref / torch.max(ref)
    deg_ = deg / torch.max(deg)
    haarpsi = HaarPSI(ref_.to(torch.float32), deg_.to(torch.float32))
    return haarpsi[0].item()

def crop_center(array, MAINBODY):
    non_zero_indices = np.nonzero(MAINBODY)
    min_row, max_row = non_zero_indices[:,0].min(), non_zero_indices[:,0].max()
    min_col, max_col = non_zero_indices[:,1].min(), non_zero_indices[:,1].max()
    res = array[min_row:max_row + 1, min_col:max_col + 1]
    res = res / np.max(res)

    return res

def error_color_picture(error, save_file, level=-1):
    if level == -1:
        plt.imshow(error, cmap='jet', interpolation='nearest')
    else:
        plt.imshow(error, cmap='jet', interpolation='nearest', vmax=level)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭绘图窗口

def tensor_split(tensor, dim=1, each_channel=2):
    tensor_splitT = torch.split(tensor.unsqueeze(-1), each_channel, dim=dim)
    return torch.cat(tensor_splitT, dim=-1).mean(-1)

def i2r(tensor):
    tensor = torch.stack([tensor.real, tensor.imag], dim=-1)
    return tensor

def r2i(tensor):
    tensor = tensor[..., 0] + 1j * tensor[..., 1]
    return tensor

def rss_complex(tensor):
    return torch.sqrt((abs(tensor) ** 2).sum(dim=-1))

def VarGuaEstimation(Data):
    Row, Col = Data.shape
    Template = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    Template = torch.tensor(Template, dtype=torch.float32, device=Data.device)
    DataNoi = imfilter_symmetric(Data, Template)
    ModRate = 1
    Var_Abs = (ModRate * np.sqrt(np.pi / 2) * torch.sum(torch.abs(DataNoi))) / (6 * (Row - 2) * (Col - 2))
    Var_Squ = torch.sum(torch.abs(DataNoi ** 2)) / (36 * (Row - 2) * (Col - 2))
    return torch.sqrt(Var_Squ * ModRate)

def VarGuaEstimation_Margin(Data):
    Row, Col = Data.shape
    row_ = int(np.floor(Row * 0.3))
    col_ = int(np.floor(Col * 0.3))
    Template = torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=torch.float64, device=Data.device).unsqueeze(0).unsqueeze(0)

    # 定义四个角落的索引范围
    corners = [(slice(None, row_), slice(None, col_)),
               (slice(None, row_), slice(-col_, None)),
               (slice(-row_, None), slice(None, col_)),
               (slice(-row_, None), slice(-col_, None))]

    # 计算四个角落的卷积并求平方和
    total_noise = sum((F.conv2d(abs(Data[c[0], c[1]]).unsqueeze(0).unsqueeze(0), Template, padding=0).squeeze() ** 2) for c in corners)

    Var_Squ = torch.sum(total_noise) / (36 * 4 * (row_ - 2) * (col_ - 2))
    return torch.sqrt(Var_Squ)

def SubbandSTD_NoiVar_DoubleFrame(Coef, STD_NoiVar, T_FilOpe1, T_FilOpe2):
    STD_NoiVar = STD_NoiVar * STD_NoiVar

    Wei1 = torch.sum(T_FilOpe1 ** 2, dim=(0, 1)).view(T_FilOpe1.size(2), 1)
    Wei2 = torch.sum(T_FilOpe2 ** 2, dim=(0, 1)).view(T_FilOpe2.size(2), 1)

    Const1 = T_FilOpe1.size(-1)  # The number of framelet subband
    Const2 = T_FilOpe2.size(-1)  # The number of framelet subband
    StoreLev = Const1 + Const2 - 1  # store in DecImg(:, :, 0 ~ StoreLev)

    ThreSigma = torch.zeros(StoreLev, dtype=torch.float32, device=Coef.device)
    ThreSigma[StoreLev - Const1 : StoreLev] = Wei1.squeeze() * STD_NoiVar
    StoreLev -= Const1
    STD_NoiVar *= Wei1[0, 0]  # 更新 STD_NoiVar
    ThreSigma[StoreLev - Const2 + 1 : StoreLev + 1] = Wei2.squeeze() * STD_NoiVar

    # 对 ThreSigma 进行平方根变换
    ThreSigma = torch.sqrt(ThreSigma)

    return ThreSigma

def PixelLocalVariance_Lev_DoubleFrame(Coe, T_FilOpe1, T_FilOpe2):
    KerSize1 = T_FilOpe1.size(0)
    KerSize2 = T_FilOpe2.size(0)

    Const1 = T_FilOpe1.size(-1) - 1
    Const2 = T_FilOpe2.size(-1) - 1
    StoreLev = Const1 + Const2

    VarLoc = torch.zeros_like(Coe)

    AveKer1 = torch.ones((KerSize1, KerSize1), dtype=torch.float32, device=Coe.device)
    AveKerNum = torch.sum(AveKer1)
    Const_Ave = np.sqrt(2) / AveKerNum
    AveKer1 *= Const_Ave

    for i in range(StoreLev - Const1, StoreLev):
        VarLoc[:,:,i] = imfilter_symmetric(torch.abs(Coe[:,:, i]), AveKer1)
    StoreLev = StoreLev - Const1

    AveKer2 = torch.ones((KerSize2, KerSize2), dtype=torch.float32, device=Coe.device)
    AveKerNum = torch.sum(AveKer2)
    Const_Ave = np.sqrt(2) / AveKerNum
    AveKer2 *= Const_Ave

    for i in range(StoreLev - Const2, StoreLev):
        VarLoc[:,:,i] = imfilter_symmetric(torch.abs(Coe[:,:, i]), AveKer2)

    return abs(VarLoc)

def ImgDoubleFrameDec2_DCT7(Img, TNTF):
    Const1 = TNTF.T_FilOpe1.size(-1)
    Const2 = TNTF.T_FilOpe2.size(-1)
    StoreLev = Const1 + Const2 - 1

    DecImg = torch.zeros(size=(Img.size(0), Img.size(1), StoreLev), dtype=Img.dtype, device=Img.device)
    DecImg[:, :, StoreLev - Const1:StoreLev] = DCT7Dec2(Img, 1)
    StoreLev = StoreLev - Const1

    DecImg[:, :, StoreLev - Const2 + 1 : StoreLev + 1] = ImgDecFram(DecImg[:, :, StoreLev], TNTF.T_FilOpe2)
    return DecImg

def ImgDoubleFrameRec2_DCT7(input, TNTF):
    DecImg = input.clone()
    Const2 = TNTF.T_FilOpe2.size(-1)
    StoreLev = 0

    DecImg[:, :, StoreLev + Const2 - 1] = ImgRecFram(DecImg[:, :, StoreLev : StoreLev + Const2], TNTF.T_TraFilOpe2)
    StoreLev = StoreLev + Const2
    RecImg = DCT7Rec2(DecImg[:, :, StoreLev - 1 :], TNTF.T_TraFilOpe1)

    return RecImg

def DenoiseByDiffusion(DualDomain, x, step=1):
    FLAG = x.ndim == 2
    x = i2r(x).unsqueeze(0).permute(0, 3, 1, 2) if FLAG else i2r(x).permute(2, 3, 0, 1)
    x = x.contiguous()
    with torch.no_grad():
        for _ in range(step):
            sigma = torch.tensor(DualDomain.lam).to(x.device).view(1, 1, 1, 1)
            noise = torch.randn_like(x[0, ...]).unsqueeze(0) * sigma
            x_max = x.pow(2).sum(1).sqrt().max()
            inputs = (x / x_max + noise).to(torch.float32)
            logP = torch.zeros_like(inputs)
            for coil in range(inputs.shape[0]):
                logP[coil, ...] = DualDomain.scoreNet(inputs[coil, ...].unsqueeze(0), sigma)
            res = x + noise + sigma * logP
            # res = x + noise + sigma * (logP * sigma ** 2)
            DualDomain.lam = DualDomain.lam / DualDomain.gamma
    return r2i(res.squeeze().permute(1, 2, 0)) if FLAG else r2i(res.permute(2, 3, 0, 1))

def DenoiseByDiffusion_HighPass(DualDomain, input, step=1):
    x = input.permute(2, 3, 0, 1)
    x = x.contiguous()
    with torch.no_grad():
        for _ in range(step):
            sigma = math.sqrt(DualDomain.lam / DualDomain.rho)
            step_lr = DualDomain.delta
            sigma = torch.tensor(sigma).to(x.device).view(1, 1, 1, 1)
            noise = torch.randn_like(x) * sigma
            x_max = x.pow(2).sum(1).sqrt().max()
            inputs = (x / x_max + noise).to(torch.float32)
            logP = torch.zeros_like(inputs)
            for coil in range(inputs.shape[0]):
                logP[coil, :, :, :] = DualDomain.scoreNet_high(inputs[coil, :, :, :].unsqueeze(0), sigma)
            res = x + (step_lr * sigma ** 2) * logP
            DualDomain.rho = DualDomain.gamma * DualDomain.rho
    return res.permute(2, 3, 0, 1)

def SolverForXProblem(AFunction, target, x):
    target += x
    r = target - AFunction(x)
    p = r.clone()
    rTr = torch.abs(torch.dot(r.view(-1).conj(), r.view(-1)))
    for iter in range(5):
        Ap = AFunction(p)
        a = rTr / torch.abs(torch.dot(p.view(-1).conj(), Ap.view(-1)))
        x += a * p
        r -= a * Ap
        rTrk = torch.abs(torch.dot(r.view(-1).conj(), r.view(-1)))
        b = rTrk / rTr
        p = r + b * p
        rTr = rTrk
    return x

def SolverForSubProblem(AFunction, target, x_in, iter=5):
    x = x_in.clone()
    r = target - AFunction(x)
    p = r.clone()
    rTr = torch.abs(torch.dot(r.view(-1).conj(), r.view(-1)))
    for iter in range(iter):
        Ap = AFunction(p)
        a = rTr / torch.abs(torch.dot(p.view(-1).conj(), Ap.view(-1)))
        x += a * p
        r -= a * Ap
        rTrk = torch.abs(torch.dot(r.view(-1).conj(), r.view(-1)))
        b = rTrk / rTr
        p = r + b * p
        rTr = rTrk
        # print(f"At iteration {iter + 1}, err = {torch.norm(torch.abs(rTr)):.4f}")
    return x

def ESPIRiT_sensemap_estimate(ksfull, mask):
    X = np.zeros_like(ksfull, dtype=np.complex)
    for i in range(ksfull.shape[-1]):
        X[:, :, i] = np.fft.fft2(np.fft.fftshift(np.fft.ifft2(ksfull[:, :, i])))
    X = X[np.newaxis, ...]
    calib, AC_region = getCalibSize_1D_Edt(mask)
    ACS_line = np.abs(AC_region[1] - AC_region[0]) + 1
    esp = np.squeeze(espirit(X, 6, ACS_line, 0.02, 0.99))
    esp = esp[..., 0]
    for idx in range(ksfull.shape[-1]):
        plt.subplot(1, ksfull.shape[-1], idx+1)
        plt.imshow(np.abs(esp[:, :, idx]), cmap='gray')
        plt.axis('off')
    plt.show()
    return esp

def imageShow(image):
    temp = abs(image) / abs(image).max()
    # temp[temp > 0.05] = 0.05
    plt.imshow(temp, cmap='gray')
    plt.axis('off')
    plt.show()

def FINDMAINBODY(image, alpha=1.5, BlackPoint=15):
    AveKer = np.ones((3, 3))
    if torch.is_tensor(image):
        res = abs(image).cpu().numpy()
    else:
        res = image
    res = convolve(res, AveKer, mode='nearest')
    res = np.power(res / res.max(), alpha)
    res[res <= BlackPoint / 255] = BlackPoint / 255
    res = np.sqrt((res - res.min()) / (res.max() - res.min()))
    res[res > 1e-6] = 1
    res[res != 1] = 0
    res = filling(res)
    M_inverted = 1 - res
    labeled_array, num_features = label(M_inverted)
    area_threshold = 90000
    BW_no_small_objects = np.copy(M_inverted)
    for region_label in range(1, num_features + 1):
        region_area = np.sum(labeled_array == region_label)
        if region_area < area_threshold:
            BW_no_small_objects[labeled_array == region_label] = 0
    res = 1 - BW_no_small_objects
    if torch.is_tensor(image):
        return torch.from_numpy(res).to(image.device)
    else:
        return res

def filling(I, windowSize=5, thre=0.95):
    h, w = I.shape
    Hmin = h
    Hmax = 0
    halfWindow = windowSize // 2
    for i in range(w):
        index = np.where(I[:, i] == 1)[0]  # 获取当前列中为1的索引
        if index.size == 0:
            continue
        for j in index:
            submatrix = I[max(0, j - halfWindow):min(h, j + halfWindow + 1),
                          max(0, i - halfWindow):min(w, i + halfWindow + 1)]
            if np.mean(submatrix) >= thre:
                Hmin = min(Hmin, j)
                break
        for j in reversed(index):
            submatrix = I[max(0, j - halfWindow):min(h, j + halfWindow + 1),
                          max(0, i - halfWindow):min(w, i + halfWindow + 1)]
            if np.mean(submatrix) >= thre:
                Hmax = max(Hmax, j)
                break
    se = np.ones((10, 10), dtype=int)
    I[Hmin:Hmax + 1, :] = binary_dilation(I[Hmin:Hmax + 1, :], structure=se)
    return I

def Multi_TNTF(COM, input):
    Const1 = COM.T_FilOpe1.size(-1)
    Const2 = COM.T_FilOpe2.size(-1)
    StoreLev = Const1 + Const2 - 1
    H, W, C = input.shape
    multi_coil_TFcoef = torch.zeros(H, W, C, StoreLev, dtype=input.dtype, device=input.device)
    for i in range(C):
        multi_coil_TFcoef[:, :, i, :] = ImgDoubleFrameDec2_DCT7(input[:, :, i], COM)

    # # ---------------------------------------------------------------
    # aveKer = torch.ones((5, 5), dtype=torch.float32, device=input.device) / 25
    # multi_coil_TFcoef[..., 0] = imfilter_symmetric_3D(multi_coil_TFcoef[..., 0], aveKer)
    # multi_coil_TFcoef[..., 0] = imfilter_symmetric_3D(multi_coil_TFcoef[..., 0], aveKer)
    # # ---------------------------------------------------------------

    return multi_coil_TFcoef

def Multi_TNTF_T(COM, input):
    H, W, C, N = input.shape
    multi_coil = torch.zeros(H, W, C, dtype=input.dtype, device=input.device)
    for i in range(C):
        multi_coil[:, :, i] = ImgDoubleFrameRec2_DCT7(input[:, :, i, :], COM)
    return multi_coil

def getH(x):
    h, w, c = x.shape
    res = torch.zeros(h, w, c, c, dtype=x.dtype, device=x.device)
    for i in range(c):
        tmp = torch.zeros_like(x)
        for j in range(c):
            if j != i:
                tmp[..., i] += x[..., j]
                tmp[..., j] = -x[..., i]
        res[..., i] = tmp
    return res

def getHTH(H):
    h, w, c,_ = H.shape
    res = torch.zeros(h, w, c, c, dtype=H.dtype, device=H.device)
    for i in range(c):
        for j in range(c):
            tmp = torch.zeros(h, w, dtype=H.dtype, device=H.device)
            for k in range(c):
                tmp += torch.conj(H[..., i, k]) * H[..., j, k]
            res[..., j, i] = tmp
    return res

def HTHx(x, HTH):
    res = torch.zeros_like(x)
    for i in range(x.shape[-1]):
        res[..., i] = torch.sum(HTH[..., i] * x, dim=-1)
    return res
