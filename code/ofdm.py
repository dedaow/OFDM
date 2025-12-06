import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import convolve
from commpy.channelcoding import Trellis, conv_encode, viterbi_decode


# ========== 辅助函数定义 ==========

def mseq(stage, ptap, regi, length):
    """生成m序列"""
    register = np.array(regi, dtype=int)
    seq = np.zeros(length, dtype=int)
    
    for i in range(length):
        seq[i] = register[-1]
        feedback = 0
        for tap in ptap:
            feedback ^= register[tap-1]
        register = np.roll(register, 1)
        register[0] = feedback
    
    return seq

def spread(data, code):
    """扩频函数"""
    spread_factor = len(code)
    rows, cols = data.shape
    spread_data = np.zeros((rows * spread_factor, cols), dtype=complex)
    
    for col in range(cols):
        for row in range(rows):
            spread_data[row*spread_factor:(row+1)*spread_factor, col] = data[row, col] * code
    
    return spread_data

def despread(data, code):
    """解扩函数"""
    spread_factor = len(code)
    rows, cols = data.shape
    despread_data = np.zeros((rows // spread_factor, cols), dtype=complex)
    
    for col in range(cols):
        for row in range(rows // spread_factor):
            despread_data[row, col] = np.sum(data[row*spread_factor:(row+1)*spread_factor, col] * code) / spread_factor
    
    return despread_data

def pskmod(data, M, phase_offset=0):
    """PSK调制"""
    constellation = np.exp(1j * (2 * np.pi * np.arange(M) / M + phase_offset))
    return constellation[data]

def pskdemod(signal, M, phase_offset=0):
    """PSK解调"""
    constellation = np.exp(1j * (2 * np.pi * np.arange(M) / M + phase_offset))
    signal = signal.flatten()
    distances = np.abs(signal[:, np.newaxis] - constellation[np.newaxis, :])
    return np.argmin(distances, axis=1)

def awgn(signal, snr_db):
    """添加高斯白噪声"""
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise

# ========== 参数设置 ==========

N_sc = 52          # 系统子载波数（不包括直流载波）
N_fft = 64         # FFT长度
N_cp = 16          # 循环前缀长度
N_symbo = N_fft + N_cp  # 1个完整OFDM符号长度
N_c = 53           # 包含直流载波的总的子载波数
M = 4              # 4PSK调制
SNR = np.arange(0, 26, 1)  # 仿真信噪比
N_frm = 10         # 每种信噪比下的仿真帧数
Nd = 6             # 每帧包含的OFDM符号数
P_f_inter = 6      # 导频间隔
data_station = []  # 导频位置
L = 7              # 卷积码约束长度
tblen = 6 * L      # Viterbi译码器回溯深度
stage = 3          # m序列的阶数
ptap1 = [1, 3]     # m序列的寄存器连接方式
regi1 = [1, 1, 1]  # m序列的寄存器初始值

# ========== 基带数据产生 ==========
np.random.seed(42)  # 设置随机种子以便复现
P_data = np.random.randint(0, 2, N_sc * Nd * N_frm)

# ========== 信道编码（卷积码） ==========
# (2,1,7)卷积编码
memory = np.array([L-1])
g_matrix = np.array([[0o133, 0o171]])  # 八进制转换
trellis = Trellis(memory, g_matrix)
code_data = conv_encode(P_data, trellis)

# ========== QPSK调制 ==========
data_temp1 = code_data.reshape(-1, int(np.log2(M)))  # 每组2比特分组
data_temp2 = data_temp1.dot(2**np.arange(int(np.log2(M)))[::-1])  # 二进制转十进制
modu_data = pskmod(data_temp2, M, np.pi/M)  # 4PSK调制

# 星座图
plt.figure(1, figsize=(8, 6))
plt.scatter(modu_data.real, modu_data.imag, alpha=0.5)
plt.grid(True)
plt.title('QPSK Constellation')
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.axis('equal')

# ========== 扩频 ==========
code = mseq(stage, ptap1, regi1, N_sc)  # 扩频码生成
code = code * 2 - 1  # 将1、0变换为1、-1

# 调整modu_data长度使其能被N_sc整除
num_symbols = len(modu_data)
pad_len = (N_sc - num_symbols % N_sc) % N_sc
if pad_len > 0:
    modu_data = np.concatenate([modu_data, np.zeros(pad_len, dtype=complex)])

modu_data = modu_data.reshape(N_sc, -1)
spread_data = spread(modu_data, code)  # 扩频
spread_data = spread_data.flatten()

# ========== 插入导频 ==========
P_f = 3 + 3*1j  # 导频
P_f_station = np.arange(0, N_fft, P_f_inter)  # 导频位置（Python从0开始）
pilot_num = len(P_f_station)

for img in range(N_fft):
    if img % P_f_inter != 0:
        data_station.append(img)

data_row = len(data_station)
data_col = int(np.ceil(len(spread_data) / data_row))

pilot_seq = np.ones((pilot_num, data_col), dtype=complex) * P_f  # 导频矩阵
data = np.zeros((N_fft, data_col), dtype=complex)  # 预设整个矩阵
data[P_f_station, :] = pilot_seq  # 放入导频

if data_row * data_col > len(spread_data):
    data2 = np.concatenate([spread_data, np.zeros(data_row * data_col - len(spread_data), dtype=complex)])
else:
    data2 = spread_data

# ========== 串并转换 ==========
data_seq = data2.reshape(data_row, data_col)
data[data_station, :] = data_seq  # 将导频与数据合并

# ========== IFFT ==========
ifft_data = np.fft.ifft(data, axis=0)

# ========== 插入保护间隔、循环前缀 ==========
Tx_cd = np.vstack([ifft_data[N_fft-N_cp:, :], ifft_data])

# ========== 并串转换 ==========
Tx_data = Tx_cd.flatten()

# ========== 信道传输与误码率计算 ==========
Ber = np.zeros(len(SNR))
Ber2 = np.zeros(len(SNR))

for jj, snr in enumerate(SNR):
    print(f"Processing SNR = {snr} dB...")
    
    # 添加高斯白噪声
    rx_channel = awgn(Tx_data, snr)
    
    # 串并转换
    Rx_data1 = rx_channel.reshape(N_fft + N_cp, -1)
    
    # 去掉保护间隔、循环前缀
    Rx_data2 = Rx_data1[N_cp:, :]
    
    # FFT
    fft_data = np.fft.fft(Rx_data2, axis=0)
    
    # 信道估计与插值（均衡）
    data3 = fft_data[:N_fft, :]
    Rx_pilot = data3[P_f_station, :]  # 接收到的导频
    h = Rx_pilot / pilot_seq
    
    # 线性插值
    H_list = []
    for col in range(data_col):
        f = interp1d(P_f_station, h[:, col], kind='linear', fill_value='extrapolate')
        H_list.append(f(data_station))
    H = np.array(H_list).T
    
    # 信道校正
    data_aftereq = data3[data_station, :] / H
    
    # 并串转换
    data_aftereq = data_aftereq.flatten()
    data_aftereq = data_aftereq[:len(spread_data)]
    data_aftereq = data_aftereq.reshape(N_sc, -1)
    
    # 解扩
    demspread_data = despread(data_aftereq, code)
    
    # QPSK解调
    demodulation_data = pskdemod(demspread_data, M, np.pi/M)
    De_data1 = demodulation_data.flatten()
    De_data2 = np.array([list(np.binary_repr(val, width=int(np.log2(M)))) for val in De_data1], dtype=int)
    De_Bit = De_data2.flatten()
    
    # 信道译码（维特比译码）
    rx_c_de = viterbi_decode(De_Bit[:len(code_data)], trellis, tb_depth=tblen)
    
    # 计算误码率
    min_len = min(len(De_Bit), len(code_data))
    Ber2[jj] = np.sum(De_Bit[:min_len] != code_data[:min_len]) / min_len  # 译码前
    
    min_len2 = min(len(rx_c_de), len(P_data))
    Ber[jj] = np.sum(rx_c_de[:min_len2] != P_data[:min_len2]) / min_len2  # 译码后

# ========== 绘制误码率曲线 ==========
plt.figure(2, figsize=(10, 6))
plt.semilogy(SNR, Ber2, 'b-s', label='4PSK调制、卷积码译码前（有扩频）')
plt.semilogy(SNR, Ber, 'r-o', label='4PSK调制、卷积码译码后（有扩频）')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('AWGN信道下误比特率曲线')
plt.legend()
plt.grid(True)

# ========== 绘制发送和接收数据对比 ==========
plt.figure(3, figsize=(10, 8))

plt.subplot(2, 1, 1)
x = np.arange(31)
plt.stem(x, P_data[:31])
plt.ylabel('Amplitude')
plt.title('发送数据（以前30个数据为例）')
plt.legend(['4PSK调制、卷积译码、有扩频'])
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(x, rx_c_de[:31])
plt.ylabel('Amplitude')
plt.title('接收数据（以前30个数据为例）')
plt.legend(['4PSK调制、卷积译码、有扩频'])
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n仿真完成！")
print(f"最终BER (SNR={SNR[-1]}dB): {Ber[-1]:.2e}")
