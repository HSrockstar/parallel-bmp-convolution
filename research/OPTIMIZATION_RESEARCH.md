# BMP 5×5 卷积加速调研、实现与验证

调研与实验日期：2026-11-24。仓库基线：`2c85bf75de84ae82e33731c2b03e532016fcf8cf`。本文面向固定 5×5 核、24 位 BGR、步长 1、常数零填充的现有作业。源文献、实现入口、实际运行结果分别列出。

## 1. 结论与交付

仓库已经完成分离卷积、五行环形缓冲、BGR 连续布局、mmap 和文件预分配等重要优化。进一步实验应优先比较对称系数合并、显式 SIMD、水平与垂直顺序交换、线程复用，以及针对图像尺寸选择线程数和分块大小。FFT、Winograd、积分图等也有可运行实现，适合作为算法对照；对于固定五点一维核，变换和额外访存的成本很容易超过减少乘法的收益。

本次交付包含：

- `kernels.c` 中 17 个 CPU 方法，均可通过 Pthread 与 OpenMP 两个程序执行。
- `alternatives.py` 中 17 个 NumPy/SciPy 实验入口，覆盖矩阵展开、积分图、DCT、低秩分解、短核级联、滑动 DFT 和改变滤波器的近似方案。
- `cuda_kernels.cu` 与 `gpu.py` 中 3 个 GPU 实现，包含直接卷积、两趟分离和共享内存融合。当前环境没有 CUDA 设备、CuPy 或 nvcc，这部分已完成源码与测试入口，尚未编译和实机验证。
- 可复现的正确性测试、随机交错基准、自动调参程序和原始 JSON/CSV 结果。
- 原有两个作业程序的 BMP 解析、像素偏移、内存分配和线程创建错误处理修复。

原有命令继续使用原计算方法。新增方法通过 `research/conv_lab_pthread` 和 `research/conv_lab_omp` 选择。默认方法仍为 `ring`，性能结论由目标机器上的测试决定。

## 2. 仓库审计

### 2.1 已有优化与历史实验

`作业/代码/conv_pthread.c` 和 `conv_omp.c` 使用相同的近似秩 1 核，水平中间值保存为 float。每个工作线程保存五行，按输出条带独立计算。水平相邻像素同通道相隔 3 字节，输出直接写入 BMP 映射。输入按文件行序处理，因为核关于垂直方向对称。

历史日志已经记录通道平面化、PGO、huge page、顺序读取建议、线程绑核、定点、x 分块和双行输出复用的实验。部分方法在 i9-13900K 上回退。本次保留独立入口，使其他 CPU、编译器和图像尺寸可以复测这些假设。

### 2.2 本次修复的问题

| 原问题 | 影响 | 修复 |
|---|---|---|
| 用 `uint32_t*` 强转 BMP 字节头 | 未对齐访问和严格别名违规 | `bmp_safe.h` 按小端逐字节读写 |
| 输出头写偏移 54，输出像素却使用输入偏移 | 输入带扩展头或间隙时输出损坏 | 输入保留其偏移，输出统一为 54 字节标准头 |
| 宽高、步长、负高度和文件长度未验证 | 整数溢出、越界映射、截断文件触发 SIGBUS | 检查尺寸、DIB、planes、bpp、compression、实际文件长度 |
| 输入输出路径可指向同一 inode | `O_TRUNC` 会破坏输入 | 打开后先检查设备号与 inode，再截断输出，覆盖软硬链接 |
| `conv_band` 分配失败静默返回 | 可能把未计算区域当成成功结果 | 返回错误并传给主线程 |
| `pthread_create` 返回值未检查 | 可能等待无效线程 | 失败后回收已启动线程并返回非零 |
| `atoi` 接受混合字符串或溢出值 | 参数行为不确定 | 严格正整数解析 |
| 每个条带最后多做一行水平卷积 | 无用工作 | 删除该行计算 |
| `posix_fallocate` 不支持即退出 | 部分文件系统无法运行 | 不支持时保留 `ftruncate`，真实容量等错误仍失败 |

BMP 输出统一生成 BITMAPINFOHEADER，保留宽高和行序，扩展头元数据不复制。测试覆盖非零输入行尾填充，输出填充清零。

## 3. 数值目标

教师提供的核每项只有八位小数，实际矩阵的秩为 3，现有一维核的外积为其近似。原仓库已有的微小误差来自核分解、float 运算、FMA 和舍入规则。评价时应同时看最大绝对误差、不同字节数、MSE 和验卷相似度。

`exact-int` 使用八位小数系数乘以 100000000 后的整数，uint64 累加，最后执行 `(sum + 50000000) / 100000000`。这给出针对这些十进制系数的精确半向上舍入参考。最大累加值约为 255×10^8，超过 uint32 范围，因此使用 64 位。独立 Python 测试重新实现同一数学定义，逐像素验证；没有调用待测 C 内核生成小图参考。

OpenCV 转为字节的平局处理可能与半向上不同。本文将实际系数语义、浮点误差和最终字节比较分开，避免用不同的舍入方式替代算法错误检查。CPU 优化路径要求最大字节误差不超过 1，相似度大于 99；`exact-int` 和 `lut` 要求与整数参考完全一致。

## 4. 检索范围与停止依据

本轮按照算法结构检索，不把 tile 大小、向量宽度和线程数的每个组合算作新算法。

| 检索批次 | 主要查询与资料 | 新增内容 |
|---|---|---|
| 1 | Gaussian convolution survey、separable FIR、Halide blur scheduling | FIR、分离、环形缓冲、分块、DFT/DCT、盒式与 IIR 家族 |
| 2 | SIMD quantization、CUDA separable/shared memory、OpenCV filter2D | 定点、显式向量、直接/两趟/融合 GPU、库实现比较 |
| 3 | Winograd small convolution、kernel algebra、sliding DCT/DST | 小块变换、短核因式分解、滑动频域递推 |
| 4 | Gaussian blur integral image/recursive SIMD、fast 5×5 integer、GCC/OpenMP | 已有家族的实现变体，以及编译、缓存和调度控制 |

最后的补充检索没有增加适用于本项目的新算法家族，因此停止本轮收集。这个结论表示检索覆盖达到实用饱和；未来更换滤波器、硬件、输入批量或延迟目标时，应重新搜索。参考文献保留了被调查但未采用的方向，方便继续追踪。

## 5. CPU 方法与代码对应

全部方法位于 [kernels.c](kernels.c)，共用 [kernels.h](kernels.h) 的条带接口。`--list` 报告当前机器的可执行方法。

| CLI 名称 | 方法与具体实现 | 数值目标 | 成本与适用条件 |
|---|---|---|---|
| `ring` | 五行 float 环形缓冲，内部五项顺序累加 | 现有秩 1 近似 | 基准方法，空间 O(W) / 线程 |
| `symmetric` | 对称像素先相加，再乘 a、b、c | 同一个秩 1 核 | 每方向乘法从 5 次降到 3 次，改变浮点结合次序 |
| `tiled` | 对称分离，沿 x 分块，每块五行缓冲 | 同上 | 缩小工作集；每块重复扫描输入行，需实测 tile |
| `vfirst` | 先计算垂直一行，再计算该行的水平输出 | 同上 | 只留一行中间值；相邻输出行依靠源数据缓存复用 |
| `box` | 水平核表示为宽 5、3、1 三个盒式和；滑动更新 | 同上 | 水平加减递推有依赖，垂直保持对称 FIR |
| `prefix` | 每个通道建立 uint64 行前缀和，查询两个区间和 | 同上 | 无滑动累加漂移；增加前缀数组访问 |
| `q15` | 系数 `[3935,7664,9570,7664,3935]`，中间 int32，最终 uint64 | Q15 近似 | 不做中间舍入，避免早期量化误差；本版与历史 int16 方案不同 |
| `winograd` | 第一组三 tap 使用 F(2,3)，末尾两 tap 直接计算，水平每次输出两像素 | 秩 1 核 | 减少水平乘法，增加变换加减；垂直仍为对称 FIR |
| `avx2` | 8 个字节位置一组，显式扩宽、float 运算、饱和打包 | 同上 | 函数级 ISA，运行时检测，精确 8 字节读取 |
| `avx512` | 16 个字节位置一组，显式扩宽和压缩 | 同上 | 检查 AVX-512F/BW；默认通用构建可移植到无 AVX-512 的 x86 |
| `direct` | 25 项 double 直接卷积 | 完整八位小数核 | 简单数学基准 |
| `sym2d` | 二维镜像位置先求和，按九组权重累计 | 完整核 | 减少乘法，边界与分组开销较大 |
| `exact-int` | 十进制整数系数，uint64 精确累加 | 完整核、半向上舍入 | 精度参考与整数对照 |
| `lut` | 预计算 25×256 个 uint64 乘积 | 与 `exact-int` 完全相同 | 乘法换查询，缓存成本可能更高；建表计入计算时间 |
| `rank3` | 三种独立水平核，加三组垂直选择器 | 完整核 | 使用三种不同核行，避免秩 1 拟合误差；中间值更多 |
| `fft` | 每线程条带做零扩展 2D radix-2 FFT，乘核频谱并逆变换 | 完整核 | 每线程两块复数工作区限制为 256 MiB；超限报错 |
| `overlap-save` | FFT 块内读取 halo，丢弃循环污染部分，只写有效输出 | 完整核 | 缓冲大小受控；默认输出块横向来自 `--tile`，纵向 60 行 |

### 5.1 对称分离和计算顺序

一维核为 `[a,b,c,b,a]`，水平运算为 `a·(x[-2]+x[2]) + b·(x[-1]+x[1]) + c·x[0]`。同通道输入偏移为 ±6、±3 字节，不能把相邻 B、G、R 当成空间邻点。`vfirst` 利用水平和垂直线性滤波可交换，改变中间缓存组织。两种顺序在实数运算下等价，float 路径会出现结合次序差异。

`tiled` 的五行 float 缓冲按 64 字节对齐，并把每槽步长补成 16 个 float 的倍数。tile 以像素计数，内部转为字节。不同块的输出互不重叠；所有输入 halo 都来自原图，不读取其他线程尚未写完的输出。[S2][S3]

### 5.2 盒式分解与积分图

对于这组五点权重，精确代数分解为 `a·box5 + (b-a)·box3 + (c-b)·identity`。它和反复做均值模糊有不同的系数定义。C 版本对水平宽 5、3 的窗口分别做滑动加减或前缀查询。宽 5 的字节和最大为 1275，float 可以精确保存这些整数和。

二维完整核还可以表示为九个中心矩形核的加权和。Python `integral` 用二维前缀和查询这些矩形，权重来自二维差分 `C[r,s]=F[r,s]-F[r+1,s]-F[r,s+1]+F[r+1,s+1]`。这里 F 保存按绝对位移排列的核值，越界系数为零。该方法保留完整八位小数核，但对于半径仅 2 的情况，积分图的构建和读取可能占主导。[S1][S4][S5]

### 5.3 秩分解、因式分解和 Winograd

`rank3` 直接利用第 0/4、第 1/3 和第 2 行三组核，垂直选择器分别为 `[1,0,0,0,1]`、`[0,1,0,1,0]`、`[0,0,1,0,0]`。Python `svd3` 使用 SVD 的三个非零分量作独立对照。

`cascade` 把一维对称五点多项式分解成两个对称三点核。令 `t+z=b/a`、`tz=c/a-2`，则 `[a,at,a] * [1,z,1]` 得到原五点系数。级联中间图需要额外 halo，提前截掉外侧中间结果会影响靠边输出。

`winograd` 是 F(2,3) 加两项直接计算的混合方案。它没有声称实现最少乘法的完整 F(2,5)。Winograd 的收益取决于数据变换成本、复用程度和数值范围，CNN 多通道结果不能直接套到三通道独立滤波。[S6][S7][S8]

### 5.4 频域、DCT 与滑动频谱

FFT 使用完整原始核，逆变换每个维度各归一化一次。块输入从目标位置减 2 开始，5 tap 线性卷积的有效结果从局部索引 4 开始。`overlap-save` 每个方向丢弃 4 个污染位置，输出区域不重复写。每个工作线程拥有自己的 FFT 缓冲和频谱。[S9][S10]

Python `dct` 先在四边各补 2 个零，再用对称核的余弦特征值乘 DCT-II 频谱，逆变换后裁剪。零扩展保证目标区域不会读到外侧的反射边界。`sliding-dft` 维护五样本窗口的三个独立频率分量，每次移动更新频谱，每 256 个位置重算一次以限制递推漂移。后者是本项目五点 FIR 的精确频谱实验，和文献中优化高斯近似参数的 sliding DCT/DST-5 有不同实现。[S11]

## 6. 其他可运行方法

### 6.1 NumPy/SciPy 对照

[alternatives.py](alternatives.py) 的所有方法都能读取和输出 BMP，保留上下行序，显式使用零边界。

| 名称 | 实现内容 | 与固定作业核的关系 |
|---|---|---|
| `scipy-direct` | `ndimage.convolve` | 完整核参考 |
| `scipy-fft` | 库 FFT 卷积 | 完整核 |
| `overlap-add` | 分块频域相加 | 完整核，区别于 C 的 overlap-save |
| `dct` | 零扩展后的 DCT 对角化 | 完整核 |
| `integral` | 九个矩形核与二维积分图 | 完整核 |
| `im2col` | 有限行分块展开为 25 列，再矩阵乘法 | 完整核，展开目标约 64 MiB / 块，单行超出时以一行为下限 |
| `svd3` | 三项 SVD 分离滤波求和 | 完整核 |
| `transpose` | 水平卷积、物化转置、再水平卷积 | 原秩 1 拟合核 |
| `cascade` | 两个三点对称核级联 | 原秩 1 拟合核 |
| `sliding-dft` | 五样本滑动频谱，定期重建 | 原秩 1 拟合核 |
| `boxes` | 三次 3 tap 均值核，保留级联外部 halo | 改变核 |
| `extended-boxes` | 三次带分数端点的盒式核，单方向总方差为 2.25 | 改变核 |
| `binomial` | `[1,4,6,4,1]/16` | 改变核；系数适合整数移位 |
| `am-recursive` | 四组一阶 causal/anticausal 递归 | 改变核 |
| `deriche` | 经典二阶 causal 与 anticausal 相加 | 改变核 |
| `vyv` | Young–van Vliet 三阶多项式系数版本 | 改变核 |
| `downsample` | 缩小、滤波、双线性恢复尺寸 | 改变核与采样过程 |

后三类 IIR 当前固定 sigma=1.5，在各维补 64 个零后做递归，再裁回原尺寸。边界、稳定性和质量测试针对这个参数；代码没有提供任意 sigma 的误差承诺。盒式级联先扩大计算域，在最后一次滤波后统一裁剪。

改变核的方法必须加 `--allow-approx`，并输出相对固定 5×5 核的误差。本次测试中，这些方法的最大误差为 19–75 灰度级，详见 [alternatives-correctness.json](results/alternatives-correctness.json)。它们适合大半径或允许改变效果的任务，当前作业不能直接用它们代替原核。[S1][S12][S13]

### 6.2 GPU

[gpu.py](gpu.py) 通过 CuPy RawModule 编译 [cuda_kernels.cu](cuda_kernels.cu)。输入上传后驻留设备，提供 direct、separable、fused 三种模式。fused 使用 32 个字节位置×8 行的块，加载带六字节、两行 halo 的共享内存，内部完成水平和垂直两段。部分边缘块的所有线程都经过同步屏障，避免条件提前返回导致死锁。

计时分开报告 JIT/分配、H2D、CUDA event 核时间和 D2H。输入 staging 使用 pinned memory；输出与 CPU 完整核参考比较，误差大于 1 时拒绝写文件。`--self-test` 覆盖 24 个小图方法组合，供真实 CUDA 机器执行。当前会话没有运行该测试，因此没有 GPU 加速比。[S14][S15]

线性纹理采样可以把相邻两个权重合并为一次采样。它要求纹理坐标、半像素偏移、线性采样精度和边界模式都匹配。此仓库没有图形渲染接口，因此本轮将它记录为 GPU 渲染场景扩展，未把 GLSL 片段当成已经集成的 BMP 程序。Kawase 和 mipmap blur 也归入改变滤波效果的渲染分支。[S16][S17]

## 7. 工程实现清单

| 范围 | 必须落实的条件 | 本次实现或复现方式 |
|---|---|---|
| 输入格式 | 24bpp、BI_RGB、正宽、非零有效高度、DIB 和像素偏移关系 | `bmp_safe.h`，实际文件长度检查 |
| 溢出 | 先扩大到 uint64/size_t 再算行长和文件大小 | 输入限制输出小于 4 GiB，FFT 单工作区上限 |
| 文件行序 | 高度符号保留；仅对垂直对称核允许免翻转 | 测试两种行序，三通道非对称值 |
| padding | 行尾不参与卷积 | 输出 padding 由各输出条带负责清零 |
| alias | 输入输出分离，避免边计算边覆盖源数据 | inode 检查；API 禁止原地使用 |
| 边界 | 宽 1–4 无内部区，宽 5–7 常暴露右边越界 | 精确分配和文件级测试 |
| SIMD | 不能读取下一行或文件头来凑满向量 | AVX2 每次读 8 字节，AVX-512 读 16 字节，剩余标量处理 |
| 指令集 | 通用构建检查 CPU 支持；native 构建需目标机重编译 | `--list` 与函数级 target |
| 数值 | FMA、结合次序、系数量化、输出舍入分别记录 | generic 禁止 FMA 收缩；native 单独验证 |
| 精确系数 | 不把十进制乘积装进 uint32 | uint64 累加，精确整数参考 |
| 并行分区 | 每个输出字节仅有一个写者，halo 只读 | 根据实际线程数分条带 |
| OpenMP | 运行时实际线程数可能小于请求 | 实际 team 内划分与输出 `threads` |
| Pthread | 创建失败、计算失败和条件变量唤醒都要处理 | 代际计数、谓词等待、失败回收、状态返回 |
| 工作集 | 完整行缓冲或 tile 缓冲是否适合缓存 | 可选 `--tile`，64 字节行槽对齐 |
| 首次触页 | 零填充或预触页会把成本挪到 setup | `--prefault` 单独记录，setup 纳入 wall |
| 文件映射 | mmap 的按需缺页属于访问成本 | 默认不把映射成功等同于数据已读入 |
| 写盘口径 | munmap/close 不等同等待存储完成 | `--sync` 执行 MS_SYNC 和 fsync；默认计页缓存提交 |
| 预分配 | `posix_fallocate` 返回错误码本身 | 显式转换 errno；支持关闭作对照 |
| NUMA/绑核 | 应按进程允许 CPU 集合选核 | `--pin` 读取 affinity；OpenMP 环境配置可单独比较 |
| GPU | 传输、JIT、resident kernel 分开 | 单独时间字段，设备同步和错误传播 |
| 变换卷积 | 零扩展、核中心、归一化、丢弃区、裁剪 | 单像素与脉冲测试、独立像素参考 |
| 级联/递归 | 中间外部支持域会影响边缘 | 扩域后执行，再统一裁剪 |
| 批处理 | 线程和设备复用收益需独立报告 | Pthread 池重复处理固定输入，CUDA 输入驻留 |
| 自动调参 | 先检查输出，再排名；不静默改默认 | `autotune.py` 写带环境和样本的 JSON |

进一步工程调优可在同一算法下对照 `OMP_PLACES=cores`、`OMP_PROC_BIND=close/spread`、物理核与 SMT、LTO/PGO、缓存大小、HugeTLB/THP、软件预取和非临时存储。这些属于调度或编译变体。历史仓库已有部分负结果，本轮没有把每种参数组合当作独立新算法，也没有在未测得收益时启用全局默认。[S2][S18][S19][S20]

## 8. 验证结果与边界

| 检查 | 结果 | 证据 |
|---|---|---|
| 通用构建，FMA contraction off | 3,411 案例通过 | [correctness-portable.json](results/correctness-portable.json) |
| native 构建，FMA contraction fast | 3,411 案例通过 | [correctness-native.json](results/correctness-native.json) |
| AddressSanitizer + UndefinedBehaviorSanitizer | 882 个文件级案例通过 | [correctness-sanitized.json](results/correctness-sanitized.json) |
| 精确长度堆分配，ASan+UBSan | 265 种宽度通过所有可用方法，逐行条带 | [buffers-sanitized.txt](results/buffers-sanitized.txt) |
| NumPy/SciPy 方法 | 309 案例通过；近似方法同时记录误差 | [alternatives-correctness.json](results/alternatives-correctness.json) |
| 分配与线程失败注入 | 51 案例通过，非零退出且无死锁 | [failure-injection.json](results/failure-injection.json) |
| 原程序官方素材回归 | 3 张素材×2 个后端，与改动前 BMP 逐字节一致；两版水平行 ASan 通过 | [legacy-regression.json](results/legacy-regression.json) |
| GCC 静态分析与人工复核 | 1 项不可达路径误报已核对 | [static-review.md](results/static-review.md) |
| GPU | 源码与自测入口完成，硬件执行待验证 | `python3 gpu.py --self-test` |

CPU 测试包含随机、全黑、全白、棋盘和单通道脉冲。检查上下行序、扩展偏移、非零 padding、非整向量宽度、不同 tile、1/7 线程、线程数超过高度、重复运行，以及两个并行后端输出一致。畸形输入覆盖零宽、高度 INT_MIN、尺寸溢出、错误 bpp/planes/compression、非法 DIB/offset 和像素截断。软硬链接指向输入时程序失败，输入哈希不变。

本环境的 LeakSanitizer 因运行时跟踪限制报错，ASan/UBSan 使用 `ASAN_OPTIONS=detect_leaks=0` 完成。因此这里记录的是地址与未定义行为检测通过，泄漏扫描没有完成。无该限制的目标机器可直接 `make sanitize` 使用默认泄漏检查。

教师验卷器依赖的 C++ OpenCV 环境在此机器没有安装。现有教师素材保持原样，官方大图由独立整数核对照；没有伪造教师验卷器运行记录。

## 9. 实测性能

性能数据由 [benchmark.py](benchmark.py) 生成。所有配置在每轮随机打乱顺序，每次输出均做误差检查。保存原始运行值、中位数、最小值、最大值和输出 SHA-256。使用中位数比较，不把不同窗口中的最低值拼成加速比。

本次机器为 AMD EPYC 9V74 虚拟环境，GCC 14.2，允许 9 个逻辑 CPU。它与历史 i9-13900K / GCC 11.4 的记录分开。新内核使用 `-O3 -march=native -ffp-contract=fast`；旧程序按原 Makefile 参数以 GCC 14.2 重新编译保存，基线源码 SHA 为本文开头的提交。

计时定义：

- `median_ms` 是程序本次重复执行中的计算耗时中位数，包含该方法的工作缓冲分配，Pthread 线程池建立计入 setup。
- `wall_ms` 包括读取准备、输出创建/预分配、计算、线程回收、输出提交、munmap 和输出 close，默认不等待持久化；不含进程启动、最终 JSON 打印及尾部的输入 fd 关闭/缓冲释放。
- `process_ms` 是外层实际运行进程的墙钟，包含进程启动和输出捕获。
- `--warmup` 与 `--repeat` 都重复读取同一原输入，没有把已经模糊后的输出继续当输入。
- 原程序 `legacy_total_ms` 不含文本导出和 munmap；`legacy_full_ms` 包含这两项。跨程序优先看包含启动的 `process_ms`，同时说明旧程序还承担文本导出。

### 9.1 正式图 4096×2304，4 线程

下表为完整处理口径的中位数，单位 ms。原程序使用 `legacy_full_ms`，包含额外的 kernel/matrix 文本导出；实验程序使用 `wall_ms`。两个窗口都独立随机交错运行。

| 后端与方法 | 第一窗口，5 轮 | 确认窗口，7 轮 |
|---|---:|---:|
| pthread 原版 | 12.594 | 12.782 |
| pthread ring | 10.094 | 12.636 |
| pthread symmetric | 11.612 | 12.351 |
| pthread vfirst | 9.434 | 14.175 |
| pthread avx2 | 10.337 | 12.018 |
| pthread avx512 | 9.669 | 12.180 |
| omp 原版 | 13.581 | 13.432 |
| omp ring | 10.642 | 12.181 |
| omp symmetric | 9.832 | 11.838 |
| omp vfirst | 9.529 | 12.581 |
| omp avx2 | 10.654 | 13.059 |
| omp avx512 | 11.340 | 12.529 |

第一窗口中垂直优先和部分 SIMD 配置有较好结果。第二窗口出现明显尾部延迟，部分方法排名反转。当前证据支持继续在目标机复测，尚不足以选出跨窗口稳定优胜者，因此没有修改原程序的计算默认值。例子是 Pthread `vfirst` 从 9.434 ms 变为 14.175 ms；OpenMP `avx2` 的确认窗口最大完整处理时间为 39.780 ms。所有慢样本保留在原始数据中。

确认窗口原始数据见 [benchmark-confirmation.json](results/benchmark-confirmation.json) 和 [CSV](results/benchmark-confirmation.csv)。其中使用的是最终边界检查版本；前一窗口之后增加了极大宽度的索引安全余量和输出文件非阻塞打开，数值内核没有变化。各记录包含对应源文件 SHA-256。

### 9.2 小图及自动调参

512×384 随机图对全部 17 种 CPU 方法完成了 5 轮、1/4 线程测试。FFT 和 overlap-save 对固定小核的成本明显高于直接分离方法，见完整 CSV。自动调参另对算法、线程数与五种 tile 大小进行 3 轮检查，结果保存在 [autotune-small.json](results/autotune-small.json)，只对该次运行环境成立。


小图完整方法原始数据为 [benchmark-small.json](results/benchmark-small.json) 和 [汇总 CSV](results/benchmark-small.csv)。正式素材候选方法原始数据为 [benchmark-timg.json](results/benchmark-timg.json) 和 [汇总 CSV](results/benchmark-timg.csv)。FFT 的自写 radix-2 实现用于验证频域方法与空间/边界成本；需要评估生产级 FFT 性能时，同时运行 SciPy FFT/overlap-add，对照 FFTW/MKL 等库。

## 10. 复现和使用

在仓库根目录执行：

```bash
# 历史两个程序。原 Makefile 默认 gcc-11；本机无该编译器时显式覆盖。
make -C 作业/代码 CC=gcc

# 通用 CPU 构建，不把 AVX 指令扩散到普通函数。
make -C research
research/conv_lab_pthread --list

# 使用目标机器本地指令集，重新构建并验证。
make -C research native
python3 research/test_methods.py
python3 research/test_failures.py
OPENBLAS_NUM_THREADS=1 python3 research/test_alternatives.py

# 原输入输出接口；线程与算法显式指定。
research/conv_lab_pthread input.bmp output.bmp --method avx2 --threads 4
research/conv_lab_omp input.bmp output.bmp --method symmetric --threads 4

# 单次文件处理，等待输出文件同步。
research/conv_lab_pthread input.bmp output.bmp --method avx2 --threads 4 --sync

# resident / warm-cache 计算测试，线程池复用。
research/conv_lab_pthread input.bmp output.bmp --method avx2 --threads 4 --warmup 2 --repeat 20

# 缓存/文件策略对照，一次只改变一个参数。
research/conv_lab_pthread input.bmp output.bmp --method avx2 --threads 4 --io buffered
research/conv_lab_pthread input.bmp output.bmp --method avx2 --threads 4 --prefault
research/conv_lab_pthread input.bmp output.bmp --method avx2 --threads 4 --pin

# 随机交错基准；完整输出在 JSON，汇总 CSV 自动同名生成。
python3 research/benchmark.py input.bmp --output research/results/local.json --rounds 5 --threads 1,2,4 --methods ring,symmetric,tiled,vfirst,avx2 --legacy

# 在固定目标机上选择算法、线程数和 tile，先验输出再排名。
python3 research/autotune.py input.bmp --output research/results/tuning.json --rounds 5

# 方法扩展。
python3 research/alternatives.py input.bmp output.bmp --method dct
python3 research/alternatives.py input.bmp output.bmp --method deriche --allow-approx

# 在已装兼容 CUDA/CuPy 的机器上运行，缺少依赖时直接报错。
python3 research/gpu.py --self-test
python3 research/gpu.py input.bmp output.bmp --method fused --repeat 20
```

ASan/UBSan：

```bash
make -C research sanitize
# 当前容器的 LeakSanitizer 限制下采用：
ASAN_OPTIONS=detect_leaks=0 make -C research sanitize
make -C research -B test_buffers CFLAGS='-O1 -g -std=c11 -Wall -Wextra -ffp-contract=off -fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie -no-pie'
ASAN_OPTIONS=detect_leaks=0 research/test_buffers
# 最后切回性能构建，避免测到 sanitizer 版本。
make -C research native
```

Python 对照环境本次为 NumPy 2.3.5、SciPy 1.17.0，可用 `python3 -m pip install -r research/requirements.txt` 安装记录的版本。`test_methods.py` 只使用标准库，性能脚本额外需要 NumPy。BLAS 线程数对 im2col/SVD 有影响，应记录 `OPENBLAS_NUM_THREADS`，避免与外层并行叠加。

## 11. 资料与代码阅读索引

以下均为原作者论文、官方文档或作者工程文章。网页检索日期为 2026-11-24。各来源的性能数字属于其原实验，本文没有把那些数字作为本仓库加速结果。

| 编号 | 资料 | 对本项目的用途 |
|---|---|---|
| S1 | [Getreuer, A Survey of Gaussian Convolution Algorithms, IPOL 2013](https://www.ipol.im/pub/art/2013/87/) | 系统整理 FIR、DFT/DCT、盒式、递归滤波及边界处理 |
| S2 | [Halide lesson 8, Scheduling multi-stage pipelines](https://halide-lang.org/docs/tutorial/lesson_08_scheduling_2.html) | fusion、compute/store 层级、sliding window 和 storage folding |
| S3 | [Ragan-Kelley et al., Halide, PLDI 2013](https://people.csail.mit.edu/jrk/halide-pldi13.pdf) | 算法与调度分离，局部性、并行与重算的权衡 |
| S4 | [Fabian Giesen, Fast blurs 1](https://fgiesen.wordpress.com/2012/07/30/fast-blurs-1/) | 滑动均值和连续窗口更新 |
| S5 | [Fabian Giesen, Fast blurs 2](https://fgiesen.wordpress.com/2012/08/01/fast-blurs-2/) | 分数端点盒式核与增量更新 |
| S6 | [Lavin & Gray, Fast Algorithms for Convolutional Neural Networks](https://arxiv.org/abs/1509.09308) | Winograd 小块变换，区别于大核 FFT |
| S7 | [Arm, Winograd convolutions meet integer quantization](https://developer.arm.com/community/arm-research/b/articles/posts/even-faster-convolutions-winograd-convolutions-meet-integer-quantization-and-architecture-search) | 变换规模、量化与数值范围 |
| S8 | [Convolutional kernel function algebra, 2022](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2022.921454/full) | 固定核的代数变换和公共表达式 |
| S9 | [FFTW, What FFTW Really Computes](https://www.fftw.org/doc/What-FFTW-Really-Computes.html) | DFT 定义与归一化约定 |
| S10 | [SciPy oaconvolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.oaconvolve.html) | overlap-add 工程对照和适用规模 |
| S11 | [Sugimoto & Kamata, Sliding DCT/DST-5, 2015](https://www.jstage.jst.go.jp/article/mta/3/1/3_12/_article/-char/en) | 恒定核宽成本的滑动频域方法，启发本项目的五点滑动 DFT 对照 |
| S12 | [Rybakova et al., Fast Gaussian Filter Approximations Comparison on SIMD Computing Platforms, 2024](https://www.mdpi.com/2076-3417/14/11/4664) | 量化与 SIMD 的联合比较，Stack/Bell/recursive 等近似家族 |
| S13 | [Young & van Vliet, Recursive implementation of the Gaussian filter, 1995](https://www.sciencedirect.com/science/article/pii/016516849500020E) | 三阶递归高斯的基础方法 |
| S14 | [NVIDIA, Image Convolution with CUDA, 2007](https://developer.download.nvidia.com/assets/cuda/files/convolutionSeparable.pdf) | halo、共享内存、分离卷积和合并访问 |
| S15 | [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | 数据传输、pinned memory、并行与计时 |
| S16 | [RasterGrid, Efficient Gaussian blur with linear sampling](https://www.rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/) | 双线性纹理采样合并相邻 tap |
| S17 | [Intel, Investigation of Fast Real-Time GPU-based Image Blur Algorithms](https://www.intel.com/content/www/us/en/developer/articles/technical/an-investigation-of-fast-real-time-gpu-based-image-blur-algorithms.html) | 分离、移动均值、Kawase 的工程比较 |
| S18 | [GCC 14.2 Optimize Options](https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/Optimize-Options.html) | FMA contraction、LTO、PGO 与编译可比性 |
| S19 | [OpenMP 6.0 OMP_PROC_BIND](https://www.openmp.org/spec-html/6.0/openmpsu15.html) | close/spread 线程亲和语义 |
| S20 | [Linux msync(2)](https://man7.org/linux/man-pages/man2/msync.2.html) | 区分脏页提交与等待同步 |
| S21 | [OpenCV image filtering](https://legacy-docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html) | filter2D 的相关运算、零边界与大核 DFT 分支 |
| S22 | [Intel IPP Separable Filters](https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2026-0/separable-filters.html) | 商业库中的行列滤波与 pipeline 工作区接口 |
| S23 | [NVIDIA VPI Recursive Gaussian Filter](https://docs.nvidia.com/vpi/algo_recursive_gaussian_filter.html) | 大 sigma 下的 IIR 应用条件 |
| S24 | [Arm, Optimizing image processing using AArch64 Neon](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/optimizing-image-processing-using-aarch64-neon) | 自动向量化与手写 SIMD 的比较方式；ARM 后续移植参考 |
| S25 | [Linux mmap(2)](https://www.man7.org/linux/man-pages/man2/mmap.2.html) | 映射、按需缺页、错误与释放 |
| S26 | [Linux madvise(2)](https://man7.org/linux/man-pages/man2/madvise.2.html) | 顺序访问与大页建议的语义 |

## 12. 采用顺序

先在用户实际评分机器上运行 `test_methods.py` 和随机交错基准，再比较 `avx2`、`symmetric`、`vfirst` 与当前默认方法。AVX-512 仅在硬件支持时参与。使用 `autotune.py` 检查小图是否应减少线程，以及大图是否值得分块。

如果输入来自连续帧或已有 GPU 流水线，再运行 CUDA 自测和端到端测试。需要扩大半径时，重新评估 FFT、DCT、积分图和递归方法。每次采用新默认值，都把图像、编译参数、实际线程数、误差和完整时间一起保留。
