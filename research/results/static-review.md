# 最终静态检查记录

GCC 14.2，`-O0 -std=c11 -Wall -Wextra -fanalyzer` 分别检查 `kernels.c` 与 `conv_lab.c`。CLI 没有诊断；内核出现一项 prefix 空指针诊断。

人工核对该诊断的路径条件发生冲突：进入 `ring` 时分析器选择 `m != M_PREFIX`，所以 `prefix=NULL`；随后进入 `horizontal` 时又选择 `m == M_PREFIX`。两处 `m` 是同一按值传递、期间未修改的枚举参数。真实 M_PREFIX 路径已在调用前检查前缀分配是否成功。该项记录为路径合并造成的误报，不作为已发现的空指针缺陷。

相关动态证据包括 prefix 的独立数值比较、ASan/UBSan、精确缓冲区测试，以及 malloc/calloc/aligned_alloc 的失败注入。所有失败均以非零状态退出。

最终人工复核还覆盖了 SIMD 读取长度、每线程输出所有权、FFT 裁剪/归一化、Q15 的 64 位累加、inode 检查先于截断、OpenMP 按实际线程数分区、Pthread 创建失败后的回收，以及 GPU 边缘线程必须到达两个屏障。GPU 编译和设备运行尚待真实 CUDA 环境验证。
