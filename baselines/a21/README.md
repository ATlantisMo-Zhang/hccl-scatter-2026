# Scatter 集合通信算子

## 1. 项目介绍

```
├── CMakeLists.txt                  # 顶层 CMake 配置
├── build.sh                        # 构建脚本
├── .clang-format                   # 代码风格配置
├── include/                        # 头文件目录
│   ├── hccl.h                      # 集合通信算子头文件
│   ├── common.h                    # 通用数据结构定义
│   ├── custom.h                    # ★ 选手编写：自定义数据结构定义
│   ├── log.h                       # 日志宏定义
│   └── binary_stream.h             # 序列化类定义
├── op_host/                        # Host侧代码目录
│   ├── scatter.cc                  # ★ 选手编写：Host侧资源申请逻辑
│   └── exec_op.cc                  # ★ 选手编写：通信算法编排逻辑
└── op_kernel_ccu/                  # CCU侧代码目录
    └── ccu_kernel.cc               # ★ 选手编写：通信算法编排逻辑
```

> [!NOTE] 注意：
> 算子工程中已提前预制好固有逻辑，选手仅允许修改 `custom.h`、`scatter.cc`、`exec_op.h`、`exec_op.cc`、`ccu_kernel.h`、`ccu_kernel.cc` 共 6 个文件内容。

## 2. 编译运行

### 2.1 安装 CANN-Toolkit 包

请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260701000328953/)，根据产品型号和环境架构下载对应软件包。安装命令如下，更多指导参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》。

```bash
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_9.1.0_linux-${arch}.run
# 安装命令
./Ascend-cann-toolkit_9.1.0_linux-${arch}.run --full --install-path=${install_path}
```

### 2.2 环境变量配置

按需选择合适的命令使环境变量生效。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

### 2.3 编译算子工程

```bash
bash build.sh

# 编译 Debug 版本，便于断点调试
bash build.sh --debug
```

## 3. 代码格式

选手代码需符合 [.clang-format](.clang-format) 文件中的代码风格规范，可通过下列命令一键修改：

```bash
bash build.sh --format
```
