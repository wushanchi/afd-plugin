# DeepSeek-V4.1 Flash AFD 适配状态与环境条件

本文记录 `afd-plugin` 在 `asyn-opt` 分支上的 DeepSeek-V4.1 Flash 适配边界。文档中的“已实现”只表示代码和静态契约已经落地；没有 Ascend NPU、HCCL、CANN 算子或真实权重运行证据的内容，统一标为 `Pending-NPU`。

## 代码基线

| 项目 | 版本/证据 |
| --- | --- |
| AFD 插件 | `asyn-opt`，基线 HEAD `4a587aa1871db68ed3f1ab33bc88746fc36894d6` |
| AFD 远端 | `https://github.com/wushanchi/afd-plugin.git`，目标分支 `asyn-opt` |
| vLLM-Ascend V4.1 参考 | 本地 checkout `f769db916b53446de5779bced0b7d3d38e8a7540` |
| cann-recipes-infer | GitCode master `93527538f1567a4cf42db58883231e4e00f8b213` |

GitCode 的 V4.1 代码提供原生 Vision、CSA/DSA、Engram、MoE、量化和 DSpark 路径，没有实现 AFD 的 Attention/FFN 角色、A2F/F2A 数据协议或 `afd-plugin` connector。其 V4 AFD 文档不能直接证明 V4.1 AFD 已经可用。

## 当前代码阶段

当前适配代码新增了 `deepseek_v41.py`，并将 V4.1 架构接入 AFD 注册、架构校验和 P2P/CAMP 的 `input_ids` 侧信道。第一阶段的实现范围是：

- 文本请求；
- P2P 或 CAMP AFD connector；
- eager、U1；
- Attention 侧保留原生 V4.1 attention/HC，FFN 侧保留原生 V4.1 MoE；
- 第 0 层通过现有 ID side-channel 传递 `input_ids`，让原生 hash routing 能够运行。

实现明确拒绝以下路径，避免把 V4 的协议误用到 V4.1：

- WindowAFDConnector、Attention-side gate、Graph/U2；
- MTP/DSpark；
- Engram transfer；
- 需要跨角色传递 `image_mask`、视觉路由状态或 Engram hash 的扩展协议。

Vision wrapper 目前只复用了原生图像编码和语言模型入口，跨 AFD 的图像元数据、视觉 bias 和图像请求精度仍为 `Pending-NPU`。

## 必须满足的运行条件

### 软件和硬件栈

V4.1 原生实现与当前 V4 AFD README 中的 `vLLM 0.26.0 + vLLM-Ascend 80d8c194f + Ascend 910C/A3` 基线不是同一套 ABI。GitCode V4.1 参考要求 Ascend 950/A5、ARM Linux、配套 CANN/`torch_npu`/TorchAir 镜像；具体 PyTorch 小版本必须以所选 vLLM-Ascend commit 和镜像锁定，不能只凭 README 的单处版本字符串配置。需要先完成：

1. vLLM、vLLM-Ascend、CANN、驱动固件、`torch_npu`、TorchAir 的成套版本确认；
2. `torch.ops._C_ascend.npu_hc_pre_v3`、`npu_hc_post`、DSA/CSA cache、MoE 和量化算子存在并通过最小 eager smoke test；
3. HCCL/Gloo 控制面端口、rank 映射、NPU 可见性和 HBM 余量确认；
4. 关闭 Engram/DSpark/Graph 选项，先完成文本 P2P U1 的单卡双角色验证。

当前 `afd-plugin/csrc/npu` 的自定义 A2E/E2A 构建脚本仍按 Ascend 910C 型号限制；它不能直接作为 Ascend 950/A5 的算子包。A5 适配必须提供对应 ACLNN/通信算子实现、编译目标和 ABI 验证，否则 Python wrapper 即使能够导入，也不能称为可运行的 NPU AFD。

### 模型和权重

需要完整的 V4.1 Flash 配置、tokenizer、文本/视觉权重和 checkpoint index。参考配置的核心结构是 40 层、384 个 routed experts、top-k=6、1 个 shared expert，并包含 KV/index source 层和 Engram 层。原始 Hybrid FP8-MXFP4 权重需要按照目标 CANN 算子要求转换为 Hybrid MXFP8-MXFP4；转换后的 scale、专家映射和 `gate_up_proj` 堆叠必须与 native `load_weights` 一致。

### AFD 协议和拓扑

P2P/U1 只传递 hidden states，并在第 0 层传递 `input_ids`。要达到 V4.1 Flash 的完整功能，还需要扩展协议并验证生命周期：

- `image_mask`/视觉 token 的位置和视觉 router bias；
- Engram hash、lookup mask、lookback token 和 stream/event；
- CSA2/Indexer 的 source-layer、candidate 和 top-k buffer 状态；
- DSpark target hidden states、draft layer 和 speculative step；
- Window 的 384 expert layout、type-2 group list、dynamic scale、shared expert 和 U2 slot reuse。

Window 当前还有容量条件：`micro_batch_num` 只能为 1/2、`max_num_batched_tokens <= 512`、top-k 不超过 16、shared expert 必须为 1；40 层和 384 experts 在现有全局 GMM group 上限下可能需要增加 FFN rank，必须按真实 DP/TP/EP 拓扑重新计算。

## 验证门禁

适配应按以下顺序推进，每一步都保存精度、通信和 profiler 证据：

1. CPU 静态导入、架构注册、权重所有权和 shape contract；
2. 单 NPU 原生 V4.1 eager 文本精度；
3. Attention/FFN 角色拆分后的本地等价性；
4. 双角色 HCCL/CAMP P2P U1；
5. Window U1，再做 U2/多流和 Graph replay；
6. DSpark/MTP、Vision/Engram、CP/EP 和量化；
7. 同硬件、同权重、同负载下的 TTFT、TPOT、吞吐和 HBM/profiler 对比。

在完成第 4 步以前，不应宣称“V4.1 AFD 已支持”；在完成第 6 步以前，不应宣称“与当前 DeepSeek-V4 AFD 功能等价”。
