# DSV4.1 Window U1 implementation log

## Scope

本阶段实现单 microbatch、eager 的 DeepSeek V4.1 Window AFD。ACLGraph、PD、DSpark
和 2/3/4 microbatch 流水不在本阶段开启；权重名称映射验证后置。

## 关键改动

1. `WindowAFDConnector` 采用参考 `afd-async` 的 A2F/F2A operator ABI：
   - `K = num_experts_per_tok` 的 routed IDs/scales 作为发送输入；
   - Window metadata、rank table 和 operator shape 使用 `K + 1`，其中逻辑 expert
     `R = n_routed_experts` 是隐式 shared slot；
   - rank 0 映射 shared slot，其余 FFN ranks 连续承载 routed expert 分片；
   - U1 的容量、active mask、`actual_token_num` 和 group list 类型转换与参考实现一致。
2. connector 顶层只保留 CPU 可导入的可选 CANN 依赖，真实初始化时才检查
   `cann_ops_transformer`/`torch_npu`。U1 明确拒绝多 microbatch、DBO/ubatching、PP/CP
   和非 eager 配置。
3. V4.1 FFN role 按 Window layout 拆分：
   - shared rank 使用 `AFDDeepseekV41SharedMoE`，仅包含 widened
     `DeepseekV41MLP(...shared_experts)`；
   - routed rank 使用 `AFDDeepseekV41RoutedMoE`，以 `n_shared_experts=None` 构造
     native routed ABI，并写入 global-to-local expert map；
   - Attention role 保留 gate shell 和 remote Window proxy。
4. 通用 attention-gate helper 在访问 routed `.experts` 前增加 shared-only 分支，
   shared MLP 与 routed MLP 通过统一 `AFDF2ATransferPayload` 回传。
5. V4.1 mHC continuation 仍由 Attention 保存 `pre_mix`、`post`、`comb` 等层间状态；
   F2A 返回的 `[tokens, hidden_size]` 输出进入 native `hc_post`，并继续传递 `ffn_pre`。

## 验证边界

- `compileall`、Ruff 和 `git diff --check` 已执行。
- 聚焦 CPU contract/package 测试已执行；当前环境没有 vLLM/torch/CANN，依赖这些组件
  的模型测试会跳过。
- 完整 `tests/unit` 的既有 `test_ascend_config_lifecycle.py` 受 Windows GBK 默认编码
  影响，无法读取 UTF-8 源文件；这不是本次 U1 代码断言失败。
- 必须在匹配的 CANN/NPU 环境补充单层 golden、A2F/F2A 闭环和真实 U1 eager 精度验证，
  之后才能宣称运行时功能完成。
