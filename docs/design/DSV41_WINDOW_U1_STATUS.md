# DeepSeek V4.1 Window AFD U1 状态

## 检查结论

当前 `dsv41-afd` 分支包含 DeepSeek V4.1 Window U1 eager 的模型注册、Window connector、FFN connector-driven loop、mHC continuation contract、shared/routed FFN role 拆分和配置校验接线。CPU 侧契约与静态检查可以通过；真实 NPU/CANN 闭环仍需硬件验证，因此本阶段不宣称硬件验收完成。

## 已接入的路径

- `DeepseekV41ForCausalLM` 映射到 NPU V4.1 AFD wrapper。
- `WindowAFDConnector` 注册到 connector factory，并限制为单 microbatch、eager、无 DBO/ubatching；CANN 依赖在初始化阶段检查，CPU 导入不再被顶层 NPU 模块阻塞。
- Attention 侧保留 V4.1 `hc_pre`/`hc_post` 调用，在 FFN 边界保存 continuation。
- FFN worker 支持 connector-driven Window receive、FFN compute 和 send。
- Window connector 按参考 ABI 计算 Window 容量，保留 quant mode 0/2/4、active mask、actual token count/type-2 group list 转换和 K routed top-k/K+1 operator metadata 约束。
- FFN rank 0 构造 shared-only `DeepseekV41MLP`；其余 FFN rank 构造 `n_shared_experts=None` 的 routed-only MoE，并安装连续的 global-to-local expert map。
- 通用 FFN helper 在访问 routed expert ABI 前处理 shared-only rank，shared 输出通过同一 F2A payload 返回。
- 新增 rank mapping、expert layout 和 continuation registry 的 CPU 可测试 contract。

## 当前阻塞项

1. Native V4.1 的 mHC 输入保留 `[tokens, hc_mult, hidden_size]`，但 `hc_pre` 输出给 MoE 的 activation 是 `[tokens, hidden_size]`。当前实现尚无 native 与 Window 的数值 golden，无法证明 `pre_mix -> ffn_pre -> hc_post` continuation 语义一致。
2. 尚无 NPU 单层 golden、A2F/F2A 闭环、U1 eager 精度或吞吐证据。
3. 权重名称映射尚按项目约定后置验证；本阶段仅保证 role-local module namespace 和 dispatch seam 已建立。

## 已确认的 shared expert 方案

DeepSeek V4 Window 的 shared expert 机制可以作为 V4.1 U1 的直接设计基线，
但需要移植模型 role construction，不能只修改 `expand_x_shared` 参数：

- 设 `K = num_experts_per_tok`、`S = n_shared_experts`、`R = n_routed_experts`。
  `selected_expert_num = K + S` 只用于 Window buffer、metadata 和 operator
  shape；A2F 的 `expert_ids` 和 `expert_scales` 仍只有 K 个 routed top-k 项。
- rank table 增加隐式 shared logical expert `R`，将它映射到专用 shared FFN
  rank（通常 FFN role rank 0）；其余 FFN ranks 只承载本地 routed expert 分片。
- shared rank 构造只包含 `DeepseekV41MLP` shared expert 的模块，不创建
  `experts`；routed rank 构造 `shared_experts=None` 的 local routed MoE。
- `npu_ffn_worker_batching` 返回的 hidden states 和 group list 在 shared rank
  直接执行 shared MLP；routed rank 执行 grouped routed MLP。两类输出都通过
  同一 F2A path 返回 Attention，由 Window operator 完成 token 合并。
- V4.1 的 native mHC seam 保持不变：Attention 侧保存 `post/comb` 等
  continuation，F2A 返回的 `[tokens, hidden_size]` routed/shared 结果进入
  native `hc_post`，并由 model 继续传递 `ffn_pre`。

因此，当前代码的 shared expert 缺口是 role module 和 Window dispatch ABI
缺失，而不是需要把 shared activation 作为额外 top-k ID 发送。权重名称映射
按当前要求暂不作为 U1 入口阻塞项，待 role module 稳定后单独验证。

## 已执行验证

在无 NPU/CANN 的开发环境中执行：

```text
uv run pytest tests/unit/model_executor/models/test_deepseek_v41_window_contract.py \
  tests/unit/compat/npu/test_dsv4_async_validation.py \
  tests/unit/package/test_package.py -q
聚焦 U1/包测试通过；无 vLLM/torch 的环境会对依赖 vLLM 的模型测试跳过。

uv run ruff check <U1 changed files>
All checks passed

uv run python -m compileall -q afd_plugin tests
git diff --check
passed

完整 `tests/unit` 仍有既有环境编码失败：
`tests/unit/compat/npu/test_ascend_config_lifecycle.py` 使用系统 GBK 读取
UTF-8 源文件，触发 `UnicodeDecodeError`；该失败与本次 U1 改动无关。
```

这些检查只证明 Python 接线、静态 contract 和代码格式正确，不能替代 NPU 运行验证。

## 下一步

先修正 mHC continuation 对照、shared expert role 路由和 Window operator ABI，再补单层 golden 与真实 U1 eager E2E；权重名称映射按当前要求后置验证。完成这些证据后才能把 WIP 提交提升为功能完成。
