# AGENTS.md

## 项目目标

这是一个面向《强化学习的数学原理》的纯原生强化学习教学框架。用户希望它既能运行书中 GridWorld 实验，又能把每一步数学计算、策略概率、状态值、算法代码执行位置实时可视化。

## 用户正向要求汇总

- `algorithms/` 文件夹只存用户写好的核心强化学习算法，尽量对应书中伪代码的几行核心计算。
- 主函数中只通过更改算法函数和参数配置来切换算法；训练过程自动调用所选核心算法。
- 框架层必须提供算法所需的通用参数包，至少覆盖书中常见算法：DP、MC、TD、SARSA、Q-learning、Expected SARSA、n-step、eligibility trace、Dyna 等表格 RL 参数。
- UI、训练引擎、TensorBoard、保存恢复逻辑都不应该读取算法私有变量；所有显示数据通过标准 `InfoDict` 或框架适配器生成。
- GridWorld 必须是纯手写环境，不依赖 Gym。
- 主训练框显示最纯的环境 Grid，可显示设定 reward 数值，不要把策略线画在主训练框中。
- 策略概率必须实时从算法返回的完整 `policy_probs` 读取，不能假设初始均等，也不能缓存错误旧值。
- 策略概率图使用每个格子中心的小十字绿线显示：四个方向分别代表动作概率，统一单位长度归一化，不能糊住格子。
- 当前运行状态、更新状态只用边框高亮，不改变策略线本身。
- 右侧值网格显示训练后的最优状态值 `V(s)=max_a Q(s,a)`，实时更新。
- 公式区显示当前更新公式、变量和具体数值计算过程。
- 算法指针必须自动解析真实核心算法函数源码，左侧指针逐行高亮正在运行的代码行；不要手写固定 1-5 句伪代码。
- 速度控制应支持无限接近 0 的正常运行速度，也能慢速观察每一步。
- TensorBoard 不需要在主 UI 中复刻；运行 `main.py` 时可以同时打开主界面和 TensorBoard 页面。
- TensorBoard 指标需要拆分 tag，例如 `rollout/reward`、`train/loss`、`train/td_error`、`train/q_value`。
- 同一算法重新运行时必须清理上次数据，包括 Q 表、策略、checkpoint、TensorBoard 日志和 UI 状态。
- 框架需要提供“重新运行”和“继续上次”两种模式。
- 训练后需要保存 checkpoint，关闭窗口时也应保存；继续训练时可恢复上次算法状态。
- 需要有硬件检测模块：当前表格 RL 默认使用 CPU；检测到 NVIDIA GPU 时提示未来深度 RL 可走 CUDA；Intel 核显主要用于显示层，不用于当前小规模表格算法加速。

## 当前架构约定

- `algorithms/*.py`: 只写核心算法函数，签名为 `fn(ctx, transition) -> RLStepResult`。
- `core/rl_parameters.py`: 保存通用参数、表格状态、transition 和算法函数协议。
- `core/table_agent.py`: 把核心算法函数包装成可训练、可视化、可保存的 Agent。
- `core/algorithm_trace.py`: 自动解析核心算法函数源码，供 UI 指针显示。
- `utils/session.py`: 管理 checkpoint、日志清理、重新运行和继续上次。
- `main.py`: 算法选择区只改 `ALGORITHM_NAME`、`CORE_ALGORITHM`、`ALGORITHM_CONFIG`。

## 开发注意

- 不要把 UI 逻辑写进 `algorithms/`。
- 不要让 UI 读取算法私有变量。
- 不要把策略概率画到主训练环境框里。
- 改算法时优先新增/修改 `algorithms/` 中的核心函数。
- 改参数时优先修改 `main.py` 的 `ALGORITHM_CONFIG`。
- 提交前至少运行 `python -m compileall -q .` 和一个无 UI 的算法 smoke test。
