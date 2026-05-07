# AGENTS.md

## Latest Algorithm Adapter Contract

- UI 点击“重新运行”必须对当前算法开启真正的新 run：关闭 writer 和 TensorBoard、清理 checkpoint/session 状态、清理旧日志，并让新的 TensorBoard launcher/logger 指向本次运行的干净日志目录；如果 Windows 锁住旧 event 文件，新的 TensorBoard 也不能继续读取那些旧 event。
- 重新运行后如果 TensorBoard 之前已经打开，需要用新 run 的唯一 URL 重新打开/刷新页面，避免浏览器停留在旧前端缓存或旧服务页面上。
- Algorithm 8.2 `sarsa_fa` 适配器必须保证使用线性 action-value 权重向量；即使 `main.py` 里旧配置仍是 `value_function="table"`，用户只改 `CORE_ALGORITHM = sarsa_fa` 也不能把二维 Q 表传入 `q_hat/grad_q_hat` 导致线程崩溃。
- `sarsa_fa(w, s0, pi, q_hat, grad_q_hat, step, ...)` 必须走 episode 模式适配：框架注入这些函数并执行原始书中训练循环，通过源码插桩把每次 `w` 更新转换为标准 `InfoDict`；不能再把它伪装成外部 transition 单步算法。
- 全书 `Algorithm x.x` 适配优先按算法族维护：以 `env` 开头的整段训练函数进入通用 episode 适配器；常用注入名包括 `states/actions/model/transition/reward/policy_probs/sample_from_policy/softmax/q/grad_q/v_hat/grad_v_hat/grad_log_policy`；常用配置名包括 `iterations/sweeps/n_steps/planning_steps/max_steps`。只有出现这些符号覆盖不了的新算法族时，才扩展 `core/algorithm_adapters.py` 的签名匹配。
- 除 `algorithms/` 外的项目代码改动前，仍必须先向用户汇报准备改哪些模块和它们的具体功用。
- `algorithms/__init__.py` 对已知算法保留静态导出，同时继续动态扫描新算法；这样 `from algorithms import *` 既能服务 `CORE_ALGORITHM = QAC` 这类写法，也能让编辑器静态分析认出已有算法名。
- TensorBoard 自动弹出应在服务端 HTTP 就绪后再打开浏览器；Windows 下优先使用稳定的 `127.0.0.1` 地址，并在 `webbrowser.open` 静默失败时用系统 URL 打开方式兜底。
- 重新运行同一算法前需要停止 TensorBoard 服务、关闭 TensorBoard writer、清理当前算法的 `runs/<algorithm>` 和 checkpoint，再重建日志目录；否则 TensorBoard 可能继续显示旧 event 缓存或被旧进程锁住的日志。

## 项目目标

这是一个面向《强化学习的数学原理》的纯原生强化学习教学框架。用户希望它既能运行书中 GridWorld 实验，又能把每一步数学计算、策略概率、状态值、算法代码执行位置实时可视化。

## 用户正向要求汇总

- `algorithms/` 文件夹只存用户写好的核心强化学习算法，尽量对应书中伪代码/用户笔记中的数学过程。
- 框架目标是“框架不变、算法可换”：除 `algorithms/` 中新增/替换算法函数，以及 `main.py` 中切换所选算法函数与参数配置外，`engine/ui/session/table_agent` 等框架层不应为了迎合某个具体算法临时改写。
- 对于书中的控制算法，`greedy/epsilon-greedy` 这类策略改进规则属于算法本体；不能把它偷偷下沉成框架统一默认行为，更不能让适配器把 Algorithm 8.2 换成 softmax 一类不匹配的控制方式。
- 用户希望最终写算法时可以采用书中风格的纯函数，而不是框架私有协议。例如 Algorithm 8.2 应允许写成：

```python
import numpy as np

def sarsa_fa(w, s0, pi, q_hat, grad_q_hat, step,
             alpha=0.01, gamma=0.9, episodes=500, max_steps=100):
    for _ in range(episodes):
        s = s0
        a = pi(s, w)

        for _ in range(max_steps):
            s_next, r, done = step(s, a)

            if done:
                delta = r - q_hat(s, a, w)
                w = w + alpha * delta * grad_q_hat(s, a, w)
                break

            a_next = pi(s_next, w)

            delta = r + gamma * q_hat(s_next, a_next, w) - q_hat(s, a, w)
            w = w + alpha * delta * grad_q_hat(s, a, w)

            s = s_next
            a = a_next

    return w
```

- 框架必须负责把上述纯算法函数适配成可实时训练、可视化、可保存、可写 TensorBoard 的过程；不能要求用户在 `algorithms/` 里手写 UI 字段、`InfoDict`、TensorBoard 指标或框架状态对象。
- 对于 `sarsa_fa` 这类整段训练函数，框架适配器应提供 `pi`、`q_hat`、`grad_q_hat`、`step` 等参数，并把算法内部每次权重更新转换成标准 `InfoDict`，从而驱动公式区、策略图、值函数图、算法指针和指标记录。
- 主函数中只通过更改算法函数和参数配置来切换算法；训练过程自动调用所选核心算法。
- `main.py` 的算法选择区应优先表达为“选择书中算法函数 + 参数配置”，例如 `CORE_ALGORITHM = sarsa_fa`；框架应在 `core/algorithm_adapters.py` 中根据算法函数签名自动识别/适配，而不是要求用户手动挑选适配器。
- `main.py` 中的 `CORE_ALGORITHM = ...` 应直接填写核心算法函数名称本身；该名称应由 `algorithms` 包导出为可调用函数，不能让同名模块对象混入并导致适配阶段拿到 module 而不是 function。
- 为了让书中算法尽量只靠 `CORE_ALGORITHM = 函数名` 和 `ALGORITHM_CONFIG` 就能接入，框架应预定义一组常见书中符号命名，并在 README 中明确区分“由适配器按签名注入的对象/函数”和“由 `ALGORITHM_CONFIG` 提供的标量参数”，便于后续按书中符号直接写算法。
- 框架的最终适配目标是覆盖《强化学习的数学原理》中所有主线 `Algorithm x.x` 盒内算法；用户新增核心算法函数后，只要函数签名落在 README 约定的算法族内，`main.py` 中只改 `CORE_ALGORITHM` 和 `ALGORITHM_CONFIG` 即可运行，UI、TensorBoard、checkpoint、代码指针和公式区都应保持正常。
- 后续新增书中算法时，核心函数写法应尽量贴近 `algorithms/QAC.py` 这种“算法自己按书中流程循环 episode / step”的风格；框架应优先适配这种整段训练函数，而不是要求用户改写成框架私有协议。
- 框架层参数应保持当前算法所需的最小集合。当前阶段优先覆盖 Algorithm 8.2 需要的 `alpha/gamma/epsilon`、策略类型、softmax 温度（仅在选择 softmax 策略用于连续概率可视化时使用）、函数近似类型、特征阶数、权重初始化和随机种子；其他算法参数等用户明确切换到相应算法时再增加。
- 当前阶段若目标是稳定演示 Algorithm 8.2 的学习路径，默认应优先使用 `value_function="linear" + feature_kind="one_hot"` 这类精确线性特征基线；Fourier 特征只在用户明确要观察函数逼近泛化效应时再切换，避免默认配置把策略学歪到局部循环或大范围错误泛化。
- UI、训练引擎、TensorBoard、保存恢复逻辑都不应该读取算法私有变量；所有显示数据通过标准 `InfoDict` 或框架适配器生成。
- GridWorld 必须是纯手写环境，不依赖 Gym。
- 主训练框显示最纯的环境 Grid，可显示设定 reward 数值，不要把策略线画在主训练框中。
- 策略概率必须实时从算法返回的完整 `policy_probs` 读取，不能假设初始均等，也不能缓存错误旧值。
- 策略概率图使用每个格子中心的小十字/箭头绿线显示：四个方向分别代表动作概率，统一单位长度归一化，不能糊住格子。
- 策略图中某方向概率增大时，该方向线段/箭头必须增长；其他方向概率相应缩减时，线段/箭头必须同步变短。UI 必须实时显示 `policy_probs` 的归一化概率，而不是只显示 argmax 方向或缓存旧策略；训练过程应逐步发出每次更新后的最新策略概率，便于观察一个方向变长、其他方向变短的动态变化。
- 概率图应尽量接近书中/MATLAB 图示：用箭头而不是普通线段表达动作方向，箭头长度正比于动作概率；如果动作空间包含 stay/still 动作，则在格子中心画圆圈表示“停留”概率。
- 当前运行状态、更新状态只用边框高亮，不改变策略线本身。
- 右侧值网格显示当前策略下的真实 state value，例如 action-value 算法中应显示 `V^π(s)=Σ_a π(a|s)Q(s,a)`，不能显示最优/贪心状态值 `max_a Q(s,a)`；该值必须随当前 `policy_probs` 和 Q/函数近似结果实时更新。
- 公式区显示当前更新公式、变量和具体数值计算过程。
- 算法指针必须自动解析真实核心算法函数源码，左侧指针逐行高亮正在运行的代码行；不要手写固定 1-5 句伪代码。
- 速度控制应优先提供“输入训练间隔（ms）”的直接数值控制，而不是依赖不直观的滑块映射；用户输入的间隔值应直接同步到训练循环与算法指针展示节奏，便于观察核心代码逐步执行。
- 高速训练模式（`delay_ms` 接近 0）下，框架必须节流 UI 刷新频率，避免 `info_ready` 洪泛把主线程淹没，导致速度滑块、暂停/停止等控制看起来失灵；节流只影响显示频率，不能改变实际训练更新逻辑。
- TensorBoard 不需要在主 UI 中复刻；点击开始训练后应自动启动 TensorBoard，等待服务端端口就绪后再打开页面，避免浏览器先打开导致连接错误；页面应读取当前算法 `runs/<algorithm_name>` 日志并实时显示训练指标。
- Windows 下旧 TensorBoard 进程可能锁住 `runs/*_tensorboard*.log` 或事件日志目录；重新运行/清理日志时不能因为文件锁让主程序崩溃。能清理的日志应清理，仍被占用的进程输出日志可跳过，新 TensorBoard stdout 日志应使用唯一文件名。
- TensorBoard 指标需要拆分 tag，例如 `rollout/reward`、`train/loss`、`train/td_error`、`train/q_value`。
- TensorBoard 的实时性必须有明确节奏配置：训练指标按 `main.py` 中的写入步频写入，按步数或秒数强制 flush，TensorBoard 服务端 reload 间隔也应参数化；这些节奏应在 UI 中可见，避免用户误以为训练没有实时更新。
- 同一算法重新运行时必须清理上次数据，包括 Q 表、策略、checkpoint、TensorBoard 日志和 UI 状态。
- 框架需要提供“重新运行”和“继续上次”两种模式。
- 训练后需要保存 checkpoint，关闭窗口时也应保存；继续训练时可恢复上次算法状态。
- 需要有硬件检测模块：当前表格 RL 默认使用 CPU；检测到 NVIDIA GPU 时提示未来深度 RL 可走 CUDA；Intel 核显主要用于显示层，不用于当前小规模表格算法加速。
- README 应体现项目作者的真实学习动机，但“为什么写这个项目”部分必须简洁：基于赵世钰老师《强化学习的数学原理》的 GridWorld 环境，辅助强化学习算法学习，直观展示价值更新、策略变化和训练结果，帮助学习者把数学推导、代码实现和算法过程对应起来；不要写成长篇小作文。
- README 需要保留中英文可切换/折叠阅读结构，保留代码运行视频占位，并清楚说明项目基于什么构建、完成了哪些功能、如何运行、算法写在哪里、在 `main.py` 哪里切换训练算法和参数、环境参数在哪里改。

## 当前架构约定

- `algorithms/*.py`: 优先只写书中/用户笔记风格的核心算法纯函数，不写 UI、训练引擎、TensorBoard、checkpoint、Qt 代码。
- `algorithms/sarsa_value_function.py`: Algorithm 8.2 的纯函数实现，当前目标签名为 `sarsa_fa(w, s0, pi, q_hat, grad_q_hat, step, alpha, gamma, epsilon, episodes, max_steps) -> w`，其中 `pi` 明确表示由当前 `w` 诱导的 epsilon-greedy 行为策略。
- `core/algorithm_adapters.py`: 负责把书中纯函数适配到框架训练协议，并维护“按算法函数签名自动选适配器”的入口。新增同族算法时，优先通过签名匹配复用现有适配器；只有遇到全新算法族时，才在这里新增一个适配器分支，不能去改 engine/UI/session。
- `core/rl_parameters.py`: 保存当前算法所需的参数、共享运行时状态、transition/episode 结果和算法函数协议；可为书中常见标量符号预留共享命名（如 `alpha/beta/gamma/epsilon/lambda_/temperature/alpha_theta/alpha_w/alpha_v`），但不要引入与书中算法无关的私有协议字段。
- `core/table_agent.py`: 把适配后的核心算法函数包装成可训练、可视化、可保存的 Agent；它可以生成标准 `InfoDict`，但不能把 UI 逻辑写进算法本体。
- `core/engine.py`: 训练循环需要同时支持“框架逐步喂 transition”的一步驱动模式，以及“算法自己驱动整段 episode”的自主模式；扩展新算法族时优先复用这两种通用训练模式，不要再为单个算法临时拐弯。
- `core/table_agent.py`: 任何 episode 级临时缓存（如 SARSA 的 `a_next`）都必须在 `reset()` 时清空，不能跨 episode 泄漏到新的起点状态。
- `core/algorithm_trace.py`: 自动解析核心算法函数源码，供 UI 指针显示。
- `core/engine.py`: 只负责训练循环、速度控制、暂停/继续、定期保存和向 UI 发标准信息；不要把算法细节写进 engine。
- `envs/grid_world.py`: 纯手写 GridWorld 环境，只提供状态转移、奖励和布局，不依赖 Gym。
- `ui/`: 只消费标准 `InfoDict`，负责主环境框、策略概率图、值函数图、公式区和算法指针显示。
- `utils/session.py`: 管理 checkpoint、日志清理、重新运行和继续上次。
- `utils/logger.py`: 只负责 TensorBoard 标量写入，不复刻 UI 图表。
- `main.py`: 算法选择区应尽量只改 `CORE_ALGORITHM` 和 `ALGORITHM_CONFIG`；`ALGORITHM_NAME` 应优先从所选算法函数自动派生，减少手工同步成本。
- `runs/`、`sessions/`、`__pycache__/`、`.vendor/`、临时日志和外部 PDF 副本都是运行/缓存/临时产物，不属于代码结构；除非用户明确要保留结果，否则不要把它们当成需要维护的源码内容。

## 开发注意

- 项目 Python 解释器使用 Conda 环境 `MathInRL`；运行、编译和 smoke test 时优先执行 `conda activate MathInRL` 后再运行命令，或使用 `conda run -n MathInRL python ...`。
- 如果出现环境/依赖/解释器异常，必须先简短告知用户并暂停；只有在用户明确表示已经处理完并批准继续验证后，才能继续运行相关命令。不要为了绕开环境问题自行改用其他解释器、替代运行时或额外诊断路径消耗 token。
- 不要在环境/依赖诊断上消耗过多 token；如果测试环境看起来异常或缺依赖，简短告知用户需要在本地 `MathInRL` 环境检查/运行即可，不要反复排查环境问题。
- 每次根据用户新要求修改代码或框架时，必须同步把相关要求补充/更新到本 `AGENTS.md`，保持项目约束与最新用户意图一致。
- 每次准备修改 `algorithms/` 之外的项目代码前，必须先向用户汇报准备改哪些文件/模块，以及这些改动的具体功用；在用户已提出该协作原则的前提下，后续默认先汇报再动手，保持框架层改动透明。
- 代码注释应像人工维护代码时写下的说明：只解释当前职责、接口语义、数学含义或必要约束。删除或避免带有 AI/开发记录痕迹的注释，例如“用户要求/用户 prompt/根据请求/修复某问题/为避免某 bug/临时改成/不再使用某方案”等历史叙述。
- 注释不要复述显而易见的代码，也不要记录一次性修改原因；如果某段代码不再需要说明，应直接删掉注释。核心算法注释应贴近书中算法含义，框架层注释应描述稳定职责而不是描述迎合某次需求的改动。
- 代码应保持最小可用结构：删除无效、测试、临时、过度兜底和未使用的代码；不要保留为了调试一时添加的目录、依赖兜底、旧算法样例或未接入 UI 的组件。
- 使用 agent 开发时应尽量避免冗余和“屎山”：优先做最小、直接、可验证的改动；少写不必要的抽象、兜底、兼容层和长篇样板；保证 coding 消耗尽量少 token。
- 对当前阶段，优先围绕 Algorithm 8.2 的书中纯函数实现和可视化链路优化；其他算法只有在用户明确要求测试时再新增。
- 删除或清理文件时必须保护用户数据：只清理明确属于当前项目的运行产物、缓存或旧测试文件；不要删除用户桌面原始 PDF 或项目外文件。
- 新增依赖应优先写入 `requirements.txt`，不要用项目内 `.vendor/` 目录作为长期解决方案。
- 不要把 UI 逻辑写进 `algorithms/`。
- 不要把 `RLAlgorithmContext`、`RLTransition`、`RLStepResult` 这类框架协议强加到用户算法文件中；需要时通过 `core/algorithm_adapters.py` 适配。
- 不要让 UI 读取算法私有变量。
- 不要把策略概率画到主训练环境框里。
- 改算法时优先新增/修改 `algorithms/` 中的核心函数。
- 如果算法是整段 episode/training-loop 风格，优先新增/修改 `core/algorithm_adapters.py` 中的适配器，而不是污染算法本体。
- 扩展新算法时，优先先问“能否通过函数签名匹配复用现有适配器”；只有复用不了时，才在 `core/algorithm_adapters.py` 增加新的算法族适配器，不能为了某个算法去改训练引擎、UI 或会话管理层。
- 改参数时优先修改 `main.py` 的 `ALGORITHM_CONFIG`。
- 提交前至少运行 `python -m compileall -q .` 和一个无 UI 的算法 smoke test。
