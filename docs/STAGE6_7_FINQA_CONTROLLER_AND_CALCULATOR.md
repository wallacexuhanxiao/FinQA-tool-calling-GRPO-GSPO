# Stage6/Stage7 FinQA Controller and Calculator

这份文档说明 Stage6 / Stage7 里用于训练、GRPO reward 和评测的 calculator 与 controller。它和后面 LangChain 网站版 agent 不是同一套入口。

## 1. 本地文件位置

Stage6：

```text
stage6_finqa_agent/remote_files/scripts/agent/run_finqa_agent_controller.py
stage6_finqa_agent/remote_files/eval/finqa_calculator.py
```

Stage7：

```text
stage7_finqa_grpo/remote_files/scripts/agent/train_stage7_grpo_finqa_agent.py
stage7_finqa_grpo/remote_files/scripts/agent/run_finqa_agent_controller.py
stage7_finqa_grpo/remote_files/eval/finqa_calculator.py
```

其中 `run_finqa_agent_controller.py` 是 Stage6/Stage7 的 FinQA 闭环评测 controller；`finqa_calculator.py` 是真正执行 FinQA program 的 calculator。

## 2. Stage6 Controller 做什么

入口文件：

```text
run_finqa_agent_controller.py
```

核心流程：

```text
FinQA prompt
  -> model generates tool_call JSON
  -> parse tool name and program
  -> execute_program(program)
  -> build calculator observation
  -> feed observation back to model
  -> model generates final answer
  -> compute scaled execution/final accuracy
```

模型第一轮目标输出类似：

```json
{
  "action": "tool_call",
  "tool_call": {
    "name": "calculator",
    "arguments": {
      "program": "subtract(1200, 1000), divide(#0, 1000)"
    }
  }
}
```

controller 解析后调用：

```python
exec_result = execute_program(program)
```

然后构造 observation：

```python
obs_payload = {
    "ok": bool(exec_result.get("ok")),
    "result": format_number(exec_value) if exec_result.get("ok") else None,
    "error": exec_result.get("error"),
}
```

再把 observation 作为下一轮 user message：

```text
Tool observation from calculator:
{"ok": true, "result": "...", "error": null}
```

## 3. Calculator 怎么执行 program

入口文件：

```text
finqa_calculator.py
```

它不是 Python `eval()`，而是一个安全白名单 executor。

支持 op：

```text
add
subtract
multiply
divide
average
max
min
greater
less
exp
```

支持常量：

```text
const_m1, const_0, const_1, const_2, const_3, const_4, const_5,
const_10, const_100, const_1000, const_1000000
```

支持中间结果引用：

```text
#0, #1, #2 ...
```

例子：

```text
subtract(1200, 1000), divide(#0, 1000), multiply(#1, const_100)
```

执行过程：

```text
step 0: subtract(1200, 1000) -> 200
step 1: divide(#0, 1000) -> 0.2
step 2: multiply(#1, const_100) -> 20
```

返回：

```json
{
  "ok": true,
  "result": 20.0,
  "steps": [
    {"op": "subtract", "args": [1200.0, 1000.0], "result": 200.0},
    {"op": "divide", "args": [200.0, 1000.0], "result": 0.2},
    {"op": "multiply", "args": [0.2, 100.0], "result": 20.0}
  ],
  "error": null,
  "ops": ["subtract", "divide", "multiply"]
}
```

## 4. Stage7 GRPO reward 怎么用 calculator

入口文件：

```text
stage7_finqa_grpo/remote_files/scripts/agent/train_stage7_grpo_finqa_agent.py
```

reward 先解析模型 completion：

```python
json_ok, action, tool, program = parse_agent_tool(raw)
```

然后执行：

```python
exec_result = execute_program(program)
```

奖励大致包括：

```text
+ JSON valid
+ action == tool_call
+ tool == calculator
+ program non-empty
+ program executable
+ scaled execution correct @5%
+ scaled execution correct @1%
- invalid JSON / wrong action / wrong tool / empty program / execution failure
```

这里的关键是：Stage7 不是奖励模型“最后嘴上说对答案”，而是奖励它生成的 calculator program 执行后是否和 gold answer 数值匹配。

## 5. 和 LangChain 网站版的区别

Stage6/7 FinQA controller：

```text
只服务 FinQA 闭环评测和 GRPO reward
核心工具是 calculator
模型输出 program
Python executor 执行 program
```

LangChain 网站版 controller：

```text
服务对外 Web demo
工具包括 calculator / market_data / web_search / finance_api
用 LangChain StructuredTool 包装工具
用 FastAPI + vLLM 部署
```

二者共享理念：模型输出 tool-call JSON，Python controller 执行工具，再把 observation 回传模型。

但 Stage6/7 的 calculator 是 FinQA 专用 executor；网站版 calculator 是后来扩展出的通用工具之一。
