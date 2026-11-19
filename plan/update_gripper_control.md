# 夹爪位置控制与开合速度调查记录

## 调查状态：暂定调查完毕

最终更新：2026-10-14。

用户将新接口的 `max_effort` 从 `0` 改为 `100` 后，观察到开关速度大幅提升。当前证据支持：**`max_effort` 是本次慢速问题的关键影响参数**。位置继续使用归一化 `[0,1]`，撤销乘以 `0.085` 的单位转换尝试。

这属于实际使用结果确认，尚未查明驱动内部机制。`max_effort` 在消息定义中仍是力参数，不能直接当作速度参数，也不能推断越大越快。按当前科研使用需求，暂不继续追查驱动源码。

## 调查范围

本次调查针对将夹爪控制从 BasicCommand 接口切换到位置控制接口后，开合速度明显变慢的问题。

调查对象：

- 机器人上位机：`192.168.1.7`
- ROS bridge：`192.168.1.7:9090`
- 当前客户端：`src/robokit/robots/robot_client_inovo.py`
- 手柄控制：`src/robokit/controllers/tele_control.py`
- 参考资料：`debug/ros/de_roslibpy.ipynb`

调查先进行了 ROS topic、消息类型、节点和服务的只读查询；随后尝试并撤销了位置单位转换，并经用户授权执行了新旧接口各一轮空载开合测试。最后由用户修改 `max_effort` 并反馈速度明显提升。本次收尾只更新此文档。

## 已确认的 ROS 接口

机器人同时提供新旧两套夹爪 action，且由同一个驱动节点处理：

```text
/devices/robotiqd/grip/goal
/devices/robotiqd/gripper_command/goal
```

驱动节点为 `/devices/robotiqd/robotiqd`，同时订阅两套 action 的 goal/cancel 话题。

### 旧 BasicCommand 接口

```text
Topic: /devices/robotiqd/grip/goal
Type:  gripper_msgs/GripperBasicCommandActionGoal
```

目标字段为 `goal.action`：

```text
0: GRIP
1: RELEASE
2: TOGGLE
```

该接口只表达预定义的开、关动作，没有位置、速度或力参数。

### 新 GripperCommand 接口

```text
Topic: /devices/robotiqd/gripper_command/goal
Type:  control_msgs/GripperCommandActionGoal
```

目标字段为：

```text
goal.command.position
goal.command.max_effort
```

反馈和结果中的字段为：

```text
position
effort
stalled
reached_goal
```

该消息没有显式的速度或加速度字段；实际速度受驱动实现及其对目标参数的处理影响，不能因此排除 `max_effort` 对运动速度的影响。

## 夹爪状态和硬件信息

状态话题为：

```text
/devices/robotiqd/gripper_state
gripper_msgs/GripperState
```

查询到的状态样例：

```text
position: 0.9882352941176471
target_position: 1.0
current: 0.0
stalled: false
reached_goal: true
activated: true
safety_switch_triggered: false
```

状态消息中的 `position` 和 `target_position` 看起来是归一化的 `[0,1]` 值。

只读服务 `/devices/robotiqd/gripper_info` 返回：

```text
gripper_name: Robotiq 2F-85
max_aperture: 0.0
min_aperture: 0.08500000089406967
max_effort: 200.0
```

其中 `0.085 m` 与 Robotiq 2F-85 的 85 mm 实际行程一致。字段名 `max_aperture` / `min_aperture` 的方向需要结合驱动文档进一步确认，但可以确认驱动暴露了约 `0.085 m` 的物理开口范围。

## 当前代码的关键行为

当前代码将旧的二值语义映射到新位置接口：

```python
# 闭合
position = 0.0

# 张开
position = 1.0
```

手柄控制仍保持原语义：按住 PS5 6/7 号键时 `g=1` 闭合，松开时 `g=0` 张开。控制循环和 `gripper_set_pub()` 的发送逻辑没有改变，只在开闭状态切换时发送一次目标。因此目前没有证据表明速度变慢是由手柄循环频率或客户端重复发送造成的。

文档收尾时读取到代码中的 `max_effort` 为 `85.0`；用户明确报告速度改善的试验值为 `100`。两者分别记录，尚无 `85` 与 `100` 的定量速度或抓取力对照数据。本次未改动该配置。

## 可能原因一：位置单位或范围不匹配

### 初始假设

标准消息的常见物理单位约定及 `GripperInfo` 返回的约 `0.085 m` 行程，曾使我们怀疑发送 `1.0` 超出有效范围。但这些信息不能证明当前厂商驱动的输入单位，也没有证据证明超范围目标导致慢速轨迹。

### 验证结果：不支持，修改已撤销

- 曾在发送处将 `position` 乘以 `0.085`，其余控制逻辑保持原样。
- 用户反馈速度没有变化，而且只会关闭、不会打开。
- 随后的只读状态为 `position=0.11372549`、`target_position=0.08235294`、`reached_goal=true`。目标接近发送的 `0.085`，而非全开对应的 `1.0`，支持它被当作接近闭合的小开度处理。
- 已撤销缩放，恢复直接发送归一化 `[0,1]`。不再将位置单位不匹配作为本次慢速问题的主要解释。

`GripperInfo` 的 min/max 字段方向仍未解释，不影响本次恢复原位置映射的决定。

## 可能原因二：新旧 action 使用不同的速度配置

### 参数与实现核查

- 完整 `/devices/robotiqd` 参数命名空间只有 `robotiqd/dev: wrist`。
- `/drivers/robotiqd` 的 package 和 executable 均为 `robotiqd`，args 为 `_dev:=wrist`。
- 当前可见的 ROS 参数中未找到夹爪速度或加速度设置，该驱动节点也未暴露 `set_parameters` 服务。这不能排除内部默认值或非 ROS 参数配置。
- 驱动 logger 为 `info`；被动监听期间没有收到相关日志，未更改日志级别。
- 本地及查阅的 Inovo 官方公开仓库未找到 `robotiqd` 服务端实现，尚未取得上位机源码或配置文件。无法确认两套回调下发的速度、力参数。

### 空载动作对照：新接口在 `max_effort=0` 时约慢 3 倍

经授权实际执行了旧接口开、关各一次，新接口开、关各一次。新接口使用 `max_effort=0.0`。启动延迟按位置相对初始值变化超过 `0.02` 计算，总耗时按客户端发送到收到对应 action 成功结果计算。

| 接口 | 动作 | 启动延迟 | 发送到返回成功 |
|---|---|---:|---:|
| 旧 `/grip` | 张开 | 0.24 秒 | 1.24 秒 |
| 新 `/gripper_command` | 张开 | 0.47 秒 | 3.68 秒 |
| 旧 `/grip` | 闭合 | 0.22 秒 | 1.02 秒 |
| 新 `/gripper_command` | 闭合 | 0.26 秒 | 3.46 秒 |

旧接口 RELEASE 的实际目标是 `0.84705882`，不是 `1.0`。首次测试按接近 `1.0` 判断完成，因此尽管动作已成功，测试仍等待至 12 秒并停止后续动作；表中的 1.24 秒来自匹配 goal ID 的成功结果，不是该等待超时。随后按实际端点继续完成剩余三次动作。

两次张开的起点分别约 `0.114`、`0.106`，终点均为 `0.847`；两次闭合均从 `0.847` 开始，目标均为 `0.0`。因此行程接近，明显耗时差异不能用行程不同解释。闭合启动延迟也接近，差异主要发生在开始运动之后。

两套接口闭合后的实际位置均约 `0.11`，同时返回 `SUCCEEDED`、`reached_goal=true`、`stalled=false`，不能只以位置接近 `0.0` 判断完成。测试最后停留在闭合目标状态。

这是单轮测试，状态采样约 5 Hz，时间用于判断明显差异，不作为精确性能指标。测试绕过手柄控制循环，支持差异位于服务端处理或驱动行为；但并未独立证明存在不同的默认速度配置，且需要结合下面的 `max_effort` 结果解释。

## 可能原因三：`max_effort` 对实际速度的影响

### 验证结果：当前关键影响参数

慢速版本及上述对照测试均发送 `max_effort=0.0`。用户随后将其改为 `100`，反馈开关速度大幅提升。该结果支持将 `max_effort` 设置作为本次问题的实际处理方向；未记录此次改善的精确耗时，也未证明速度完全恢复到旧接口水平。

此前以“最大作用力不是速度参数”为由弱化其影响的判断不成立，现予以修正。字段语义不是速度，并不意味着不会影响运动速度；驱动对零值的处理、电机输出或内部参数映射都可能相关，但目前均未获得实现证据。不能断言零值表示“不限力”，也不能把 `max_effort` 当成可线性调节速度的量。

### 当前处理

- 保留归一化位置输入，不做米制缩放。
- 保留用户当前设置的非零 `max_effort`；收尾时代码为 `85.0`，明确报告改善的测试值为 `100`。
- 不改变手柄语义、状态机或数据采集格式，不扩展配置系统。
- `gripper_info.max_effort=200.0` 仅作为服务返回信息记录，不作为推荐设置值。

## 可能原因四：Action goal 直接发布方式的影响

当前代码通过 `roslibpy.Topic.publish()` 直接发布 `ActionGoal`，没有显式使用 `roslibpy.actionlib.ActionClient`。理论上 action goal 还包含 `header`、`goal_id` 和 `goal`，而当前发布消息只填写了 `goal.command`。

### 验证结果：不足以解释本次速度差异

空载对照中两套接口均直接发布 ActionGoal，并显式设置唯一 goal ID，均收到了对应的 `SUCCEEDED` 结果；新接口仍明显更慢。因此缺少 goal ID 不是解释此次慢速的充分原因，暂不为该问题改用 ActionClient。

没有执行 ActionClient 对照测试，完整抢占及生命周期行为不在本次结案范围内。

## 暂定结论与后续追踪条件

**本问题暂定调查完毕：取消无效的位置单位转换，保留 `[0,1]` 位置控制，并依据用户实测使用非零 `max_effort`。** 当前已经找到能显著改善速度的设置，尚未确认驱动内部根因，不再将单位不匹配或默认速度差异列为已证实结论。

以下项目只在后续使用出现问题或需要精确调参时重新开启，不作为当前结案的阻塞项：

- 定量比较 `85`、`100` 的耗时与实际抓取力度；空载速度改善不能代替物体抓取验证。
- 核对全开、全闭、中间开度及运动中反向的表现，记录当前尚未完成的验证。
- 获取 `robotiqd` 实现，确认 `max_effort=0` 的解释及速度/力参数转换。
