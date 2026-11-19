# AGENTS.md

## Code style
- 新增 public class 和 public function 时添加简洁 docstring。
* 在关键位置加上注释
* 变量如果是 tenser, numpy 类型，要用 `x_B_C_T_H_W` 的命名方式
* 减少非必要的异常处理，因为作为科研代码，用户运行时会尽可能保证变量合法，只在必要的地方报错方便 debug 即可
* 在未说明时，不要新增 test 脚本，只做静态编译检查即可，因为本地 coding 环境不可运行，不像实际服务器上的训练环境

## Implementation preferences
- 代码实现应尽可能简短、易读，因为这是科研代码，不是工程项目
- 修改代码时，要尽量把所有相关代码纳入阅读
- 注释重点解释设计原因和 tensor shape，不逐行翻译代码。

## Validation
- 未明确要求时，不新增独立测试脚本。
- 如果当前机器缺少训练依赖，进行静态检查并明确说明未运行的验证。

## Plan preferences
- 在 plan 时不确定的细节要及时询问
- plan 的输出内容要总结需要修改哪些文件
