"""电商客服 Agent 应用包。

Windows 中文控制台默认代码页是 GBK（cp936），代码里用于 CLI 提示的 emoji
（如路由打印的 🔀）在 GBK 下会抛 UnicodeEncodeError 直接中断对话或评测。
这里统一把标准输出/错误流的编码错误降级为替换字符：终端支持 UTF-8 时输出不变，
不支持时退化成 "?"，不再因为一个表情符号让整条链路失败。
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):  # pytest 等包装过的流不支持 reconfigure
        pass
