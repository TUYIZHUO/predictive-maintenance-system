# prompt/ —— 与 AI 交流记录存档

本目录用于保存课程设计过程中「与 AI 助手的对话记录」，作为 AI 辅助开发的留痕依据，

## 记录来源

Claude Code 会自动把每次会话的完整记录保存在本机：

```
C:\Users\ASUS\.claude\projects\C--Users-ASUS\c56814fc-5117-4723-8155-55e0908dae72.jsonl
```

该文件为 JSON Lines 格式（每行一个 JSON 对象），即任务书要求的「json 文件」形式，
可直接用文本编辑器或 `pandas.read_json(..., lines=True)` 打开。


## 重要提醒

AI 对话过长时会发生「上下文压缩」，早期原始记录会被摘要化、无法恢复。
务必在每个阶段结束前、且对话尚未被压缩时，及时执行一次备份，避免早期问答细节丢失。
