# AgroLake Git 与 GitHub 协作流程

本项目使用 Git 与指定私有仓库 [hoanhkhai1hk-wq/-](https://github.com/hoanhkhai1hk-wq/-)。仓库归用户指定账号所有，本地 origin 与此地址一致。

## 日常流程

WorkBuddy 按编号任务写代码和运行，Codex 检查代码差异、实际测试与依赖记录。每个通过验收的功能在开发分支保存独立提交；推送到指定私有仓库后，以提交/分支或 PR 对应任务证据。主分支保留可运行版本。历史提交和原始结果保留，禁止用强制推送掩盖失败。

只添加明确审阅的源码、配置样例、依赖锁和使用说明，不运行 `git add .`。`.gitignore` 已排除运行环境、数据库、原始农业数据、权重、科研原稿、聊天与协作日志；共享记忆只在本地维护，必要的远程工作说明另行编写。密钥通过本地环境变量或忽略的配置提供，不写入提交。

## GitHub 连接与仓库所有权

GitHub 插件已通过实际账号查询和上游固定文件读取验证可用。本人完成设备授权后，本机 GitHub CLI 登录到仓库所有者账号；已实际检查该仓库为 private、具有 pull/push 权限，并 fetch 得到远程初始提交。CLI 与插件是两套认证，不由其中一项推定另一项可访问。

使用其他账号的私有库时，由所有者邀请当前使用账号为协作者并接受邀请，再为 GitHub App 授权相应仓库。密码和令牌由本人管理，不发到聊天或项目记录。依据：[GitHub 的个人仓库协作者说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository)。

Codex 保留远程初始提交并合并到本地历史，以普通推送同步已审阅内容；禁止 force push。每次同步核对远程提交 SHA，失败保持实际失败状态，不把能够读取公开上游称为本项目仓库已同步。
