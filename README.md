# Zhenpeng 的技术博客

基于 GitHub Pages + Jekyll（minima 主题）。

## 写作方式
在 `_posts/` 目录下新增 Markdown 文件，命名格式为：
```
YYYY-MM-DD-你的标题.md
```
示例：
```
2025-10-31-hello-world.md
```

Front matter 示例：
```yaml
---
layout: post
title: "你的文章标题"
date: 2025-10-31 20:00:00 +0800
categories: [后端, 架构]
tags: [Java, 性能优化]
---
```

## 本地预览（可选）
需要 Ruby 环境，随后：
```bash
bundle install
bundle exec jekyll serve
```
打开 http://127.0.0.1:4000 预览。
