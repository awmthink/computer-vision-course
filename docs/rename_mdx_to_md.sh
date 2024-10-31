#!/bin/bash

# 指定要处理的文件夹路径，默认为当前目录
folder_path=${1:-.}

# 查找所有 .mdx 文件并重命名为 .md
find "$folder_path" -type f -name "*.mdx" | while read file; do
    # 使用重命名命令将 .mdx 后缀改为 .md
    mv "$file" "${file%.mdx}.md"
done

echo "重命名完成！"

