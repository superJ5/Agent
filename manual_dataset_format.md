# 图文说明书 TXT 数据格式说明

## 1. 这批数据是什么

你现在拿到的不是原始 PDF，而是一种已经做过初步抽取的半结构化语料。

常见单条记录长这样：

```json
[
  "正文文本，里面夹着 <PIC> 占位符",
  ["Manual17_0", "Manual17_1", "fridge_01"]
]
```

它表示：

1. 第一项是说明书正文
2. 正文里的 `<PIC>` 表示这里原来有图片
3. 第二项是图片 ID 列表
4. `<PIC>` 和图片 ID 按顺序一一对应

也就是：

```text
第 1 个 <PIC> -> Manual17_0
第 2 个 <PIC> -> Manual17_1
第 3 个 <PIC> -> fridge_01
```

---

## 2. 为什么这不是普通 txt

因为它其实同时包含了三类信息：

1. 文本内容
2. 图片插入位置
3. 图片索引

所以它更像一条“图文对齐样本”，而不是单纯纯文本。

---

## 3. 我们要把它转成什么

为了进入后续清洗、切片、检索流程，我们先把它转成一条结构化 JSONL 记录：

```json
{
  "doc_id": "manual17",
  "source_txt": ".../manual17.txt",
  "image_dir": ".../manual17_images",
  "raw_text": "原始文本，保留 <PIC>",
  "clean_text": "去掉 <PIC> 后的纯文本",
  "pic_placeholder_count": 53,
  "image_id_count": 53,
  "alignment_ok": true,
  "pic_refs": [
    {
      "pic_index": 0,
      "image_id": "Manual17_0",
      "image_path": ".../Manual17_0.png",
      "matched_placeholder": true
    }
  ]
}
```

这一步的目的不是立刻做问答，而是先把：

- 文本
- 图片占位符
- 图片 ID
- 图片真实路径

整理到一条稳定的结构化记录里。

---

## 4. 这一步和我们 V1 的关系

如果原始语料是这种 txt 数据，而不是 PDF，那么 V1 的 Step 2 可以改成：

1. 先把单条 txt 样本转成结构化 JSONL
2. 校验 `<PIC>` 数量和图片 ID 数量是否一致
3. 校验图片目录里是否真的有对应图片
4. 再在此基础上做切片和 chunk-image 绑定

也就是说，这份 txt 数据已经替代了“PDF 解析”中的一部分工作。

---

## 5. 怎么转换

仓库里已经新增脚本：

- [scripts/convert_manual_dataset.py](/E:/.codex/worktrees/3f3c/ai_agent_competition/scripts/convert_manual_dataset.py)

用法：

```powershell
python scripts/convert_manual_dataset.py ^
  --input data/manuals/raw/manual17.txt ^
  --image-dir data/manuals/raw/manual17_images ^
  --output data/manuals/parsed/manual17.jsonl ^
  --doc-id manual17
```

脚本会做这些事：

1. 读取 txt 文件
2. 解析成数组或字典
3. 提取正文和图片 ID 列表
4. 统计 `<PIC>` 个数
5. 在图片目录中查找每个图片 ID 对应的真实文件
6. 输出结构化 JSONL

---

## 6. 你最先要检查什么

先检查两件事：

1. `<PIC>` 的数量是否等于图片 ID 列表长度
2. 图片目录中是否存在对应的图片文件

如果两者都成立，这份数据就很适合直接进入下一步切片。

---

## 7. 下一步是什么

等这一步转完后，下一步不是回头做 PDF，而是：

1. 设计切片规则
2. 把 `clean_text` 切成 chunk
3. 根据 `<PIC>` 的相对位置，把图片引用分配给相邻 chunk
4. 最终产出 `chunks.jsonl`
