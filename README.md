# SEALM
*Search Engine Augmented Language Model*

## Day 1: Description-as-Document with Prompt Engineering with ChatGPT WebUI

1. Completed description-as-document retrieval from google
See [here](./sealm/search.py)

2. Tested prompts for ChatGPT for prompt deep-processing and prompt-RAG
For prompt deep-processing, a tested system input for ChatGPT is:
```python
"""
            你是一个诚实的问答助手，如果你认为你能直接回答以下问题，则按以下格式输出：“答案: 该问题的答案”，否则，请针对以下问题生成{k}个更加深入的问题并直接输出。
            注意：
            1. 总是生成易于进行搜索引擎检索的问题。例如，不要生成类似“您……”或“你……”的问题，
            2. 请必须以JSON列表的形式返回问题
            问题：{question}
        """
```
The model is more prone to generate questions than answer it directly although it knows the answer
For in-prompt RAG(which means references are in the prompt), a tested prompt for ChatGPT is:

```python
"""
            你是一个诚实的问答助手，以下是用户的问题以及可能有用的文档，文档以 “序号：内容” 的格式给出：
            问题：{question}
            文档：
            ...
            {docs}
            ...
            请逐步深入思考问题与文档的关系，并在思考的过程中注意：
            1. 如果你不能直接回答这个问题，则参考文档尝试回答
            2. 如果你参考文档回答了这个问题，请把参考文档的序号以Markdown上标的形式标注在相关回答之后
            3. 如果你认为参考文档无法帮助你回答这个问题，请回答“我还不了解这个问题的答案”
            4. 不要输出与问题不相关的文字
            尝试回答这个问题
        """
```

However, this do not always work properly. ChatGPT tend to drop into hallucination than admit that it doesn't know this when the documents is irrelavent. Among these malfunctions there is [a surprising one](https://chat.openai.com/share/9c26faf9-4ad7-4430-ae03-2ca15833c628) which shows in multi-turn dialogue ChatGPT **can admit its incapability in answering the question only after the 1st turn of dialogue**

## Day 2: Description-as-Document with Prompt Engineering with ChatGPT API