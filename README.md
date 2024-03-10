# SERA-LLM
*Search Engine Retrieval Augmented Large Language Model*

> Intern working log below
## Day 1: Description-as-Document with ChatGPT WebUI

1. Implemented description-as-document retrieval from google

2. Tested prompts for ChatGPT for prompt deep-processing and prompt-RAG
For question processing, a tested system input for ChatGPT is:
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
For **in-prompt SERAG** (*Search Engine Retrieval Augmented Generation*), a tested prompt for ChatGPT is:

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

## Day 2: Description-as-Document with ChatGPT API

1. Splitted and tuned the prompt, taking the system prompt and the user input apart
For question processing prompt:
```python
process_questions_system_input = """
            你是一个诚实的问答助手，如果你认为你能直接回答以下问题，则按以下格式输出：“答案: 该问题的答案”，否则，请针对以下问题生成{k}个更加深入的问题并以“问题：问题内容”输出。
            注意：
            1. 总是生成易于进行搜索引擎检索的问题。例如，不要生成类似“您……”或“你……”的问题，
            2. 总是以“问题：问题内容”输出生成的问题，不要任何附加输出
        """
process_questions_user_input = """
                问题：{question}
        """
```
When calling the gpt-turbo-3.5 API:
```python
response = model.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": self._final_system_prompt},
                {'role': 'user', "content": self._final_user_prompt}
            ]
            )
      output = response.choices[0].message.content
```

For final prompt input into ChatGPT:
```python
final_system_prompt = """
            你是一个诚实的问答助手，以下是用户的问题以及可能有用的文档，文档以 “序号：内容” 的格式给出：
            请逐步深入思考问题与文档的关系，并在思考的过程中注意：
            1. 如果你不能直接回答这个问题，则参考文档尝试回答
            2. 如果你参考文档回答了这个问题，请把参考文档的序号标注在相关回答之后
            3. 如果你认为参考文档无法帮助你回答这个问题，请回答“我还不了解这个问题的答案”
            4. 不要输出与问题不相关的文字
            尝试回答这个问题
        """
final_user_prompt = """
                问题：{question}
                可能有用的文档
                ...
                {docs}
                ...
        """
```
The API call is alike with question processing stage.
2. Added fallback in case that the LLM cannot proceed the question correctly

There are cases that ChatGPT cannot process the question in a correct format. To handle this, I added a fallback schema which returns the original prompt once ChatGPT cannot respond in the expected format.

3. Implemented in-prompt SERAG with ChatGPT API

Comparison between raw ChatGPT and ChatGPT with SERAG-question is in `./demos/chatgpt/chatgpt-desc-question-*.txt`

## Day 3: Description-as-Document with ChatGPT API(Evaluation)

1. (Update) Implemented the function that process the question by keywords instead of by deeper questions
Comparison between raw ChatGPT and ChatGPT with SERAG-keyword is in `./demos/chatgpt/chatgpt-desc-keyword-*.txt`

2. Evaluated SERAG with ChatGPT on [hotpot_qa](https://huggingface.co/datasets/hotpot_qa) 

F1-score:
- With RAG(documents from hotpot-qa): 0.715
- With SERAG-keyword: 0.890
- with SERAG-question: 0.914

## Day 4: Periodical Summary
Done: A naive SERAG system with Google Search and ChatGPT and its preliminary evaluation
TODOs in the next period: 
- [x] Add DuckDuckGo as a search engine
- [ ] Implement R3 query rewrite schemas
- [ ] Implement Page-as-Ducument with ChatGPT API
- [ ] Evaluate with LLM feedback

## Day 5: Add DuckDuckGo as a Search Engine
1. Added DuckDuckGo as a search engine

2. Evaluated SERAG-keyword & SERAG-question with DDG as a backend
