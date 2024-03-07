from typing import List, Tuple, Union
from openai import OpenAI
from base_llm import RagBaseLLM

OPENAI_KEY = 'sk-wKk81cwZTJucpgbSaMcVZr1dTvgoISB1vaSBkw0mH3nc2d0c'

model = OpenAI(
    api_key=OPENAI_KEY,
    base_url="https://api.chatanywhere.tech/v1"
)

class RagChatGPT(RagBaseLLM):
    def __init__(self, model, k=3) -> None:
        super().__init__(model, k)
        self._process_questions_system_input = """
            你是一个诚实的问答助手，如果你认为你能直接回答以下问题，则按以下格式输出：“答案: 该问题的答案”，否则，请针对以下问题生成{k}个更加深入的问题并以“问题：问题内容”输出。
            注意：
            1. 总是生成易于进行搜索引擎检索的问题。例如，不要生成类似“您……”或“你……”的问题，
            2. 总是以“问题：问题内容”输出生成的问题，不要任何附加输出
        """
        self._process_questions_user_input = """
                问题：{question}
            """
        self._final_system_prompt = """
            你是一个诚实的问答助手，以下是用户的问题以及可能有用的文档，文档以 “序号：内容” 的格式给出：
            请逐步深入思考问题与文档的关系，并在思考的过程中注意：
            1. 如果你不能直接回答这个问题，则参考文档尝试回答
            2. 如果你参考文档回答了这个问题，请把参考文档的序号标注在相关回答之后
            3. 如果你认为参考文档无法帮助你回答这个问题，请回答“我还不了解这个问题的答案”
            4. 不要输出与问题不相关的文字
            尝试回答这个问题
        """
        self._final_user_prompt = """
                问题：{question}
                可能有用的文档
                ...
                {docs}
                ...
            """
    
    def _process_questions(self, 
                           prompt: str) -> Union[Tuple[List, bool], Tuple[str, bool]]:
        
        self._process_questions_system_input = self._process_questions_system_input.replace("{k}",str(self.k))
        self._process_questions_user_input = self._process_questions_user_input.replace("{question}",prompt)
        response = self.model.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": self._process_questions_system_input},
                {'role': 'user', "content": self._process_questions_user_input}
            ]
            )
        query = response.choices[0].message.content
        # print(query)
        final_query = []
        if '问题' in query:
            questions = query.split('\n')
            # print(questions)
            for q in questions:
                # print(q)
                if ':' in q:
                    if len(q.split(':')[1]) != 0:
                        final_query.append(q.split(':')[1])
                elif '：' in q:
                    if len(q.split('：')[1]) != 0:
                        final_query.append(q.split('：')[1])
                elif '.' in q:
                    if len(q.split('.')[1]) != 0:
                        final_query.append(q.split('.')[1].strip())
                else:
                    final_query.append(q)
            return (final_query, True) if len(final_query) != 0 else (prompt, True)
        else:
            if(query.startswith('答案')):
                return (query[3:], False)
            else:
                return (prompt, True)
    
    def _get_final_user_prompt(self, 
                               prompt: str, 
                               method: str = "description", 
                               num_results_single: int = 10, 
                               num_results_multi: int = 2):
        
        docs_list = []
        # Stage 1
        query = self._process_questions(prompt=prompt)
        # print(query)

        if not query[1]:
            # docs_list.append(query[0])
            final_query = prompt
        else:
            final_query = query[0]
        
        # Stage 2
        if method == 'description':
            docs_list = self._get_desc_as_doc(query=final_query, 
                                              docs=docs_list,
                                              num_results_single=num_results_single,
                                              num_results_multi=num_results_multi)
            self._final_user_prompt = self._final_user_prompt.replace("{question}", prompt)
            docs = ''
            for i, d in enumerate(docs_list):
                docs += f'[{i + 1}]：{d}\n'
            self.docs = docs
            # print(docs)
            self._final_user_prompt = self._final_user_prompt.replace("{docs}",docs)
            # print(self._final_user_prompt)

        elif method == 'page':
            raise NotImplementedError
        else:
            raise NotImplementedError
                     
    def test_process_questions(self, prompt):
        return self._process_questions(prompt=prompt)
    
    def chat(self, 
             prompt: str, 
             method: str = 'description',
             num_results_single: int = 10, # number of results for a single query
             num_results_multi: int = 2   # number of results for multiple queries
            ):
      self._process_questions_user_input_cache = self._process_questions_user_input
      self._final_user_prompt_cache = self._final_user_prompt
      self._get_final_user_prompt(prompt=prompt,
                                  method=method,
                                  num_results_single=num_results_single,
                                  num_results_multi=num_results_multi)
      response = self.model.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": self._final_system_prompt},
                {'role': 'user', "content": self._final_user_prompt}
            ]
            )
      output = response.choices[0].message.content
      self._process_questions_user_input = self._process_questions_user_input_cache
      self._final_user_prompt = self._final_user_prompt_cache
      return output
    

    
"""
Test code below
"""

# ### Raw model 
# question = "请介绍一下RAG技术中，Embedding模型对检索结果的影响"
# raw_system_input = """
#     你是一个诚实的问答助手，以下是用户的问题，请尝试回答这个问题，注意：
#     1. 如果你不能直接回答这个问题，请回答“我还不了解这个问题的答案”
# """
# output_raw = model.chat.completions.create(
#      model="gpt-3.5-turbo",
#                 messages=[
#                 {"role": "system", "content": raw_system_input},
#                 {'role': 'user', "content": f"问题：{question}"}
#             ]
# ).choices[0].message.content
# print("""
# Raw model
# ------------------------------------
#       """)
# print(output_raw)

# ### RAG model
# rag_llm = RagChatGPT(model)
# # response = llm.test_process_questions("我要使用xtuner微调一个语言模型，该如何做？")
# # print(response)
# output_rag = rag_llm.chat(question)
# print("""
# RAG model
# ------------------------------------
#       """)
# print(output_rag)