from typing import List, Tuple, Union
from openai import OpenAI
from base_llm import RagBaseLLM

OPENAI_KEY = 'sk-wKk81cwZTJucpgbSaMcVZr1dTvgoISB1vaSBkw0mH3nc2d0c'

model = OpenAI(
    api_key=OPENAI_KEY,
    base_url="https://api.chatanywhere.tech/v1"
)

class RagChatGPT(RagBaseLLM):
    def __init__(self, 
                 model, 
                 k=3, 
                 lang='zh-CN',
                 rewrite_method='hyqr',
                 retrieve_backend='ddg',
                 read_method='description') -> None:
        super().__init__(model, 
                         k, 
                         lang, 
                         rewrite_method, 
                         retrieve_backend, 
                         read_method)
        if self.lang == 'zh-CN':
            if self.rewrite_method == 'hykr':
                self._process_questions_system_input = """
            你是一个诚实的问答助手，如果你认为你能直接回答以下问题，则按以下格式输出：“答案: 该问题的答案”，否则，请针对以下问题生成{k}个关键字并直接输出。
            注意：
            1. 总是生成易于进行搜索引擎检索的关键字
            2. 总是直接输出生成的关键字，不要任何附加输出
        """
            elif self.rewrite_method == 'hyqr':
                self._process_questions_system_input = """
            你是一个诚实的问答助手，如果你认为你能直接回答以下问题，则按以下格式输出：“答案: 该问题的答案”，否则，请针对以下问题生成{k}个更加深入的问题并以“问题：问题内容”输出。
            注意：
            1. 总是生成易于进行搜索引擎检索的问题。例如，不要生成类似“您……”或“你……”的问题，
            2. 总是以“问题：问题内容”输出生成的问题，不要任何附加输出
        """
            elif self.rewrite_method == 'r3':
                self._process_questions_system_input = """
             你是一个诚实的问答助手，如果你认为你能直接回答以下问题，则按以下格式输出：“答案: 该问题的答案”，否则，请将此问题改写成1个更加易于搜索引擎检索的查询并以“查询：查询内容”输出
             注意：
             1. 总是生成易于进行搜索引擎检索的问题。例如，不要生成类似“您……”或“你……”的问题，
             2. 总是以“查询：查询内容”输出生成的问题，不要任何附加输出
             下面是一个例子：
             问题：杜甫的春望表达了诗人的什么情感?
             查询：杜甫 春望 思想感情
        """
            else:
                raise NotImplementedError
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
        else:
            if self.rewrite_method == 'hykr':
                self._process_questions_system_input = """
            You are an honest Q&A assistant. If you think you can directly answer the following questions, output in the following format: "Answer: the answer to the question". Otherwise, please generate {k} keywords for the following questions and output them directly.
             Notice:
             1. Always generate keywords that are easy to retrieve by search engines
             2. Always output the generated keywords directly without any additional output
        """
            elif self.rewrite_method == 'hyqr':
                self._process_questions_system_input = """
            You are an honest Q&A assistant. If you think you can directly answer the following questions, output in the following format: "Answer: the answer to the question". Otherwise, please generate {k} more in-depth questions for the following questions and "Question:Question content" output.
             Notice:
             1. Always generate questions that are easy to retrieve by search engines.
             2. Always output generated questions as "Question: Question content" without any additional output
        """
            elif self.rewrite_method == 'r3':
                self._process_questions_system_input = """
            You are an honest Q&A assistant. If you think you can directly answer the following question, output it in the following format: "Answer: the answer to the question". Otherwise, please rewrite this question into a query that is easier for search engines to retrieve. And output as "Query:Query content"
              Notice:
              1. Always generate questions that are easily indexable by search engine.
              2. Always output generated questions as "Query: Query content" without any additional output
              Below is an example:
              Question: Who are the main characters in Harry Potter?
              Query: Harry Potter main characters
        """
            else:
                raise NotImplementedError
            self._process_questions_user_input = """
                    Question: {question}
            """
            self._final_system_prompt = """
                You are an honest Q&A assistant. The following are user questions and potentially useful documents. The documents are given in the format of "Index: Content":
                Please gradually and deeply think about the relationship between the problem and the document, and pay attention to the following during the thinking process:
                1. If you cannot answer the question directly, please refer to the documentation and try to answer it.
                2. If you refer to the document to answer this question, please mark the index of the reference document after the relevant answer.
                3. If you feel that the reference documentation cannot help you answer this question, please answer "I don't know the answer to this question yet"
                4. Do not output text that is not relevant to the question
                try to answer this question
            """
            self._final_user_prompt = """
                    Question：{question}
                    Possibly useful documents
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
        if self.rewrite_method == 'hykr':
            final_query = ''
            # if(query.startswith("关键字")):
            #     final_query = query[4:]
            # elif query.startswith(("Keywords",'keywords')):
            #     final_query = query[9:]
            # else:
            #     final_query = prompt
                
            # return (final_query, True)
            if '关键字' in query or 'keyword' in query.lower():
                if ':' in query:
                    if len(query.split(':')[1]) != 0:
                        final_query = query.split(':')[1].strip()
                elif '：' in query:
                    if len(query.split('：')[1]) != 0:
                        final_query = query.split('：')[1].strip()
                elif '.' in query:
                    if len(query.split('.')[1]) != 0:
                        final_query = query.split('.')[1].strip()
                else:
                    if query.strip() != '':
                        final_query = query.strip()
                return (final_query, True) if len(final_query) != 0 else (prompt, True)
            # else:
            #     if(query.startswith(('答案'))):
            #         return (query[3:], False)
            #     elif query.startswith(("Answer",'answer')):
            #         return (query[7:], False)
            #     else:
            #         return (prompt, True)
        elif self.rewrite_method == 'hyqr':
            final_query = []
            if '问题' in query or 'question' in query.lower():
                questions = query.split('\n')
                # print(questions)
                for q in questions:
                    # print(q)
                    if ':' in q:
                        if len(q.split(':')[1]) != 0:
                            final_query.append(q.split(':')[1].strip())
                    elif '：' in q:
                        if len(q.split('：')[1]) != 0:
                            final_query.append(q.split('：')[1].strip())
                    elif '.' in q:
                        if len(q.split('.')[1]) != 0:
                            final_query.append(q.split('.')[1].strip())
                    else:
                        if q.strip() != '':
                            final_query.append(q.strip())
                return (final_query, True) if len(final_query) != 0 else (prompt, True)
        elif self.rewrite_method == 'r3':
            final_query = ''
            if '查询' in query or 'query' in query.lower():
                if ':' in query:
                    if len(query.split(':')[1]) != 0:
                        final_query = query.split(':')[1].strip()
                elif '：' in query:
                    if len(query.split('：')[1]) != 0:
                        final_query = query.split('：')[1].strip()
                elif '.' in query:
                    if len(query.split('.')[1]) != 0:
                        final_query = query.split('.')[1].strip()
                else:
                    if query.strip() != '':
                        final_query = query.strip()
                return (final_query, True) if len(final_query) != 0 else (prompt, True)
        else:
            raise NotImplementedError
        if(query.startswith(('答案'))):
            return (query[3:], False)
        elif query.startswith(("Answer",'answer')):
            return (query[7:], False)
        else:
            return (prompt, True)
    
    def _get_final_user_prompt(self, 
                               prompt: str,
                               num_results_single: int, 
                               num_results_multi: int):
        
        docs_list = []
        # Stage 1
        query = self._process_questions(prompt=prompt)
        # print(query)

        if not query[1]:
            if self.lang == 'en':
                docs_list.append(query[0])
            final_query = prompt
        else:
            final_query = query[0]
        
        # Stage 2
        if self.read_method == 'description':
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

        elif self.read_method == 'page':
            raise NotImplementedError
        else:
            raise NotImplementedError
                     
    def test_process_questions(self, prompt):
        return self._process_questions(prompt=prompt)
    
    def chat(self, 
             prompt: str,
             num_results_single: int = 10, # number of results for a single query
             num_results_multi: int = 2   # number of results for multiple queries
            ):
      self._process_questions_user_input_cache = self._process_questions_user_input
      self._final_user_prompt_cache = self._final_user_prompt
      self._get_final_user_prompt(prompt=prompt,
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
    Evaluation on [hotpot_qa](https://huggingface.co/datasets/hotpot_qa)
    """
    def _get_final_user_prompt_hotpot(self, 
                                      prompt: str, 
                                      docs_list: List[str],
                                      ):
            self._final_user_prompt = self._final_user_prompt.replace("{question}", prompt)
            docs = ''
            for i, d in enumerate(docs_list):
                docs += f'[{i + 1}]：{d}\n'
            self.docs = docs
            # print(docs)
            self._final_user_prompt = self._final_user_prompt.replace("{docs}",docs)
        

    def evaluation_hotpot(self, 
                    prompt: str,  
                    docs_list: List[str] | None = None,
                    method: str = 'description',
                    num_results_single: int = 10, # number of results for a single query
                    num_results_multi: int = 2   # number of results for multiple queries
                    ):
        self._final_system_prompt = """
                You are an honest Q&A assistant. The following are user questions and potentially useful documents. The documents are given in the format of "Index: Content":
                Please gradually and deeply think about the relationship between the problem and the document, and pay attention to the following during the thinking process:
                1. Only the answer to the question is needed, no other words are needed!
                try to answer this question
            """
        self._process_questions_user_input_cache = self._process_questions_user_input
        self._final_user_prompt_cache = self._final_user_prompt
        if docs_list is None:
            docs_list = []
            # Stage 1
            query = self._process_questions(prompt=prompt)
            # print(query)

            if not query[1]:
                if self.lang == 'en':
                    docs_list.append(query[0])
                final_query = prompt
            else:
                final_query = query[0]
            
            # Stage 2
            if method == 'description':
                docs_list = self._get_desc_as_doc(query=final_query, 
                                                docs=docs_list,
                                                num_results_single=num_results_single,
                                                num_results_multi=num_results_multi)
                
                self._get_final_user_prompt_hotpot(prompt=prompt, docs_list=docs_list)

            elif method == 'page':
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            self._get_final_user_prompt_hotpot(prompt=prompt, docs_list=docs_list)
        
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

# # Raw model 
# question = "介绍几个比较成熟的RAG框架"
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

# # RAG model
# rag_llm = RagChatGPT(model, rewrite_method='hykr')
# # response = llm.test_process_questions("我要使用xtuner微调一个语言模型，该如何做？")
# # print(response)
# output_rag = rag_llm.chat(question)
# # print(rag_llm.docs)
# print("""
# RAG model
# ------------------------------------
#       """)
# print(output_rag)