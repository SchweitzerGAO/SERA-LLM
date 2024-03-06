from typing import List, Tuple, Union
from openai import OpenAI
from base_llm import BaseLLM

OPENAI_KEY = 'sk-wKk81cwZTJucpgbSaMcVZr1dTvgoISB1vaSBkw0mH3nc2d0c'

model = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=OPENAI_KEY,
    base_url="https://api.chatanywhere.tech/v1"
)

class ChatGPT(BaseLLM):
    def __init__(self, model, k=3) -> None:
        super().__init__(model, k)
        self._proceed_questions_system_input = """
            你是一个诚实的问答助手，如果你认为你能直接回答以下问题，则按以下格式输出：“答案: 该问题的答案”，否则，请针对以下问题生成{k}个更加深入的问题并直接输出。
            注意：
            1. 总是生成易于进行搜索引擎检索的问题。例如，不要生成类似“您……”或“你……”的问题，
            2. 总是直接输出生成的问题，不要任何附加输出
            问题：{question}
        """
        self._final_prompt = """
            你是一个诚实的问答助手，以下是用户的问题以及可能有用的文档，文档以 “序号：内容” 的格式给出：
            请逐步深入思考问题与文档的关系，并在思考的过程中注意：
            1. 如果你不能直接回答这个问题，则参考文档尝试回答
            2. 如果你参考文档回答了这个问题，请把参考文档的序号标注在相关回答之后
            3. 如果你认为参考文档无法帮助你回答这个问题，请回答“我还不了解这个问题的答案”
            4. 不要输出与问题不相关的文字
            尝试回答这个问题
        """
    
    def _proceed_questions(self, prompt: str) -> Union[Tuple[List, bool], Tuple[str, bool]]:
        self._proceed_questions_system_input = self._proceed_questions_system_input.replace("{k}",str(self.k))
        response = self.model.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": self._proceed_questions_system_input},
                {'role': 'user', "content": f'问题：{prompt}'}
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
    def test_proceed_questions(self, prompt):
        return self._proceed_questions(prompt=prompt)
    
    def chat(self, prompt: str, method: str = 'desc'):
        pass
    

if __name__ == "__main__":
    llm = ChatGPT(model)
    response = llm.test_proceed_questions("特朗普的生日是？")
    print(response)