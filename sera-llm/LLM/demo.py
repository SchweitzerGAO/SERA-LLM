from chatgpt import RagChatGPT, model
raw_system_input = """
    你是一个诚实的问答助手，以下是用户的问题，请尝试回答这个问题，注意：
    1. 如果你不能直接回答这个问题，请回答“我还不了解这个问题的答案”
"""
query_list = [
    '唐宋八大家中，唐朝的都有谁？',
    '杜甫的《春望》表达了作者的什么情感？',
    '《红楼梦》中妙玉是一个什么样的人？',
    '简单介绍一下大模型微调所用的LoRA技术',
    '电影《第二十条》主要讲了什么？'
]
def demo_chatgpt_desc():
    llm = RagChatGPT(model=model)
    for i, q in enumerate(query_list):
        output_raw = model.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": raw_system_input},
                {'role': 'user', "content": f"问题：{q}"}
            ]
            ).choices[0].message.content
        output_rag = llm.chat(prompt=q)
        docs = llm.docs
        with open('../../demos/chatgpt-desc-3.txt','a',encoding='utf-8') as f:
            f.write(f"###############\nQuestion{i + 1}: {q}\n###############\n")
            f.write('-----------------------------------------\n')
            f.write("Raw ChatGPT\n")
            f.write(f"Output:\n{output_raw}\n")
            f.write('-----------------------------------------\n')
            f.write("SERAG ChatGPT\n")
            f.write(f"References:\n{docs}")
            f.write(f"Output\n{output_rag}\n")
            f.write('-----------------------------------------\n')
            f.write('\n')
        print(f'Query {i + 1} finished')

demo_chatgpt_desc()