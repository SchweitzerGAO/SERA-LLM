from googlesearch import search
from typing import List, Union
from duckduckgo_search import DDGS

GOOGLE_SEARCH_KEY = 'AIzaSyCeCxFvKd-4FUaVj8AI45zQNqSYgd4UPpo'

ddgs = DDGS(proxies="http://localhost:7890")

def desc_as_doc_search(query: Union[List, str], 
                  num_results_single, # number of results for a single query
                  num_results_multi, # number of results for multiple queries
                  lang: str,
                  backend: str,
                  advanced: bool = True,
                ):
    """
    The 'description' field in the search result are deemed directly as the reference docs
    """
    documents = []
    if isinstance(query, List):
        for q in query:
            if backend == 'google':
                results = search(q, 
                             num_results=num_results_multi,
                             lang=lang, 
                             advanced=advanced)
                for result in results:
                    doc = result.description.replace("\xa0...","").replace("...","")
                    if '—' in doc:
                        doc = doc.split('—')[1]
                    documents.append(doc)
            elif backend == 'ddg':
               results = ddgs.text(q, max_results=num_results_multi)
               for result in results:
                   doc = result['body'].replace('...','').strip()
                   documents.append(doc)

            else:
                raise NotImplementedError
    
    else:
        if backend == 'google':
            results = search(query, 
                         num_results=num_results_single, 
                         lang=lang, 
                         advanced=advanced)
            for result in results:
                doc = result.description.replace("\xa0...","").replace("...","")
                if '—' in doc:
                    doc = doc.split('—')[1]
                documents.append(doc)
        elif backend == 'ddg':
            results = ddgs.text(query, max_results=num_results_single)
            for result in results:
                doc = result['body'].replace('...','').strip()
                documents.append(doc)
        else:
            raise NotImplementedError

    return documents

def page_as_doc_search(
                  query: Union[List, str], 
                  num_results_single: int, # number of results for a single query
                  num_results_multi: int, # number of results for multiple queries
                  lang: str,
                  backend: str,
                  advanced: bool = True):
    """
    Store webpages in the result in a vector database using LangChain(or llama index), which is deemed as reference docs.
    And the LLM will retrieve again in the database to get a answer more accurate
    """
    raise NotImplementedError
            



"""
Test code below
""" 
# docs = desc_as_doc_search("唐宋八大家都是谁", 
#                           num_results_single=10,
#                           num_results_multi=2,
#                           lang='zh-CN')
# # # docs = desc_as_doc_search(["妙玉在《红楼梦》中的人物形象如何反映了中国传统文化中对女性角色的理解与塑造？",
# # #                       "妙玉的性格特点和行为举止如何在小说中得到体现？这些特点如何与她的家庭背景和成长环境相关联？",
# # #                       "《红楼梦》中，妙玉与其他女性角色的关系是怎样的？她与宝玉、黛玉等其他主要人物之间的互动如何揭示了她的个性与情感状态？"])

# print(docs)