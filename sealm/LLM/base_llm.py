from typing import List, Union, Tuple
from utils.search import desc_as_doc_search
class BaseLLM:

    def __init__(self, model, k=3) -> None:
        """
        model: The LLM to interact with
        k: The number of questions to generate by the LLM
        """
        self.model = model
        self.k = k
        self._proceed_questions_system_input = None
        self._final_prompt = None
    
    def _proceed_questions(self, prompt: str) -> Union[Tuple[List, bool], Tuple[str, bool]]:
        raise NotImplementedError
    
    
    def _get_desc_as_doc(
                  self,
                  query: Union[List, str], 
                  num_results_single=10, # number of results for a single query
                  num_results_multi=2, # number of results for multiple queries
    ):
        return desc_as_doc_search(
            query=query,
            num_results_single=num_results_single,
            num_results_multi=num_results_multi
        )
    
    def _get_final_prompt(self, 
                          prompt: str, 
                          method: str = "desc" # 'desc' or 'page'
                          ):
        query = self._proceed_questions(prompt=prompt)
        if method == 'desc':
            docs_list = self._get_desc_as_doc(query=query)
            self._final_prompt = self._final_prompt.replace("{question}", prompt)
            docs = ''
            for i, d in enumerate(docs_list):
                docs += f'[{i}]：{d}\n'
            self._final_prompt = self._final_prompt.replace("{docs}",docs)
        elif method == 'page':
            pass
        else:
            raise NotImplementedError


    
    def chat(self, prompt: str, method: str = 'desc'):
        raise NotImplementedError