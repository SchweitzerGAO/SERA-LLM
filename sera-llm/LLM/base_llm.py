from typing import List, Union, Tuple
from utils.search import desc_as_doc_search, page_as_doc_search
class RagBaseLLM:

    def __init__(self, model, k=3) -> None:
        """
        model: The LLM to interact with
        k: The number of questions to generate by the LLM
        """
        self.model = model
        self.k = k

        self._process_questions_system_input = None
        self._process_questions_system_input_keyword = None
        self._process_questions_user_input = None
        self._final_system_prompt = None
        self._final_user_prompt = None
        self._process_questions_user_input_cache = None
        self._final_user_prompt_cache = None

        self.docs = None
    
    def _process_questions(self, prompt: str) -> Union[Tuple[List, bool], Tuple[str, bool]]:
        raise NotImplementedError
    
    
    def _get_desc_as_doc(
                  self,
                  docs: List[str],
                  query: Union[List, str], 
                  num_results_single: int = 10, # number of results for a single query
                  num_results_multi: int = 3 # number of results for multiple queries
    ):
        docs.extend(
            desc_as_doc_search(
            query=query,
            num_results_single=num_results_single,
            num_results_multi=num_results_multi
        )
        )
        return docs
    def _get_page_as_doc(self, 
                         query: Union[List, str], 
                         num_results_single: int = 10,
                         num_results_multi: int = 3
                         ):
        return page_as_doc_search(
            query=query,
            num_results_single=num_results_single,
            num_results_multi=num_results_multi,
        )
        
    
    def _get_final_user_prompt(self, 
                          prompt: str, 
                          method: str = "description", # 'description' or 'page'
                          num_results_single: int = 10, # number of results for a single query
                          num_results_multi:int = 3, # number of results for multiple queries
                          ):
        raise NotImplementedError


    
    def chat(self, prompt: str, method: str = 'desc'):
        raise NotImplementedError