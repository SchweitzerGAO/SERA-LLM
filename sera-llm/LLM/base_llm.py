from typing import List, Union, Tuple
from utils.search import desc_as_doc_search, page_as_doc_search
class RagBaseLLM:

    def __init__(self, 
                 model, 
                 k, 
                 lang, 
                 rewrite_method) -> None:
        """
        model: The LLM to interact with
        k: The number of questions to generate by the LLM
        lang: The language in which the user is querying
        rewrite method: The method to rewrite the query:
                        Supported methods code and corresponding methods are:
                        1. hyqr: Hypothetical Query Retrieve
                        2. hykr: Hypothetical Keyword Retrieve
                        3. r3: Rewrite Retrieve Read(To be implemented)
                        4. sp: Step-back Prompt(To be implemented)
        """
        self.model = model
        self.k = k
        self.lang = lang
        self.rewrite_method = rewrite_method

        self._process_questions_system_input = None
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
                  num_results_single: int, # number of results for a single query
                  num_results_multi: int # number of results for multiple queries
    ):
        docs.extend(
            desc_as_doc_search(
            query=query,
            num_results_single=num_results_single,
            num_results_multi=num_results_multi,
            lang=self.lang
        )
        )
        return docs
    def _get_page_as_doc(self, 
                         query: Union[List, str], 
                         num_results_single: int,
                         num_results_multi: int
                         ):
        return page_as_doc_search(
            query=query,
            num_results_single=num_results_single,
            num_results_multi=num_results_multi,
        )
        
    
    def _get_final_user_prompt(self, 
                          prompt: str, 
                          method: str, # 'description' or 'page'
                          num_results_single: int, # number of results for a single query
                          num_results_multi:int, # number of results for multiple queries
                          ):
        raise NotImplementedError


    
    def chat(self, 
             prompt: str, 
             method: str,
             num_results_single: int, # number of results for a single query
             num_results_multi: int
       ):
        raise NotImplementedError
    
    def _get_final_user_prompt_hotpot(self, 
                                      prompt: str, 
                                      docs_list: List[str],
                                      ):
        raise NotImplementedError
    
    def evaluation_hotpot(self, 
                    prompt: str,  
                    docs_list: List[str] | None,
                    method: str,
                    num_results_single: int, # number of results for a single query
                    num_results_multi: int  # number of results for multiple queries
                    ):
        raise NotImplementedError