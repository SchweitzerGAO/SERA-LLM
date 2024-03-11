from typing import List, Union, Tuple
from utils.search import desc_as_doc_search, page_as_doc_search
class RagBaseLLM:

    def __init__(self, 
                 model, 
                 k, 
                 lang,
                 rewrite_method,
                 retrieve_backend,
                 read_method) -> None:
        """
        model: The LLM to interact with
        k: The number of questions to generate by the LLM
        lang: The language in which the user is querying
        rewrite_method: The method to rewrite the query
                        Supported methods codes and corresponding methods are:
                        1. hyqr: Hypothetical Query Retrieve
                        2. hykr: Hypothetical Keyword Retrieve
                        3. r3: Rewrite Retrieve Read(To be implemented)
        retrieve_backend: The search engine to retrieve
                        Supported backend codes and corresponding backends are:
                        1. google: Google
                        2. ddg: DuckDuckGo
        read_method: The method of reading the 
                        Supported method codes and corresponding methods are:
                        1. description: Description-as-Document, the 'description' field in the search result are deemed directly as the reference docs
                        2. page: Page-as-Document, store webpages in the result in a vector database using LangChain(or llama index), which is deemed as reference docs.
    And the LLM will retrieve again in the database to get a answer more accurate
        """

        self.model = model
        self.k = k
        self.lang = lang
        self.rewrite_method = rewrite_method
        self.retrieve_backend = retrieve_backend
        self.read_method = read_method

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
            lang=self.lang,
            backend=self.retrieve_backend
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
                          num_results_single: int, # number of results for a single query
                          num_results_multi:int, # number of results for multiple queries
                          ):
        raise NotImplementedError


    
    def chat(self, 
             prompt: str, 
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