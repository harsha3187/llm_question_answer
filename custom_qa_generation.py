import torch
from langchain.document_loaders import DirectoryLoader,AsyncChromiumLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import WikipediaLoader
import nest_asyncio
from peft import PeftModel, PeftConfig
from transformers import T5ForConditionalGeneration,Seq2SeqTrainingArguments,Seq2SeqTrainer,AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, T5Tokenizer , DataCollatorForSeq2Seq
import warnings
warnings.filterwarnings("ignore")


class qa_llm:
    
    '''
    generating the answers based on the input using the docs storage and fine tuned LLM
    
    '''
    def __init__(self):
        #loading llm
        self.tokenizer , self.peft_model  = self.load_llm()
        self.retriever = self.get_retriver()
        pass
    
    def load_llm(self):
        #loading the base
        peft_model_base = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", torch_dtype=torch.bfloat16)
        #loading the pre trained tokenizer
        tokenizer = T5Tokenizer.from_pretrained('./peft-qa-model-local2/')
        #loading the fine tuned model on Q&A
        peft_model = PeftModel.from_pretrained(peft_model_base, 
                                               './peft-qa-model-local2/', 
                                               torch_dtype=torch.bfloat16,
                                               is_trainable=False)

        return tokenizer , peft_model
    
    # create a vector db from the transcripts
    def create_vector_db(self):
        # we use sentence transformer to get the vector embeddings for the database
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        nest_asyncio.apply()

        #articles to scrape
        articles = ["https://www.dubaichamber.com/en/home/",
                    "https://www.dubaichamber.com/en/services/",
                    "https://www.dubaichamberdigital.com/en/home/",
                    "https://www.dubaichamberinternational.com/en/home/",
                    "https://www.dubaichamberinternational.com/en/offices/india/",
                   "https://www.dubaichamber.com/en/about-us/our-history/"]

        #wiki articles to scrape
        wiki_names = ["Dubai_Chamber_of_Commerce_and_Industry"]

        # Scrapes the blogs and articles above
        wiki_docs = WikipediaLoader(query=wiki_names, load_max_docs=2).load()

        loader = AsyncChromiumLoader(articles)
        docs = loader.load()

        all_docs = docs + wiki_docs

        # Converts HTML to plain text 
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(all_docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs_transformed)

        # cache the embeddings for faster loadup
        fs = LocalFileStore("./cache/")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            hf, fs, namespace="sentence"
        )

        # create the vector db
        db = FAISS.from_documents(documents, cached_embedder)

        db.save_local("faiss_index")
        return db
    
    def get_embeddings(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return hf
    
    def get_prediction_and_scores(self,prompt):
        #return the predictions based on the prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs =  self.peft_model.generate(input_ids = input_ids, generation_config = GenerationConfig(output_scores=True, return_dict_in_generate=True, max_length=1000))
        generated_sequence = outputs.sequences[0]

        # get the probability scores for each generated token
        transition_scores = torch.exp(self.peft_model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )[0])
        return self.tokenizer.decode(generated_sequence), generated_sequence, transition_scores
    
    def get_retriver(self,threshold=0.6):
        #loading the FAISS vector storage index
        db2 = FAISS.load_local("faiss_index", self.get_embeddings())
        # loading the retriver to get the similarity matches
        retriever = db2.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": threshold})
        return retriever
    
    def generate_text(self,input_text):
        # keep generating tokens until we get a </s> token
        #calling the retriver 
        docs = self.retriever.get_relevant_documents(input_text)
        context = "\n".join([doc.page_content for doc in docs])
        if context:
            new_input_text = f"Given the below context:\n{context}\n\n explain and answer the question: \n{input_text}\n"
        else:
            new_input_text = f"explain and answer the question: \n{input_text}\n"
        # get the prediction and scores from the LLM, given the new input
        generated_sequence, _, _ = self.get_prediction_and_scores(new_input_text)

        # print the final output
        return generated_sequence.replace('<pad>','').replace('</s>','')