
from langchain_pinecone import PineconeVectorStore  
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings

# AI 응답 생성 함수 
def get_ai_messgae(user_message):
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(embedding = embedding, index_name=index_name)
    
    llm = ChatOpenAI(model="gpt-4o")
    prompt = hub.pull("rlm/rag-prompt")
    retriever = database.as_retriever(search_kwargs={"k" : 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt" : prompt})
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

# 사전 기반 질문 보정
    prompt = ChatPromptTemplate.from_template(f""" 
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요.
    
        사전 : {dictionary}
    
        질문 : {{question}}
    
        """)

    dictionary_chain = prompt | llm | StrOutputParser()
    
    # 최종 파이프라인
    tax_chain = {"query": dictionary_chain} | qa_chain 
    ai_message = tax_chain.invoke({"question" : user_message})
    return ai_message['result']