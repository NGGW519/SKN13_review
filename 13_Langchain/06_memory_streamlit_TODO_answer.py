############################################################################
#  streamlit/06_memory_streamlit_TODO_answer.py
############################################################################
# Streamlit으로 간단한 UI를 구성하고
# RunnableWithMessageHistory를 통해 LangChain LLM 체인에 세션별 대화 이력 저장 기능을 붙임
# 사용자의 입력/응답을 SQLite DB에 자동으로 저장하면서 스트리밍으로 응답도 제공함

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine
from langchain_core.runnables import RunnableWithMessageHistory


# LLM 모델과 프롬프트를 정의하고 cache에 저장
@st.cache_resource  # cache에 올리겠다는 뜻. cache에 올라가 있으니 rerun해도 이 함수를 실행하지 않음. cache에 새로운 내용 올릴려면 streamlit을 껐다켜야함.
def get_llm_model():
    load_dotenv()   # .env 파일 로드 (API 키 등)
    # OpenAI 모델 인스턴스 생성
    model = ChatOpenAI(model_name = "gpt-4o-mini")
    # prompt template 정의
    prompt_template = ChatPromptTemplate(
    messages = [
        ("system", "답변을 100단어 이내로 작성해줘"),
        MessagesPlaceholder(variable_name = "history", optional = True),    # 대화 이력
        ("user", "{query}")    # 사용자 입력
        ]
    )
    # 프롬프트 -> 모델 -> 문자열 출력 파서 연결
    return prompt_template | model | StrOutputParser()

# RunnableWithMessageHistory 인스턴스를 생성하고 cache에 저장
@st.cache_resource
# RunnableWithMessageHistory를 생성해서 반환
def get_chain():
    # 위에서 만든 model + prompt
    runnable = get_llm_model()
    # SQLite 엔진 생성 (로컬 DB)
    engine = create_engine("sqlite:///chat_history.sqlite")
    # RunnableWithMessageHistory로 체인 구성
    chain = RunnableWithMessageHistory(
        runnable=runnable,
        get_session_history = lambda session_id : SQLChatMessageHistory(session_id = session_id, connection = engine),
        input_messages_key= "query",    # 사용자 입력 키
        history_messages_key= "history" # 히스토리 키
    )
    return chain


model = get_chain()   # cache에 올려져 있는걸 사용, chain이 넘어옴.

st.title("무엇이든지 물어보세용~ 아는것만 알려드려용")


# Session State 생성
## session_state : dictionary 구현체, 시작 ~ 종료할 때 까지 사용자 별로 유지되어야 하는 값들을 저장하는 곳

# 0. 대화 내욕을 session_state의 "messages" : list 로 저장
# 1. session state에 messages key 조회(없으면 생성)
if "messages" not in st.session_state:   # return T/F
    st.session_state["messages"] = []    # 대화 내용들을 저장할 list를 "messages" 키로 저장


# session_state에서 session_id를 조회, 없으면 빈 상태값을 저장.
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None


# Sidebar에 session_id 입력 위젯 생성
session_id = st.sidebar.text_input("Session ID", placeholder="ID를 입력하세요.")


# 기존 대화 이력 출력
message_list = st.session_state["messages"] # 변수로 저장.
for message in message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])    # 출력


# user의 prompt를 입력 받는 위젯
prompt = st.chat_input("User Prompt")    # user가 입력한 문자열을 반환.


## 대화 작업
# 사용자가 입력한 경우
if prompt is not None:
    # session_state의 messages에 대화 내역을 저장
    st.session_state["messages"].append({"role" : "user", "content" : prompt})
    # 사용자 입력을 화면에 표시
    with st.chat_message("user"):
        st.write(prompt)
    # session_id가 아직 없다면 입력값으로 설정
    if st.session_state["session_id"] is None:
        st.session_state["session_id"] = session_id

    # 세션 ID를 RunnableWithMessageHistory에 넘겨주기 위한 설정
    config = {"configurable" : {"session_id" : st.session_state["session_id"]}}

    # AI 응답 영역
    with st.chat_message("ai"):
        message_placeholder = st.empty()    # update가 가능한 container
        full_message = ""   # LLM이 응답하는 토큰들을 저장할 문자열 변수.
        # stream()으로 실시간 토큰 출력
        for token in model.stream({"query" : prompt}, config = config):
            full_message += token   # 토큰을 누적
            message_placeholder.write(full_message) # 기존 내용을 full_message로 갱신.
        # AI 응답도 session_state에 저장
        st.session_state["messages"].append({"role" : "ai", "content" : full_message})