import streamlit as st
import os
import requests
import json
# 这里先写个假架子，下一把你昨天的类粘贴进来

st.set_page_config(page_title="我的专属 AI 助理", layout="wide")

st.title("🧠 专属 AI 超级助理")
st.caption("基于 OOP 架构与 LangChain RAG 技术打造")

# 侧边栏：用来放上传文件和 API Key 设置
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input("请输入 DeepSeek API Key", type="password")
    uploaded_file = st.file_uploader("上传你的 PDF 知识库", type=["pdf"])
    
    if uploaded_file is not None and api_key:
        if st.button("初始化知识库"):
            # 这里未来写把 PDF 塞给 AdvancedKnowledgeTool 的逻辑
            st.success("知识库加载成功！")

# 主界面：聊天窗口
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请向我提问..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 这里未来接入你的 SuperAgent
        response = "我收到你的问题了！但我的大脑（SuperAgent）还没接进来。"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})