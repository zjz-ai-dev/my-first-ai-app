import streamlit as st
import os
import requests
import json
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import HuggingFaceEmbeddings

# 设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

st.set_page_config(page_title="专属 AI 助理", layout="wide")
st.title("🧠 专属 AI 超级助理")
st.caption("基于 OOP 架构与 LangChain RAG 技术打造")

class WeatherTool:
    def __init__(self):
        self.name = "get_weather"
        self.description = "查询南宁的真实天气情况"
    def execute(self, args):
        try:
            url = "https://api.open-meteo.com/v1/forecast?latitude=22.82&longitude=108.32&current_weather=true"
            res = requests.get(url, timeout=5)
            data = res.json()
            temp = data["current_weather"]["temperature"]
            return f"南宁当前气温 {temp}°C"
        except Exception as e:
            return f"查询失败：{e}"

class AdvancedKnowledgeTool:
    def __init__(self, pdf_path, api_key):
        self.name = "search_knowledge"
        self.description = "查询上传的PDF知识库获取文档信息"
        self.llm = ChatDeepSeek(model="deepseek-chat", api_key=api_key, temperature=0)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        self.splits = text_splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(self.splits, self.embeddings)

    def execute(self, args):
        query = args.get("query", "")
        if not query: return "未提供查询内容。"
        docs = self.vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"请仅根据以下上下文回答，没有提到请说“不知道”。\n【上下文】：{context}\n【问题】：{query}"
        try:
            return self.llm.invoke(prompt).content
        except Exception as e:
            return f"查询失败：{e}"

class SuperAgent:
    def __init__(self, api_key, pdf_path=None):
        self.api_key = api_key
        self.url = "https://api.deepseek.com/chat/completions"
        self.tools = [WeatherTool()]
        if pdf_path:
            self.tools.append(AdvancedKnowledgeTool(pdf_path, api_key))

    def get_tools_description(self):
        desc = "你可以使用以下工具。如果需要，请仅输出 JSON：{\"tool\": \"工具名\", \"args\": {\"query\": \"问题\"}}\n"
        for t in self.tools:
            desc += f"- {t.name}: {t.description}\n"
        return desc

    def execute_tool(self, tool_name, args):
        for t in self.tools:
            if t.name == tool_name:
                return t.execute(args)
        return "未找到工具"

    def chat(self, user_message):
        system_prompt = f"你是一个智能助手。{self.get_tools_description()}\n【绝对铁律】：遇到需要查天气或查文档时，必须仅输出 JSON，禁止输出废话。"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json; charset=utf-8"}
        
        try:
            data = {"model": "deepseek-chat", "messages": messages}
            response = requests.post(self.url, headers=headers, data=json.dumps(data, ensure_ascii=False).encode('utf-8'))
            ai_reply = response.json()["choices"][0]["message"]["content"]

            if "{" in ai_reply and "tool" in ai_reply:
                start, end = ai_reply.find("{"), ai_reply.rfind("}") + 1
                tool_json = json.loads(ai_reply[start:end])
                tool_result = self.execute_tool(tool_json.get("tool"), tool_json.get("args", {}))
                
                messages.append({"role": "assistant", "content": ai_reply})
                messages.append({"role": "user", "content": f"工具结果：{tool_result}。请回答用户。"})
                data = {"model": "deepseek-chat", "messages": messages}
                final_res = requests.post(self.url, headers=headers, data=json.dumps(data, ensure_ascii=False).encode('utf-8'))
                return final_res.json()["choices"][0]["message"]["content"]
            return ai_reply
        except Exception as e:
            return f"系统错误：{e}"

# ---------------- Streamlit 界面交互 ----------------
# ---------------- Streamlit 界面交互 ----------------
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input("请输入 DeepSeek API Key", type="password")
    uploaded_file = st.file_uploader("上传你的 PDF 知识库", type=["pdf"])
    
    # 🌟 核心改动：必须点按钮，才去建库
    if st.button("加载 PDF 知识库"):
        if uploaded_file is not None and api_key:
            with st.spinner("正在解析 PDF 并建立向量库..."):
                # 每次上传新文件，强制生成一个唯一的新文件名，绝不覆盖旧文件
                temp_path = f"doc_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 把生成的路径存到浏览器的全局内存里
                st.session_state["pdf_path"] = temp_path
                st.success(f"✅ {uploaded_file.name} 加载成功！")
        else:
            st.warning("请先输入 API Key 并上传 PDF！")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请向我提问..."):
    if not api_key:
        st.warning("请先在左侧输入 API Key！")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                # 从内存里读取刚才加载的 PDF 路径
                current_pdf = st.session_state.get("pdf_path", None)
                
                # 如果内存里没有 PDF，就不传，当纯聊天机器人
                if current_pdf:
                    agent = SuperAgent(api_key, current_pdf)
                else:
                    agent = SuperAgent(api_key, None)
                    
                response = agent.chat(prompt)
                st.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})