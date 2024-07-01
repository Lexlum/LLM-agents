""" DELTAI FELLOWSHIP : PROJECT 1 - AI CHATBOT APPLICATION """

#######################################################################################################

# """ NECESSARY IMPORTS """
import logging
import pdb

import streamlit as st # alias 
import openai
import pandas as pd
import pdb

import transformers
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # Facebook library for similarity search of text
from langchain_community.llms import OpenAI #langchain.OpenAI is just a wrapper & openAI not belong to langchain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from langchain_experimental.agents import create_pandas_dataframe_agent

from PyPDF2 import PdfReader
import time
from dotenv import load_dotenv
import os

from transformers import pipeline

os.environ['OPENAI_API_KEY'] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
#####################################################################################################

# To view app on browser : streamlit run main.py --browser.serverAddress localhost
# First click the play button and wait till ' streamlit run c:/.../....> ' appears
# Then enter command given above ( everything starting from streamlit to localhost )


# Load environment variable and assign as API key
load_dotenv()

#####################################################################################################

# """ MAIN FUNCTION FOR CREATING NAVBAR AND LOADING DIFFERENT PAGES """

def main():

    # All design and structural elements occupy wider area 
    st.set_page_config('wide')

    # Create sidebar - On button click call function corresponding to that page
    st.sidebar.title("导航")
    pages = {
        "Home": homepage,
        "My Chatbot": chatbot_page,
        "Article Generator": article_generator,
        "ChatCSV": chat_csv,
        "ChatPDF": chat_pdf,
        "DALL-E" : image_generator
    }

    selected_page = st.sidebar.button("主页面", key="home",use_container_width=True,type='primary')
    if selected_page:
        # URL in the browser's address bar will be updated to include the query parameter 'page=home'
        st.experimental_set_query_params(page="home")
    selected_page = st.sidebar.button("聊天机器人", key="chatbot",use_container_width=True)
    if selected_page:
        st.experimental_set_query_params(page="chatbot")
    selected_page = st.sidebar.button("文章生成器", key="seo_article",use_container_width=True)
    if selected_page:
        st.experimental_set_query_params(page="seo_article",)
    selected_page = st.sidebar.button("CSV对话Agent", key="chatcsv",use_container_width=True)
    if selected_page:
        st.experimental_set_query_params(page="chatcsv")
    selected_page = st.sidebar.button("PDF对话Agent", key="chatpdf",use_container_width=True)
    if selected_page:
        st.experimental_set_query_params(page="chatpdf")
    selected_page = st.sidebar.button("DALL-E图片生成器", key="dall_e",use_container_width=True)
    if selected_page:
        st.experimental_set_query_params(page="dall_e")

    # Get the page name from the URL, default to "home"
    page_name = st.experimental_get_query_params().get('page', ['home'])[0]

    # Call the corresponding page based on the selected page_name
    if page_name == "home":
        homepage()
    elif page_name == "chatbot":
        chatbot_page()
    elif page_name == "seo_article":
        article_generator()
    elif page_name == "chatcsv":
        chat_csv()
    elif page_name == "chatpdf":
        chat_pdf()
    elif page_name == "dall_e":
        image_generator()

##################################################################################################

# """ HOMEPAGE WITH DESCRIPTIONS OF VARIOUS TOPICS RELATED TO LLM, ChatGPT, ETC. """

def homepage():

    # Custom CSS for homepage spefically for content containers
    st.markdown(
        """
        <style>
        .homepage-subheading {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .gpt-example-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 2rem;
        }
        .gpt-example-box {
            width: 45%;
            padding: 1rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            background-size: 200% 100%;
            animation: gradientAnimation 3s linear infinite;
        }
        .gpt-example-box:nth-child(1) {
            background-image: linear-gradient(45deg, #1E90FF 0%, #4682B4 100%);
        }
        .gpt-example-box:nth-child(2) {
            background-image: linear-gradient(45deg, #32CD32 0%, #228B22 100%);
        }
        .gpt-example-box:nth-child(3) {
            background-image: linear-gradient(45deg, #9370DB 0%, #6A5ACD 100%);
        }
        .gpt-example-box:nth-child(4) {
            background-image: linear-gradient(45deg, #FFA500 0%, #FF8C00 100%);
        }
        .gpt-example-box p {
            font-size: 16px;
            margin: 0;
            color: white;
        }
        .gpt-example-box h3 {
            font-size: 20px;
            margin-bottom: 0.5rem;
            color: white;
        }

        @keyframes gradientAnimation {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Page title
    st.title("多模态Agent工作流")

    # Section 1 : LLM Versions of ChatGPT
    st.markdown('<div class="homepage-subheading">不同版本的ChatGPT</div>', unsafe_allow_html=True)
    st.write("在这里，您可以找到ChatGPT使用过的语言模型（LLM）的描述。每个模型都具有独特的能力和特征。")

    # Content boxes with version name and description
    with st.container():
        st.markdown(
            """
            <div class="gpt-example-container">
                <div class="gpt-example-box">
                    <h3>GPT-3</h3>
                    <p>这是ChatGPT创建过程中首个使用的版本，它经过大量文本数据训练，能够预测给定序列中的下一个词。它在回答问题和生成文本等任务上表现出色，特别擅长与聊天机器人功能核心相关的自然语言处理任务。
                    </p>
                </div>
                <div class="gpt-example-box">
                    <h3>GPT-3.5</h3>
                    <p>当ChatGPT于2022年11月30日发布供公众使用时，它运行的是基于GPT-3.5系列进行微调的模型，这是从原始GPT-3改进而来的模型。GPT-3.5版本能够涉及各种主题，包括编程、电视剧本和科学概念。</p>
                </div>
                <div class="gpt-example-box">
                    <h3>GPT-3.5-Turbo</h3>
                    <p>这个版本有1750亿个参数，比GPT-3.5显著多。通过这种额外的复杂性，GPT-3.5 Turbo可以执行编写和调试计算机程序、创作音乐、生成商业思路以及模拟Linux系统等任务。</p>
                </div>
                <div class="gpt-example-box">
                    <h3>GPT-4</h3>
                    <p>这个版本有超过一万亿个参数，是一个多模型，可以接受文本和图像输入。它具有更长的记忆力（高达64,000个词），改进的多语言能力，更多对响应的控制，并且有限的搜索能力，当在提示中分享URL时，可以从网页中提取文本。它还引入了与插件一起工作的能力，允许第三方开发者使ChatGPT-4变得“更智能”。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section 2 : Special parameters in OpenAI API requests
    st.markdown('<div class="homepage-subheading">OpenAI API请求中的关键术语</div>', unsafe_allow_html=True)
    st.write("在OpenAI Playground中，这些参数对生成的输出有重大影响，请尝试进行一些实验。")

    with st.container():
        st.markdown(
            """
            <div class="gpt-example-container">
                <div class="gpt-example-box">
                    <h3>Temperature</h3>
                    <p>温度控制模型输出的随机性。例如，较高的值如0.8会使输出更加多样化，而较低的值如0.2则会使其更加集中和确定性。</p>
                </div>
                <div class="gpt-example-box">
                    <h3>Top-p</h3>
                    <p>Top-p（核心）采样将模型的输出截断为累积超过给定概率阈值（例如0.9）的最可能的标记。这可以防止模型生成过于罕见或荒谬的标记。</p>
                </div>
                <div class="gpt-example-box">
                    <h3>Presence Penalty</h3>
                    <p>Presence Penalty（存在惩罚）用于阻止模型在输出中生成特定的标记。通过添加存在惩罚，可以避免获取特定类型的响应。</p>
                </div>
                <div class="gpt-example-box">
                    <h3>Frequency Penalty</h3>
                    <p>Frequency Penalty（频率惩罚）控制模型响应中重复内容的数量。较高的值如2.0可以减少重复行为，而较低的值如0.2则允许更多重复。</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Section 3 : Other LLM Models
    st.markdown('<div class="homepage-subheading">其他大语言模型</div>', unsafe_allow_html=True)
    st.write("这些是由其他公司和研究机构开发的几个其他LLM模型的描述。")
    
    with st.container():
        st.markdown(
            """
            <div class="gpt-example-container">
                <div class="gpt-example-box">
                    <h3>PaLM 2 (Bison-001) by Google:</h3>
                    <p>这个模型是谷歌的PaLM 2系列的一部分，专注于常识推理、形式逻辑、数学和超过20种语言中的高级编码。它训练了5400亿个参数，最大上下文长度为4096个标记。它还是一个多语言模型，能够理解成语、谜语以及来自不同语言的微妙文本。</p>
                </div>
                <div class="gpt-example-box">
                    <h3>Falcon by the Technology Innovation Institute (TII), UAE</h3>
                    <p>这是一个开源的LLM模型，经过了训练，拥有40亿个参数（Falcon-40B-Instruct模型）。它主要在英语、德语、西班牙语和法语中进行了训练，但也可以在意大利语、葡萄牙语等其他几种语言中工作。Falcon的开源性质使其可以在商业用途中无限制地使用。</p>
                </div>
                <div class="gpt-example-box">
                    <h3>RoBERTa by Facebook</h3>
                    <p>RoBERTa是由Facebook开发的BERT的变体，采用了不同的训练方法。它在更大量的数据上进行训练，使用更大的批次和更长的序列，在训练过程中移除了BERT使用的下一句预测任务。这些改变使得RoBERTa在多个基准任务上表现优于BERT。</p>
                </div>
                <div class="gpt-example-box">
                    <h3>Vicuna 33B by LMSYS</h3>
                    <p>Vicuna是从LLaMA衍生出的开源LLM模型。它训练了33亿个参数，并使用监督指导进行了微调。尽管与某些专有模型相比规模较小，Vicuna展现出了显著的性能。</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

#########################################################################################################

# """ STAR FEATURE : CHATBOT APPLICATION USING OPENAI API"""

def chatbot_page():

    st.title("欢迎来到Multimodel-Agent工作流")
    col1,col2,col3 = st.columns(3) # Create 3 seperate columns for structure and alignment
    options = ["默认", "古怪", "傲慢",'睿智']

    with col2:
        # Default + Unique personalities for user to experiment with
        personality = st.selectbox("人格选项", options,label_visibility='collapsed',index=options.index('默认'))
    with col3:
        submit_button1 = st.button('选择人格',type='primary',use_container_width=True)
    with col1:
        clear_button = st.button('清除聊天',use_container_width=True,type='primary')

    # Assign AI Chatbot role/personality
    content = ""
    if submit_button1:
        if personality=='古怪':
            content ="""->->->You are now a funky 
                        freak personality, make jokes, dark humour, add cringe statements. Have a skaterboard vibe
                        add weird crazy comments, freak out with panic attacks and keep making hilarious joked, puns and talk about memes
                        Reply with a greeting to the user embodying this personality"""
        elif personality=='傲慢':
            content = """->->->You are now an extremely sassy personality, boast, make sassy remarks, offer unwanted advice.
                        Keep focusing on yourself, praise yourself, make excuses and be very judgemental. Make comments,
                        and just be extremely SASSSYYY!! Reply with a greeting to the user embodying this personality"""
        elif personality=='睿智':
            content = """->->->You are now an extremely sarcastic yet wise personality. Be extremeley philosophical, keep branching out
                        into conversations about ethical dilemnas and the purpose of life. Warn the user of the future and their role in life. Be sarcastic yet 
                        prophesize about the world and give advice to the user. Reply with a greeting to the user embodying this personality"""
        else:
            content = """->->->You are an AI chatbot and your goal is to answer user queries. You have a neutral personality.
                        """
        temperature = 0.7
        st.session_state.messages.append({"role": "user",'content':content})

        # Add a standard assistant reply so openAI knows in message history this is the personality
        st.session_state.messages.append({"role": "assistant",
                                           "content": '->->->I have undertood and will answer all upcoming messages through this personality specifically and will reply embodying the personality described by you'})

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    colx,coly = st.columns(2)
    with colx:
        # Model version ( Legacy + other models are deprecated/inaccessible )
        st.session_state['openai_model'] = st.selectbox("",['gpt-3.5-turbo','gpt-3.5-turbo-16k','gpt-4'], label_visibility='collapsed',index=0)
    with coly:
        # Temperature affects how random or standard the responses are 
        # Ex : The cat sits on the ____ (0.5 - mat, 1.7 - windowsill)
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, step=0.1, value=0.7,format="Temp : %f", label_visibility='collapsed')


    if "messages" not in st.session_state:
        st.session_state.messages = [] # Create message history as empty list

    if clear_button:
        st.session_state.messages = [] # Clear history
        st.session_state.messages.append({"role": "user", "content": '->->->You role now is to simply be an extremely helpful AI chatbot assistant and answer user queries. Keep no personality and be neutral'})
        st.session_state.messages.append({"role": "assistant",
                                           "content": '->->->I have undertood and will answer all upcoming messages through this personality specifically and will reply embodying the personality described by you'})

    # If messages pertain to the personality selection, don't display - better user experience
    for message in st.session_state.messages:
        if not message['content'].startswith('->->->'): # ->->-> special sequences to identify personality message
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # If user prompt is not empty
    if prompt:= st.chat_input('What is up?'):

        # Add and display user prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # OpenAI API request with streaming functionality

            openai.api_key=os.environ["OPENAI_API_KEY"]

            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],

                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ], temperature = temperature,
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

#########################################################################################################

# """ DALL-E API BASED IMAGE GENERATION ACCORDING TO USER PROMPT """

def image_generator():

    # Function that sends API request using user's prompt and returns image generated
    def generate_dall_e_image(prompt):
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            response = openai.Image.create(
                model="dall-e-3",  # DALL-E model
                prompt=prompt,
                n=1,  # Number of images to generate
            )
            image_url = response['data'][0]['url']
            return image_url
        
        except Exception as e:
            st.error(f"Error generating the image: {e}")
            return None
    
    # Working graphic interface
    st.title("DALL-E生成图片💡")
    st.info("输入提示 ")
    prompt = st.text_area("输入提示以生成图片:", "超酷的月球殖民地",label_visibility='collapsed')

    if st.button("生成图片",type='primary'): # When button is clicked
        if prompt.strip() == "":
            st.warning("请输入一句提示语...") # Warning message if prompt empty
        else:
            with st.spinner("生成中..."):
                image_url = generate_dall_e_image(prompt)
                if image_url:
                    st.image(image_url, caption=prompt, width=500) # Display image

####################################################################################################################

# """ ARTICLE/PARAGRAPH GENERATOR """
def image2text(img_url):
    imagetotext= pipeline("image-to-text", model="image2text_model")

    text = imagetotext(img_url)[0]["generated_text"]
    return text

def article_generator():

    # Request OpenAI for response based on options/criteria chosen by user
    def generate_article(keyword, writing_style, word_count,article_type):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "user", "content":f""" Write an {article_type} about ({keyword}) in a 
                    {writing_style} writing style with the length not exceeding {word_count} words, 用中文写作。"""}
                ]
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        return result

    # Working graphic interface
    st.title("基于ChatGPT的文章生成器😲")

    # Options available to user for generating specific article/post

    keyword = st.text_input("输入关键词:")
    writing_style = st.selectbox("选择写作风格:", ["休闲","信息丰富", "诙谐","引人入胜",'学术'])
    article_type = writing_style = st.selectbox("选择类型:", ["小论文", "博客", "小红书帖子","微博帖子",'摘要总结','榜单标题'])
    
    col1,col2 = st.columns([0.8,0.2]) # Adjust width percentage for each column
    with col1:
        # Word count - Special Note : ChatGPT not great at following the word limit
        # Could replace it with selectbox and options short, medium, long
        word_count = st.slider("字数", min_value=50, max_value=1000, step=50, value=500,format="%d 字", label_visibility='collapsed')
    with col2:
        submit_button = st.button("生成文章",use_container_width=True,type='primary')

    st.info('也可以上传图片，根据图片生成文章: ')
    input_fig = st.file_uploader("上传图片", type=['jpg', 'png'], label_visibility='collapsed')
    if input_fig:
        st.image(input_fig, caption=input_fig.name, width=500)  # Display image

    if submit_button:
        if input_fig:
            file_name = input_fig.name
            # 将文件保存到服务器的特定目录
            file_path = './upload_fig/' + file_name
            # 将上传的文件保存到磁盘
            with open(file_path, 'wb') as f:
                f.write(input_fig.getbuffer())

            fig_title = image2text(file_path)
            keyword += fig_title
        print(keyword)
        # Simulate progress bar animation
        progress_bar = st.progress(0)
        for i in range(51):
            time.sleep(0.05)  
            progress_bar.progress(i)
        article = generate_article(keyword, writing_style, word_count,article_type) # Generate article
        for i in range(51,101):
            time.sleep(0.05)  
            progress_bar.progress(i)  

        st.info("Process completed!")
        st.write(article)

        # Download file as text - Add additional functionality
        st.download_button(
            label="Download",
            data=article,
            file_name='Article.txt',
            mime='text/txt',
        )

#################################################################################################

# """ CHATBOT THAT CAN INTERPRET CSV FILES AND ANSWER USER QUERIES ( DATA ANALYSIS ) """

def chat_csv():

    st.title('LLM赋能的CSV对话Agent! 👾 ')
    st.info('上传CSV文件: ')
    input_csv = st.file_uploader("上传CSV文件",type=['csv'],label_visibility='collapsed') # File upload
    if input_csv is not None:
        st.info("CSV 上传成功!")
        data = pd.read_csv(input_csv)
        openai.api_key = os.environ["OPENAI_API_KEY"]
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0),data,verbose=True, allow_dangerous_code=True)
        st.dataframe(data,use_container_width=True) # Visual dataframe with rolws and columns

        st.info('输入问题..')
        input_text = st.text_area('输入问题..',label_visibility='collapsed')
        if input_text != None:
            if st.button('基于CSV提问'):
                result = agent.run(input_text) # Thinking process shown in terminal
                st.success(result)
        else:
            st.warning('Error : No input query given')

#########################################################################################################

# """ CHATBOT THAT CAN READS PDFs USING PYPDF2, LANGCHAIN AND ANSWERS USER QUERIES """

def chat_pdf():
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
    st.title('LLM赋能的PDF对话Agent 🧑‍🚀')
    st.info('上传PDF文件: ')
    pdf = st.file_uploader("上传PDF文件",type=['pdf'],label_visibility='collapsed')

    # Extract all text from PDF 
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks - ChatGPT cannot process extremely large message history from PDF
        # Have a overlap to ensure context in paragraphs that start from middle of sentences 
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        #Create embeddings ( Vector representations of text ) and store all chunks collection in 1 base
        embeddings = OpenAIEmbeddings()
        info_base = FAISS.from_texts(chunks,embeddings)

        user_question = st.text_input('输入问题: ')
        if user_question:
            # RETRIEVAL AUGMENTED GENERATION
            # Find relevant chunks as per langchain model using similarity search of vectors
            docs = info_base.similarity_search(user_question)
            llm = OpenAI()
            chain = load_qa_chain(llm,chain_type="refine") # Handle question answering tasks 
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,question=user_question)
                st.info(f"Completion tokens : {cb.completion_tokens}") # Tokens used
            st.write(response)

##########################################################################################################

# Only if application is run directly ( not imported ), run the code
if __name__ == "__main__":
    main()
