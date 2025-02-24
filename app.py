import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os

def initial_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )
    return llm

def create_db():
    pdf_path = "The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"🚨 PDF file not found at: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db_path = "./chroma_db"
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
    print("✅ ChromaDB created and medical data saved!")
    return vector_db

def setup_qachain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are MedBuddy, an empathetic medical assistant focused on providing evidence-based health information. Your responses should:
1. Be clear, concise, and use simple language with medical terms in parentheses when needed
2. Always emphasize the importance of consulting healthcare professionals
3. Mark urgent symptoms with ⚠️
4. Provide practical lifestyle and preventive care recommendations
5. Include reliable sources when possible
6. Stay within your scope: no diagnoses, prescriptions, or definitive medical advice
7. Maintain HIPAA compliance and medical ethics
8. Show empathy while remaining professional
9. Keep the answers, diet plans India oriented, cause most of the users will be Indians.
10. Provide mental health related guidelines as well, citing proper references.
11. If users ask questions in bengali, respond them in bengali. If they ask questions in english, respond them in english.
Context from medical resources:
{context}
Question: {question}
Response (structured with clear headings and bullet points when appropriate):"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

def chatbot_response(message, chat_history):
    if not message.strip():
        return "", chat_history
    
    try:
        response = qa_chain.invoke({"query": message})
        chat_history.append((message, response["result"]))
        return "", chat_history
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        chat_history.append((message, error_message))
        return "", chat_history

# Example questions
example_questions = [
    "For the past two days, I've had a persistent headache along with a fever."
    "Sharp pain in the lower right abdomen with nausea that gets worse after eating.",
    "Frequent dizziness and exhaustion, especially when getting up too quickly.",
    "Mild activity leaves me with chest pain and difficulty breathing.",
    "Even after sufficient rest, I’m struggling with a throbbing headache and sensitivity to light.",
    "Abdominal pain in the lower right side and nausea that worsens after eating"
]

# Initialize database and chain
db_path = "./chroma_db"
if not os.path.exists(db_path):
    vector_db = create_db()
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

llm = initial_llm()
qa_chain = setup_qachain(vector_db, llm)

def launch_chatbot():
    with gr.Blocks() as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.Image("logo.png", show_label=False, container=False)
        
        gr.Markdown("""
        MedBuddy is a fun lab project that was given by my wife **Sadia Islam**.
        """)
        
        chatbot = gr.Chatbot(
            show_copy_button=True,
            height=400
        )
        
        with gr.Row():
            txt = gr.Textbox(
                placeholder="Describe your symptoms or ask a health related question...",
                scale=4
            )
            send_btn = gr.Button("Send", scale=1)
        
        gr.Markdown("### Common Symptom:")
        with gr.Row():
            for question in example_questions:
                gr.Button(question).click(
                    lambda q: q,
                    inputs=[gr.Textbox(value=question, visible=False)],
                    outputs=[txt]
                ).then(
                    chatbot_response,
                    inputs=[txt, chatbot],
                    outputs=[txt, chatbot]
                )
        
        gr.Markdown("""
        Important Disclaimer: Our MedBuddy offers general health information and initial insights based on reported symptoms.
It is NOT intended for emergencies or to replace professional medical advice.
The information provided does not constitute a diagnosis. Always consult a licensed healthcare professional for any medical concerns.
If you are experiencing severe symptoms, seek immediate medical attention.😊
        """)
        
        txt.submit(chatbot_response, inputs=[txt, chatbot], outputs=[txt, chatbot])
        send_btn.click(chatbot_response, inputs=[txt, chatbot], outputs=[txt, chatbot])

    demo.launch(share=True)

if __name__ == "__main__":
    launch_chatbot()