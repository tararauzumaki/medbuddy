# MedBuddy 🏥🤖  
MedBuddy is an AI-powered **medical assistant** that transforms a static medical reference into an **interactive chatbot**. It uses **The GALE Encyclopedia of Medicine** as its knowledge base and provides evidence-based answers to health-related queries.  

---

## ✨ Features  
✅ **Knowledge-Powered** – Uses *The GALE Encyclopedia of Medicine* as its core reference.  
✅ **Smart Search** – Retrieves information on **symptoms, conditions, and treatments**.  
✅ **Bangladesh-Oriented** – Provides **dietary & health advice tailored for Bangladesh**.  
✅ **Multilingual Support** – **Responds in Bangla** when queried in Bangla.  
✅ **Empathetic & Ethical** – Avoids diagnoses, prescriptions, or misinformation.  
✅ **Interactive UI** – Built with **Gradio** for seamless chat interactions.  

---

## 🚀 How It Works  

MedBuddy processes **medical data**, understands **user queries**, and delivers structured **AI-driven responses**. Here's a step-by-step breakdown:  

### 1️⃣ Initializing the AI Model  
MedBuddy uses **LLaMA-3 (70B)** from **Groq API** to generate human-like medical insights.  

```python
def initial_llm():
    llm = ChatGroq(
        temperature=0,  # Ensures deterministic responses
        groq_api_key="your_api_key_here",
        model_name="llama-3.3-70b-versatile"
    )
    return llm

# MedBuddy: Your AI Medical Assistant 🏥🤖

## 2️⃣ Creating the Medical Knowledge Database
To enable fast and accurate retrieval, MedBuddy embeds the encyclopedia into ChromaDB.

```python
# create_db.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def create_db():
    pdf_path = "The_GALE_ENCYCLOPEDIA_of_MEDICINE.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    return vector_db
```

---

## 3️⃣ Setting Up the AI-Powered Query System
MedBuddy's RetrievalQA chain is fine-tuned with a custom medical prompt that:
✔ Explains symptoms clearly using simple language.
✔ Warns about urgent symptoms ⚠️.
✔ Encourages professional medical consultation.
✔ Provides Bangladesh-specific diet recommendations.
✔ Uses internationally recognized medical journals.

```python
# qa_chain.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def setup_qachain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are MedBuddy, an empathetic medical assistant focused on providing evidence-based health information.
Context from medical resources:
{context}
Question: {question}
Response:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain
```

---

## 4️⃣ Processing User Queries
When a user submits a question:
🔹 MedBuddy searches ChromaDB for relevant medical data.
🔹 The AI analyzes the query and retrieves an appropriate response.
🔹 The chatbot formats the output for easy understanding.

```python
# chatbot.py
def chatbot_response(message, chat_history):
    if not message.strip():
        return "", chat_history
    try:
        response = qa_chain.invoke({"query": message})
        chat_history.append((message, response["result"]))
        return "", chat_history
    except Exception as e:
        chat_history.append((message, f"An error occurred: {str(e)}"))
        return "", chat_history
```

---

## 5️⃣ User-Friendly Chatbot UI
Built with Gradio, MedBuddy offers a simple & intuitive chat interface.

```python
# app.py
import gradio as gr

def launch_chatbot():
    with gr.Blocks() as demo:
        gr.Markdown("# MedBuddy: Your AI Medical Assistant 🏥🤖")
        chatbot = gr.Chatbot(show_copy_button=True, height=400)
        txt = gr.Textbox(placeholder="Describe your symptoms...")
        send_btn = gr.Button("Send")
        txt.submit(chatbot_response, inputs=[txt, chatbot], outputs=[txt, chatbot])
        send_btn.click(chatbot_response, inputs=[txt, chatbot], outputs=[txt, chatbot])
    demo.launch(share=True)
```

---

## 📥 Installation & Setup

### 1️⃣ Install Dependencies
Make sure you have Python 3.8+ and the required libraries installed:

```sh
pip install gradio langchain chromadb sentence-transformers langchain-community langchain-groq
```

### 2️⃣ Add Your Medical PDF
Place *The GALE Encyclopedia of Medicine* PDF in the project folder.

### 3️⃣ Run MedBuddy
Launch the chatbot with:

```sh
python medbuddy.py
```

---

## 🛡️ Important Disclaimer
⚠️ **MedBuddy is NOT a replacement for professional medical advice.** It provides general health insights but does not diagnose or prescribe treatments. Always consult a qualified healthcare professional for medical concerns.

If you experience severe symptoms, seek immediate medical attention! 🚑

---

## 📌 Why Use MedBuddy?
✔ **Instant Medical Information** – No need to manually search lengthy medical texts.
✔ **Easy-to-Understand** – Simplifies complex medical terms.
✔ **Bangladesh-Specific Advice** – Provides localized dietary recommendations.
✔ **AI-Powered Smart Responses** – Uses cutting-edge AI to generate relevant medical insights.

---

## 🤝 Contributing
We welcome contributions! If you'd like to improve MedBuddy:

1. Fork the repository
2. Create a feature branch:
   ```sh
   git checkout -b feature-branch
   ```
3. Commit changes:
   ```sh
   git commit -m "Your changes"
   ```
4. Push to GitHub:
   ```sh
   git push origin feature-branch
   ```
5. Open a pull request

---

## 📜 License
This project is licensed under the **MIT License**. Feel free to use, modify, and distribute MedBuddy while adhering to the terms of the license.

---

## 🌟 Acknowledgments
Special thanks to my wife **Sadia Islam**, who inspired this project! ❤️
 

🚀 *Stay informed with MedBuddy!*  
