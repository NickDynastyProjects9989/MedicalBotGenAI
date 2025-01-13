## Medical Chatbot: AI-Powered Healthcare Assistant
<img width="791" alt="image" src="https://github.com/user-attachments/assets/b19a4aca-9b73-4d36-8425-eb7a60916e84" />


This project is a **Flask-based Medical Chatbot** application designed to provide AI-powered answers to medical queries. It leverages state-of-the-art technologies such as **LangChain**, **Pinecone**, and **Groq Model (Gemma2-9b-it)** to create a reliable, real-time query resolution system for users.

---

### Features
1. **Document-Based Retrieval**:
   - Utilizes Pinecone's vector database for storing and retrieving medical documents.
   - Retrieves the most relevant document chunks using semantic similarity for accurate query responses.

2. **Advanced Question-Answering**:
   - Powered by **Groq LLM (Gemma2-9b-it)** for generating precise and context-aware answers.
   - Combines retrieved documents and a tailored prompt to enhance the quality of responses.

3. **Embeddings and Indexing**:
   - Employs Hugging Face embeddings for high-quality vector representations of medical data.
   - Uses Pinecone to store and retrieve these vectors efficiently.

4. **Interactive Chat UI**:
   - Built using Flask and HTML templates for a seamless user experience.
   - Provides a user-friendly interface for inputting queries and receiving responses.

5. **Environment Variable Configuration**:
   - Securely manages API keys (e.g., Pinecone, Groq) using environment variables loaded via `dotenv`.

---

### Workflow
1. **Document Indexing**:
   - Pre-processes medical documents to generate embeddings using Hugging Face.
   - Stores embeddings in a Pinecone index for fast similarity search.

2. **Query Processing**:
   - User inputs a medical query via the chat interface.
   - The system retrieves relevant document chunks from Pinecone using the query.
   - Combines the document context with the user query in a structured prompt.
   - Groq LLM generates the final response.

3. **Real-Time Interaction**:
   - Displays responses to user queries in the chat interface.

---

### Requirements
- **Python 3.9+**
- **Dependencies**:
  - `flask`
  - `langchain`
  - `langchain_groq`
  - `pinecone-client`
  - `dotenv`

---

### How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/your_repo_name.git
   cd your_repo_name
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Add your API keys to a `.env` file:
     ```
     PINECONE_API_KEY=your_pinecone_api_key
     GROQ_API_KEY=your_groq_api_key
     ```

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the Chat Interface**:
   - Open your browser and navigate to `http://localhost:8080`.

---

### Directory Structure
```
.
├── data/
│   └── data.pdf         # Medical documents for indexing
├── research/
│   └── (Optional research notebooks)
├── src/
│   ├── __init__.py
│   ├── helper.py         # Helper functions for embedding downloads
│   ├── prompt.py         # Custom prompt templates
├── static/
│   └── style.css         # CSS for chat UI
├── templates/
│   └── chat.html         # HTML template for chat interface
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── setup.py              # Setup script
├── README.md             # Project description
└── LICENSE               # License information
```

---

### Future Enhancements
1. Add support for additional document types (e.g., Word, TXT).
2. Integrate real-time medical data for enhanced query responses.
3. Improve the chatbot's natural language understanding with fine-tuned models.
4. Add authentication to secure sensitive medical queries.

---

Feel free to explore, contribute, and enhance this project! 😊
