from fastapi import FastAPI, HTTPException, Query
import ctranslate2
import transformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Optional
from functools import lru_cache

def retrieve_context(query: str):
    """
    Retrieve relevant context from the vector database using a lightweight embedding model.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        db = FAISS.load_local("./VecDB", embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})  # Retrieve 3 documents for better context
        docs = retriever.get_relevant_documents(query)
        print(docs)
        context = "\n".join([doc.page_content for doc in docs])
        print(context)
        return context
        

    except Exception as e:
        return ""  # Fallback to empty context if retrieval fails

# Initialize FastAPI app
app = FastAPI()

# Load CTranslate2 model and tokenizer
model_path = "./models/tinyllama-1.1b-ct2"  # Update with actual path
try:
    generator = ctranslate2.Generator(model_path, device="cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
except Exception as e:
    raise RuntimeError(f"Failed to load CTranslate2 model: {e}")

@lru_cache(maxsize=1000)
def query_llm_cached(question: str, context: str):
    """
    Cached function to generate response from the LLM.
    """
    # Enhanced prompt engineering
    full_prompt = (
        "You are an advanced AI assistant designed to provide accurate, concise, and helpful answers. "
        "When answering, ensure your response is clear and directly addresses the question. "
        "If additional context is provided, incorporate it into your response for better accuracy.\n\n"
    )
    if context:
        full_prompt += f"Context: {context}\n\n"
    full_prompt += f"Question: {question}\n\nAnswer:"

    # Tokenize input
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(full_prompt))
    
    # Generate response with optimized decoding parameters
    results = generator.generate_batch(
        [tokens],
        max_length=50,  # Increased max tokens for more detailed responses
        sampling_topk=5,  # Use top-k sampling for diversity
        sampling_temperature=0.7  # Lower temperature for more deterministic responses
    )
    
    # Decode output
    output = tokenizer.decode(results[0].sequences_ids[0])
    return output.strip()

@app.get("/query")
async def query_llm(
    question: str = Query(..., description="Your question to the LLM", max_length=1000),
    context: Optional[str] = None,
    use_rag: bool = Query(True, description="Whether to use RAG for context retrieval")
):
    """
    Ask a question to the LLM with optional user-provided context or retrieved context.
    """
    try:
        # Use user-provided context or retrieve context via RAG
        final_context = context if context else (retrieve_context(question) if use_rag else "")
        result = query_llm_cached(question, final_context)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")