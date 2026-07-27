# 🚀 Yerel RAG Asistanı (Local Offline Q&A Assistant)

Bu proje, internet bağlantısı olmadan tamamen yerel (offline) çalışan bir **Retrieval-Augmented Generation (RAG)** asistanıdır. Yüklenen PDF ve TXT belgelerinden vektör tabanlı arama yaparak kullanıcının sorularını yerel bir Büyük Dil Modeli (LLM) ile yanıtlar.

## 🛠️ Teknolojiler
* **Dil:** Python
* **LLM & Embeddings:** Ollama (Llama 3), SentenceTransformers
* **Vektör Veritabanı / Arama:** SQLite / Numpy Cosine Similarity
* **Veri İşleme:** PyPDF

## 📦 Kurulum ve Çalıştırma

1. Repoyu klonlayın:
   ```bash
   git clone [https://github.com/thelolix/RAG.git](https://github.com/thelolix/RAG.git)
   cd RAG
