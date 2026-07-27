import sqlite3
import numpy as np
import json
import os
import ollama
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# 0. ANLAMSAL EMBEDDING MODELİ
print("⏳ Anlamsal Embedding Modeli Yükleniyor...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 1. VERİTABANI OLUŞTURMA (SQLite)
def init_db():
    conn = sqlite3.connect("rag_knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            content TEXT,
            embedding TEXT
        )
    """)
    conn.commit()
    conn.close()

# 2. VEKTÖR BENZERLİK HESABI
def compute_cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# 3. EMBEDDING ALMA
def get_text_embedding(text):
    embedding = embed_model.encode(text)
    return embedding.tolist()

# 4. DOKÜMAN EKLEME
def add_document_to_db(filename, text):
    embedding_vector = get_text_embedding(text)
    conn = sqlite3.connect("rag_knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (filename, content, embedding) VALUES (?, ?, ?)",
        (filename, text, json.dumps(embedding_vector))
    )
    conn.commit()
    conn.close()

# 5. DOSYA OKUMA VE CHUNKING
def load_files_from_folder(folder_path="."):
    conn = sqlite3.connect("rag_knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    count = cursor.fetchone()[0]
    conn.close()

    if count > 0:
        return

    print("\n📥 Klasördeki belgeler taranıyor ve veritabanına işleniyor...")
    found_any = False
    for file in os.listdir(folder_path):
        filepath = os.path.join(folder_path, file)
        
        if file.endswith(".txt") and not file.startswith("requirements"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                if text.strip():
                    add_document_to_db(file, text.strip())
                    print(f"📄 TXT Yüklendi: {file}")
                    found_any = True

        elif file.endswith(".pdf"):
            try:
                reader = PdfReader(filepath)
                full_text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        full_text += extracted + "\n"
                
                chunks = [full_text[i:i+300] for i in range(0, len(full_text), 250)]
                chunk_count = 0
                for chunk in chunks:
                    if chunk.strip():
                        add_document_to_db(file, chunk.strip())
                        chunk_count += 1
                print(f"📕 PDF YÜKLENDİ! ({chunk_count} Parça): {file}")
                found_any = True
            except Exception as e:
                print(f"⚠️ PDF Okuma Hatası ({file}): {e}")

# 6. EN YAKIN DOKÜMANI BULMA (Retrieval)
def get_top_chunks(query, top_k=2):
    query_vector = get_text_embedding(query)
    
    conn = sqlite3.connect("rag_knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute("SELECT filename, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for filename, content, emb_str in rows:
        doc_vector = json.loads(emb_str)
        sim = compute_cosine_similarity(query_vector, doc_vector)
        results.append((sim, filename, content))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

# 7. LOCAL LLM YANIT ÜRETİMİ (Generation via Ollama)
def generate_llm_response(query, context):
    """Bulunan belge parçalarını (context) yerel LLM'e verip yanıt üretir."""
    prompt = f"""Sen yardımcı bir yapay zeka asistanısın.
Sadece aşağıdaki 'BELGE İÇERİĞİ' bölümündeki bilgileri kullanarak kullanıcının sorusunu Türkçe olarak yanıtla.
Eğer verilen içerikte sorunun cevabı yoksa, 'Belgelerde bu konu hakkında bilgi bulunmamaktadır.' de. Asla dışarıdan bilgi uydurma.

BELGE İÇERİĞİ:
{context}

KULLANICI SORUSU:
{query}

YANIT:"""

    try:
        # Ollama üzerinden yerel model çağrısı (varsayılan: llama3 veya phi3)
        response = ollama.chat(
            model='llama3', # Bilgisayarında yüklü model adı (ör. llama3, phi3, mistral)
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        # Ollama servisi veya model hazır değilse yedek mesaj
        return f"⚠️ Yerel LLM Çalıştırılamadı (Ollama Hatası): {e}\n\n[Sistem Modeli Bulamadığı İçin Doğrudan Metin Gösteriliyor]:\n{context}"

# 8. SOHBET KONTROLÜ
def check_chat_intent(query):
    q = query.lower().strip()
    chat_keywords = ["sa", "as", "selam", "merhaba", "naber", "nasılsın", "günaydın", "iyi akşamlar", "knk", "kanka", "kimsin"]
    
    if len(q) <= 2 or any(keyword in q for keyword in chat_keywords):
        if "naber" in q or "nasılsın" in q:
            return "İyidir kanka, sen naber? Belgelerinle ilgili soruları yanıtlamaya hazırım!"
        elif "sa" in q or "selam" in q or "merhaba" in q:
            return "Aleykümselam / Selam! Nasıl yardımcı olabilirim?"
        elif "kimsin" in q:
            return "Ben senin yerel RAG asistanınım! Yüklediğin PDF/TXT dosyalarından sorularını yanıtlarım."
        else:
            return "Selam! Belgenle ilgili bir soru sormak ister misin?"
    return None

# ANA UYGULAMA
def main():
    print("\n==============================================")
    print("🚀 Local RAG Asistanı (Çevrimdışı Soru-Cevap)")
    print("==============================================\n")
    
    init_db()
    load_files_from_folder()

    print("\n✅ Sistem hazır! Çıkmak için 'q' yazabilirsiniz.\n")
    print("-" * 50)

    while True:
        sorgu = input("\n❓ Sorunuz: ")
        if sorgu.lower() in ['q', 'çıkış', 'exit']:
            print("👋 Görüşmek üzere!")
            break
        
        if not sorgu.strip():
            continue

        # 1. Sohbet Kontrolü
        sohbet_cevabi = check_chat_intent(sorgu)
        if sohbet_cevabi:
            print(f"\n🤖 Asistan Yanıtı:\n💬 {sohbet_cevabi}")
            continue

        # 2. Belge Arama (Retrieval)
        en_yakinlar = get_top_chunks(sorgu, top_k=2)
        
        if en_yakinlar:
            en_iyi_skor, dosya, metin = en_yakinlar[0]
            
            if en_iyi_skor < 0.25:
                print("\n🤖 Asistan Yanıtı:\n📄 [Bilgi Bulunamadı]: Belgelerde bu soruyla alakalı bir bilgi yer almamaktadır.")
            else:
                # Bağlamı (Context) Hazırla
                birlestirilmis_baglam = "\n---\n".join([item[2] for item in en_yakinlar])
                print(f"🔍 [Arama Skoru: %{en_iyi_skor*100:.1f}] Bulunan Kaynak: '{dosya}'")
                print("⏳ Yerel LLM yanıt üretiyor...")
                
                # 3. Yanıt Üretimi (Generation)
                llm_yaniti = generate_llm_response(sorgu, birlestirilmis_baglam)
                print(f"\n🤖 Asistan Yanıtı:\n{llm_yaniti}")
        else:
            print("❌ Veritabanında alakalı bir bilgi bulunamadı.")

if __name__ == "__main__":
    main()