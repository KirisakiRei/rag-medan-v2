#!/usr/bin/env python3
# ============================================================
# 🤖 DEV CHATBOT — Testing RAG System (Text + Document)
# ============================================================
import requests
import json
import sys
import os
import logging
from datetime import datetime
from config import CONFIG

# ============================================================
# 🎨 ANSI Color Codes untuk Terminal
# ============================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# ============================================================
# 🔧 Configuration
# ============================================================
RAG_TEXT_URL = f"http://{CONFIG['api']['host']}:{CONFIG['api']['port']}/api/search"
RAG_DOC_URL = f"{CONFIG['doc_api']['base_url']}/api/doc-search"

# ============================================================
# 📝 Setup Logging
# ============================================================
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "dev-chatbot.log")

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("dev_chatbot")
logger.info("=" * 60)
logger.info("🤖 DEV CHATBOT - Session Started")
logger.info("=" * 60)

# ============================================================
# 📊 Logging Helper
# ============================================================
def print_separator(char="=", length=80):
    print(f"{Colors.CYAN}{char * length}{Colors.END}")

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{text}{Colors.END}")

def print_info(label, value, color=Colors.BLUE):
    print(f"{color}{label:.<30} {value}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

# ============================================================
# 🔍 Query RAG Text
# ============================================================
def query_rag_text(question: str, wa_number: str = "dev-test"):
    """Query ke RAG Text API dengan logging detail."""
    
    logger.info("\n" + "=" * 60)
    logger.info(f"[RAG TEXT] Query: {question}")
    logger.info(f"[RAG TEXT] WA Number: {wa_number}")
    logger.info("=" * 60)
    
    print_separator()
    print_header("🔍 QUERY RAG TEXT")
    print_separator()
    print_info("Pertanyaan", question, Colors.BOLD)
    print_info("WA Number", wa_number)
    print_info("Endpoint", RAG_TEXT_URL)
    print_info("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print_separator()

    try:
        payload = {
            "question": question,
            "wa_number": wa_number
        }

        print(f"\n{Colors.CYAN}📤 Mengirim request...{Colors.END}")
        response = requests.post(RAG_TEXT_URL, json=payload, timeout=30)
        
        print_info("HTTP Status", response.status_code)
        logger.info(f"[RAG TEXT] HTTP Status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"[RAG TEXT] Request failed: {response.status_code}")
            print_error(f"Request gagal dengan status {response.status_code}")
            print(response.text[:500])
            return None

        data = response.json()
        
        # =====================================================
        # 📋 Parse Response
        # =====================================================
        print_separator("─")
        print_header("📋 RESPONSE DETAILS")
        print_separator("─")
        
        status = data.get("status", "-")
        message = data.get("message", "-")
        source = data.get("source", "text")
        
        logger.info(f"[RAG TEXT] Status: {status} | Message: {message} | Source: {source}")
        
        print_info("Status", status, Colors.GREEN if status == "success" else Colors.YELLOW)
        print_info("Message", message)
        print_info("Source", source, Colors.CYAN)
        
        # =====================================================
        # ⏱️ Timing Information
        # =====================================================
        timing = data.get("timing", {})
        if timing:
            print_separator("─")
            print_header("⏱️ TIMING")
            print_separator("─")
            print_info("AI Domain Check", f"{timing.get('ai_domain_sec', 0)}s")
            print_info("AI Relevance Check", f"{timing.get('ai_relevance_sec', 0)}s")
            print_info("Embedding", f"{timing.get('embedding_sec', 0)}s")
            print_info("Qdrant Search", f"{timing.get('qdrant_sec', 0)}s")
            print_info("Total Time", f"{timing.get('total_sec', 0)}s", Colors.BOLD)
        
        # =====================================================
        # 📦 Data & Results
        # =====================================================
        data_section = data.get("data", {})
        similar_questions = data_section.get("similar_questions", [])
        metadata = data_section.get("metadata", {})
        
        print_separator("─")
        print_header("📦 METADATA")
        print_separator("─")
        print_info("Original Question", metadata.get("original_question", "-"))
        print_info("Final Question", metadata.get("final_question", "-"))
        print_info("Category", metadata.get("category", "-"))
        print_info("AI Reason", metadata.get("ai_reason", "-"))
        print_info("Final Score Top", str(metadata.get("final_score_top", "-")))
        
        # =====================================================
        # 🎯 Results
        # =====================================================
        if similar_questions:
            print_separator("─")
            print_header(f"🎯 HASIL PENCARIAN ({len(similar_questions)} items)")
            print_separator("─")
            
            logger.info(f"[RAG TEXT] Found {len(similar_questions)} results")
            for idx, item in enumerate(similar_questions[:3], start=1):
                print(f"\n{Colors.BOLD}{Colors.BLUE}[{idx}] {Colors.END}")
                print_info("Question", item.get("question", "-")[:80])
                print_info("RAG Name", item.get("question_rag_name", "-")[:80])
                print_info("Answer ID", str(item.get("answer_id", "-")))
                print_info("Category ID", str(item.get("category_id", "-")))
                print_info("Dense Score", f"{item.get('dense_score', 0):.3f}")
                print_info("Overlap Score", f"{item.get('overlap_score', 0):.3f}")
                print_info("Final Score", f"{item.get('final_score', 0):.3f}", Colors.GREEN)
                print_info("Note", item.get("note", "-"))
                
                logger.info(f"[RAG TEXT] Result #{idx}: Score={item.get('final_score', 0):.3f} | AnswerID={item.get('answer_id', '-')} | Note={item.get('note', '-')}")
        else:
            logger.warning("[RAG TEXT] No results found")
            print_warning("Tidak ada hasil yang ditemukan")
        
        print_separator()
        logger.info("=" * 60)
        
        return data

    except requests.exceptions.Timeout:
        logger.error("[RAG TEXT] Request timeout (>30s)")
        print_error("Request timeout (>30s)")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"[RAG TEXT] Connection error to {RAG_TEXT_URL}")
        print_error(f"Tidak dapat terhubung ke {RAG_TEXT_URL}")
        print_warning("Pastikan service RAG Text sudah berjalan!")
        return None
    except Exception as e:
        logger.error(f"[RAG TEXT] Error: {str(e)}", exc_info=True)
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# 📄 Query RAG Document (Direct)
# ============================================================
def query_rag_document(question: str, limit: int = 3):
    """Query langsung ke RAG Document API."""
    
    logger.info("\n" + "=" * 60)
    logger.info(f"[RAG DOC] Query: {question}")
    logger.info(f"[RAG DOC] Limit: {limit}")
    logger.info("=" * 60)
    
    print_separator()
    print_header("📄 QUERY RAG DOCUMENT")
    print_separator()
    print_info("Query", question, Colors.BOLD)
    print_info("Limit", str(limit))
    print_info("Endpoint", RAG_DOC_URL)
    print_info("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print_separator()

    try:
        payload = {"query": question, "limit": limit}

        print(f"\n{Colors.CYAN}📤 Mengirim request...{Colors.END}")
        response = requests.post(RAG_DOC_URL, json=payload, timeout=30)
        
        print_info("HTTP Status", response.status_code)
        logger.info(f"[RAG DOC] HTTP Status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"[RAG DOC] Request failed: {response.status_code}")
            print_error(f"Request gagal dengan status {response.status_code}")
            print(response.text[:500])
            return None

        data = response.json()
        
        # =====================================================
        # 📋 Parse Response
        # =====================================================
        print_separator("─")
        print_header("📋 RESPONSE DETAILS")
        print_separator("─")
        
        status = data.get("status", "-")
        mode = data.get("mode", "-")
        
        logger.info(f"[RAG DOC] Status: {status} | Mode: {mode}")
        
        print_info("Status", status, Colors.GREEN if status == "success" else Colors.YELLOW)
        print_info("Mode", mode)
        print_info("Query", data.get("query", "-"))
        
        # =====================================================
        # 📑 Summary (jika ada)
        # =====================================================
        summary = data.get("summary")
        if summary:
            print_separator("─")
            print_header("📑 SUMMARY (dari LLM)")
            print_separator("─")
            print(f"{Colors.GREEN}{summary}{Colors.END}")
        
        # =====================================================
        # 🎯 Results
        # =====================================================
        results = data.get("results", [])
        if results:
            print_separator("─")
            print_header(f"🎯 HASIL DOKUMEN ({len(results)} items)")
            print_separator("─")
            
            logger.info(f"[RAG DOC] Found {len(results)} results")
            for idx, item in enumerate(results[:3], start=1):
                print(f"\n{Colors.BOLD}{Colors.BLUE}[{idx}] {Colors.END}")
                print_info("Filename", item.get("filename", "-"))
                print_info("OPD", item.get("opd", "-"))
                print_info("Page Number", str(item.get("page_number", "-")))
                print_info("Chunk Index", str(item.get("chunk_index", "-")))
                print_info("Score", f"{item.get('score', 0):.3f}", Colors.GREEN)
                print_info("Text Preview", (item.get("text", "-")[:100] + "..."))
                
                logger.info(f"[RAG DOC] Result #{idx}: Score={item.get('score', 0):.3f} | File={item.get('filename', '-')} | Page={item.get('page_number', '-')}")
        else:
            logger.warning("[RAG DOC] No results found")
            print_warning("Tidak ada hasil yang ditemukan")
        
        print_separator()
        logger.info("=" * 60)
        
        return data

    except requests.exceptions.Timeout:
        logger.error("[RAG DOC] Request timeout (>30s)")
        print_error("Request timeout (>30s)")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"[RAG DOC] Connection error to {RAG_DOC_URL}")
        print_error(f"Tidak dapat terhubung ke {RAG_DOC_URL}")
        print_warning("Pastikan service RAG Document sudah berjalan!")
        return None
    except Exception as e:
        logger.error(f"[RAG DOC] Error: {str(e)}", exc_info=True)
        print_error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# 🎮 Interactive Chatbot Mode
# ============================================================
def interactive_mode():
    """Mode interaktif untuk testing."""
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("=" * 80)
    print("🤖 DEV CHATBOT — RAG System Testing")
    print("=" * 80)
    print(f"{Colors.END}")
    print(f"{Colors.CYAN}Commands:{Colors.END}")
    print(f"  {Colors.BOLD}text{Colors.END}     → Query ke RAG Text (dengan fallback otomatis)")
    print(f"  {Colors.BOLD}doc{Colors.END}      → Query langsung ke RAG Document")
    print(f"  {Colors.BOLD}both{Colors.END}     → Query ke keduanya (perbandingan)")
    print(f"  {Colors.BOLD}exit{Colors.END}     → Keluar dari chatbot")
    print(f"  {Colors.BOLD}quit{Colors.END}     → Keluar dari chatbot")
    print_separator()

    while True:
        try:
            # Input mode
            print(f"\n{Colors.BOLD}{Colors.YELLOW}Pilih mode [text/doc/both]:{Colors.END} ", end="")
            mode = input().strip().lower()

            if mode in ["exit", "quit", "q"]:
                print_success("Terima kasih! Goodbye! 👋")
                break

            if mode not in ["text", "doc", "both"]:
                print_warning("Mode tidak valid! Gunakan: text, doc, atau both")
                continue

            # Input question
            print(f"{Colors.BOLD}{Colors.YELLOW}Pertanyaan:{Colors.END} ", end="")
            question = input().strip()

            if not question:
                print_warning("Pertanyaan tidak boleh kosong!")
                continue

            # Execute query
            if mode == "text":
                query_rag_text(question)
            elif mode == "doc":
                query_rag_document(question)
            elif mode == "both":
                print(f"\n{Colors.BOLD}{Colors.CYAN}🔄 MODE PERBANDINGAN{Colors.END}")
                query_rag_text(question)
                print("\n")
                query_rag_document(question)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️  Interrupted by user{Colors.END}")
            break
        except Exception as e:
            print_error(f"Error: {str(e)}")

# ============================================================
# 🚀 Main Entry Point
# ============================================================
def main():
    """Main function dengan argument support."""
    
    if len(sys.argv) > 1:
        # Mode CLI dengan argument
        if sys.argv[1] in ["-h", "--help"]:
            print(f"{Colors.BOLD}Usage:{Colors.END}")
            print(f"  python dev_chatbot.py                    → Interactive mode")
            print(f"  python dev_chatbot.py text \"pertanyaan\"   → Query RAG Text")
            print(f"  python dev_chatbot.py doc \"pertanyaan\"    → Query RAG Document")
            print(f"  python dev_chatbot.py both \"pertanyaan\"   → Query keduanya")
            return

        mode = sys.argv[1].lower()
        question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""

        if not question:
            print_error("Pertanyaan tidak boleh kosong!")
            print(f"Usage: python dev_chatbot.py {mode} \"pertanyaan anda\"")
            return

        if mode == "text":
            query_rag_text(question)
        elif mode == "doc":
            query_rag_document(question)
        elif mode == "both":
            query_rag_text(question)
            print("\n")
            query_rag_document(question)
        else:
            print_error(f"Mode '{mode}' tidak dikenal. Gunakan: text, doc, atau both")
    else:
        # Mode interaktif
        interactive_mode()

if __name__ == "__main__":
    main()
