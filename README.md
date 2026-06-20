# AI_Meeting_Agent

# 🎬 AI Meeting Agent

An end-to-end **AI-powered meeting intelligence system** that transforms any video or audio — from a YouTube link or a local file — into a structured, searchable knowledge base. Powered by **OpenAI Whisper**, **Mistral AI**, **LangChain LCEL**, and **ChromaDB**, with a polished **Streamlit** UI.

---

## ✨ What It Does

| Capability | Description |
|---|---|
| 🎙️ **Transcription** | Converts audio/video to text using local OpenAI Whisper |
| 🌐 **Translation** | Optional Hindi → English translation during transcription |
| 📋 **Summarization** | Map-Reduce summarization for long meetings (bullet points) |
| 🏷️ **Title Generation** | Auto-generates a short professional meeting title |
| ✅ **Action Items** | Extracts tasks, owners, and deadlines |
| 🔑 **Key Decisions** | Pulls all important decisions made in the meeting |
| ❓ **Open Questions** | Identifies unresolved topics needing follow-up |
| 💬 **RAG Chat** | Chat with your meeting transcript using semantic search |

---

## 🧠 System Architecture

```
Input (YouTube URL / Local File)
          │
          ▼
 ┌─────────────────────┐
 │  utils/             │
 │  audio_processor.py │  → Download (yt-dlp) / Convert (pydub) → Chunk (10-min WAV slices)
 └─────────────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  core/              │
 │  transcriber.py     │  → Whisper ASR → Full Transcript
 └─────────────────────┘
          │
          ├──────────────────────────────────────────────────┐
          ▼                                                   ▼
 ┌─────────────────────┐                          ┌─────────────────────┐
 │  core/              │                          │  core/              │
 │  sammarize.py       │  → Map-Reduce Summary    │  vector_store.py    │  → ChromaDB
 │  extractor.py       │  → Action Items          │  rag_engine.py      │  → RAG Chain
 └─────────────────────┘    Key Decisions         └─────────────────────┘
                             Open Questions                  │
          │                                                   ▼
          └────────────────────────────────────► 💬 Chat with Meeting
                                                   (Semantic Q&A)
```

---

## 📁 Project Structure

```
AI_Meeting_Agent/
│
├── app.py                    # Streamlit UI — full dark-themed web interface
├── main.py                   # CLI entry point — terminal pipeline runner
├── requirements.txt          # All Python dependencies
│
├── core/                     # Core AI processing modules
│   ├── transcriber.py        # Whisper speech-to-text transcription
│   ├── sammarize.py          # Map-Reduce meeting summarization + title generation
│   ├── extractor.py          # Action items, key decisions, open questions extraction
│   ├── rag_engine.py         # RAG pipeline — retriever + LLM Q&A chain
│   └── vector_store.py       # ChromaDB vector store build / load / retrieve
│
├── utils/
│   └── audio_processor.py    # YouTube download, WAV conversion, audio chunking
│
└── Meetings/                 # Auto-created — stores downloaded/converted audio files
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Sidd28122003/AI_Meeting_Agent.git
cd AI_Meeting_Agent
```

### 2. Install System Dependency — FFmpeg

FFmpeg is required for audio processing and is **not** installed via pip.

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.10+ recommended**

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
WHISPER_MODEL=small          # Options: tiny | base | small | medium | large
```

- **`MISTRAL_API_KEY`** — Required. Get it from [console.mistral.ai](https://console.mistral.ai)
- **`WHISPER_MODEL`** — Optional. Defaults to `small`. Larger models are more accurate but slower.

### 5. Run the App

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**Terminal / CLI mode:**
```bash
python main.py
# → Enter YouTube URL or local file path when prompted
```

---

## 🖥️ UI Overview

The Streamlit app features a dark-themed interface with an aurora animated background and sidebar controls.

**Sidebar** — Input controls:
- Paste a **YouTube URL** or provide a **local file path**
- Click **▶ Analyze Meeting** to launch the pipeline
- Progress is shown step by step as each stage completes

**Main area — tabbed results:**

| Tab | Content |
|---|---|
| 📋 **Summary** | Bullet-point meeting summary (Map-Reduce) |
| ✅ **Action Items** | Numbered list of tasks with owner and deadline |
| 🔑 **Key Decisions** | All major decisions made during the meeting |
| ❓ **Open Questions** | Unresolved topics needing follow-up |
| 📜 **Transcript** | Full raw transcription text |
| 💬 **Chat** | RAG-powered Q&A — ask anything about the meeting |

---

## ⚙️ Core Modules

### `utils/audio_processor.py`
Handles all audio ingestion and preparation:
- `download_youtube_audio(url)` — downloads best quality audio via `yt-dlp`, converts to WAV
- `convert_to_wav(path)` — converts any local audio/video file to 16kHz mono WAV via `pydub`
- `chunk_audio(wav_path)` — splits WAV into **10-minute chunks** (Whisper works best on short segments)
- `process_input(source)` — unified entry point; auto-detects URL vs local file

### `core/transcriber.py`
- Loads Whisper model **once** (singleton pattern) to avoid repeated loading
- `transcribe_chunk(path, translate)` — transcribes a single chunk; `translate=True` converts to English
- `transcribe_all(chunks)` — iterates all chunks and concatenates into one full transcript
- Model size configurable via `WHISPER_MODEL` env var (`tiny` / `base` / `small` / `medium` / `large`)

### `core/sammarize.py`
Uses a **Map-Reduce** strategy to handle long transcripts that exceed LLM token limits:
1. **Map** — each chunk is summarized independently
2. **Reduce** — all chunk summaries are merged into one final bullet-point summary
- `generate_title(transcript)` — generates a max 8-word professional meeting title from the first 2000 chars

### `core/extractor.py`
Three independent LangChain LCEL chains, each with a specialized system prompt:
- `extract_action_items(transcript)` — task + owner + deadline per item
- `extract_key_decisions(transcript)` — all decisions as a numbered list
- `extract_questions(transcript)` — unresolved or follow-up questions

### `core/vector_store.py`
Manages the ChromaDB vector database:
- Chunks transcript into **500-char segments** (50-char overlap)
- Embeds using `all-MiniLM-L6-v2` from HuggingFace (runs on CPU)
- Persists to `./vector_db/` — can be reloaded across sessions without re-embedding

### `core/rag_engine.py`
Builds a full **LCEL RAG pipeline**:
```
User Question → Retriever (top-4 chunks) → Prompt → Mistral LLM → Answer
```
- Answers are grounded strictly in the meeting transcript
- Returns `"I could not find this information in the meeting transcript."` if the answer isn't found
- `load_rag_chain()` — reuses a persisted vector DB for production use cases

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `openai-whisper` | Local speech-to-text transcription |
| `yt-dlp` | YouTube audio download |
| `pydub` | Audio format conversion and chunking |
| `ffmpeg-python` | FFmpeg bindings (requires FFmpeg binary) |
| `langchain` + `langchain-core` | LCEL chains and prompt templates |
| `langchain-mistralai` | Mistral AI LLM integration |
| `chromadb` + `langchain-chroma` | Local vector database for RAG |
| `sentence-transformers` | HuggingFace embeddings (`all-MiniLM-L6-v2`) |
| `streamlit` | Web UI |
| `deep-translator` | Hindi → English translation support |
| `fpdf2` / `reportlab` | PDF export |
| `python-dotenv` | `.env` API key loading |

---

## 🗒️ Notes

- The `Meetings/` folder is auto-created on first run to store downloaded and converted audio.
- The `vector_db/` folder is auto-created to persist the ChromaDB vector store — delete it to force a rebuild on next run.
- Whisper runs **fully locally** — no audio data is sent to any external API.
- For very long meetings, use `WHISPER_MODEL=medium` or `large` for better accuracy at the cost of more RAM and time.
- The RAG chat will only answer from the meeting transcript — it will not hallucinate outside the source content.
- Both `main.py` (CLI) and `app.py` (Streamlit) run the same underlying pipeline from `core/` and `utils/`.
