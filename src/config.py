import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai


class GeminiEmbeddingFunction:
	def __init__(self, api_key: str, model: str):
		self.client = genai.Client(api_key=api_key)
		self.model = model
		self.is_embedding_001 = "embedding-001" in model

	def embed_documents(self, texts):
		embeddings = []
		for text in texts:
			if self.is_embedding_001:
				result = self.client.models.embed_content(
					model=self.model,
					contents=[text],
					config={"task_type": "RETRIEVAL_DOCUMENT"},
				)
			else:
				result = self.client.models.embed_content(
					model=self.model,
					contents=[f"title: none | text: {text}"],
				)
			embeddings.append(result.embeddings[0].values)
		return embeddings

	def embed_query(self, text):
		if self.is_embedding_001:
			result = self.client.models.embed_content(
				model=self.model,
				contents=[text],
				config={"task_type": "RETRIEVAL_QUERY"},
			)
		else:
			result = self.client.models.embed_content(
				model=self.model,
				contents=[f"task: search result | query: {text}"],
			)
		return result.embeddings[0].values

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

BASE_DIR = Path(__file__).resolve().parents[1]

_persist_dir_value = os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma_db")
CHROMA_PERSIST_DIRECTORY = str(
	Path(_persist_dir_value)
	if Path(_persist_dir_value).is_absolute()
	else (BASE_DIR / _persist_dir_value)
)
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "travel_documents")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
CHAT_MODEL_FALLBACKS = os.getenv(
	"GEMINI_CHAT_MODEL_FALLBACKS",
	"gemini-3.6-mini,gemini-2.0-flash-lite"
)
FASTAPI_URL = "http://localhost:8000"


def get_embeddings():
	if not GEMINI_API_KEY:
		raise ValueError("GEMINI_API_KEY is not set in the environment")

	return GeminiEmbeddingFunction(api_key=GEMINI_API_KEY, model=EMBEDDING_MODEL)


def build_chat_model(model_name: str):
	if not GEMINI_API_KEY:
		raise ValueError("GEMINI_API_KEY is not set in the environment")

	return ChatGoogleGenerativeAI(
		model=model_name,
		google_api_key=GEMINI_API_KEY,
	)


def get_chat_model_candidates() -> list[str]:
	fallbacks = [m.strip() for m in CHAT_MODEL_FALLBACKS.split(",") if m.strip()]
	ordered = [CHAT_MODEL, *fallbacks]
	seen = set()
	result = []
	for model_name in ordered:
		if model_name not in seen:
			seen.add(model_name)
			result.append(model_name)
	return result


def get_chat_model():
	return build_chat_model(CHAT_MODEL)
