from dotenv import load_dotenv
from pathlib import Path
import os, traceback
load_dotenv(dotenv_path=Path('.').resolve() / '.env')
print('GEMINI_CHAT_MODEL=', os.getenv('GEMINI_CHAT_MODEL'))
print('GEMINI_API_KEY=', 'SET' if os.getenv('GEMINI_API_KEY') else 'MISSING')
from src.config import build_chat_model, get_chat_model_candidates
print('candidates=', get_chat_model_candidates())
for model_name in get_chat_model_candidates():
    try:
        print('testing model:', model_name)
        model = build_chat_model(model_name)
        print('model created', model.model)
        r = model.invoke('Reply with exactly: OK')
        print('invoke response:', getattr(r, 'content', r))
    except Exception as e:
        print(f'ERROR for {model_name}: {type(e).__name__} - {str(e)[:1200]}')
