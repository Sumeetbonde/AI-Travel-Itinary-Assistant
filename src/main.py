from fastapi import FastAPI, File, HTTPException, UploadFile
from contextlib import asynccontextmanager
from pydantic import BaseModel
import time

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    build_chat_model,
    get_chat_model_candidates,
    get_embeddings,
)
from src.ingest import ingest_pdf
from src.copilot import get_travel_copilot_instructions
from src.copilot import (
    set_itinerary,
    get_itinerary,
    update_itinerary_sections,
    get_itinerary_memory_prompt,
)
from typing import Dict, Any
from pydantic import BaseModel

TRAVEL_KEYWORDS = {
    "travel", "kerala", "KERALA", "trip", "itinerary", "destination", "tourism", "tourist",
    "hotel", "resort", "hostel", "flight", "airport", "visa", "passport",
    "booking", "transport", "train", "bus", "taxi", "metro", "cruise",
    "beach", "mountain", "museum", "landmark", "sightseeing", "guide",
    "vacation", "holiday", "stay", "accommodation", "route", "city",
    "country", "attraction", "backpacking", "check-in", "check out",
}

SECTION_KEYWORDS = {
    "hotels": {"hotel", "resort", "hostel", "stay", "accommodation", "check-in", "check out", "rent", "rate", "rating", "night"},
    "transport": {"flight", "airport", "train", "bus", "taxi", "metro", "cab", "transport"},
    "food": {"restaurant", "food", "cuisine", "breakfast", "lunch", "dinner", "cafe"},
    "attractions": {"attraction", "museum", "landmark", "beach", "sightseeing", "tour", "place to visit"},
    "budget": {"budget", "cost", "price", "expense", "currency", "per day", "estimate", "total", "package"},
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    yield

app = FastAPI(lifespan = lifespan)

class QueryRequest(BaseModel):
    query : str


def dedupe_docs(docs: list[Document]) -> list[Document]:
    seen = set()
    unique_docs = []
    for doc in docs:
        key = (
            doc.metadata.get("source", "unknown"),
            doc.metadata.get("page", "unknown"),
            doc.page_content.strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)
    return unique_docs


def extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    # LangChain / Google GenAI wrapper may expose a content field.
    content = None
    if hasattr(response, "content"):
        content = getattr(response, "content")
    elif hasattr(response, "text"):
        return getattr(response, "text")
    elif isinstance(response, dict):
        content = response.get("content") or response.get("output") or response.get("response")
    else:
        content = response

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])
                elif "text" in item:
                    text_parts.append(item["text"])
                elif "content" in item:
                    text_parts.append(extract_response_text(item["content"]))
        return "".join(text_parts).strip()
    if isinstance(content, dict):
        return extract_response_text(content.get("text") or content.get("content") or content.get("output"))

    return str(response)


def classify_model_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "generate_content_free_tier_requests" in lowered or "generate_content_free_tier_input_token_count" in lowered:
        return "Gemini free-tier quota is exhausted or not enabled for this project/model."
    if "resource_exhausted" in lowered or "quota exceeded" in lowered or "429" in lowered:
        return "Model quota exceeded for this API key/project."
    if "unavailable" in lowered or "high demand" in lowered or "503" in lowered:
        return "Model is temporarily unavailable due to high demand."
    if "not_found" in lowered or ("model" in lowered and "not found" in lowered):
        return "Configured chat model is not available for this API key/version."
    if "permission" in lowered or "403" in lowered or "unauthorized" in lowered:
        return "API key does not have permission for this model or endpoint."
    if "deadline" in lowered or "timed out" in lowered or "timeout" in lowered:
        return "Model request timed out."
    return "Model generation failed."


def _extract_snippet_for_keywords(docs: list[Document], keywords: list[str], max_chars: int = 1000) -> str:
    """Return the first matching snippet from docs for any of the keywords, truncated."""
    for doc in docs:
        text = doc.page_content
        lower = text.lower()
        for k in keywords:
            if k in lower:
                # return a contextual slice around first occurrence
                idx = lower.find(k)
                start = max(0, idx - 120)
                end = min(len(text), idx + 360)
                return text[start:end].strip()[:max_chars]
    return ""


def make_extractive_response(docs: list[Document], query: str, missing_sections: list[str]) -> str:
    """Create a PDF-grounded extractive response when LLM is unavailable.

    This returns a concise, structured Markdown string using text directly from the uploaded PDF(s).
    """
    parts = []
    parts.append("### Trip Planner (PDF-grounded extract)")
    parts.append("")
    parts.append("I couldn't generate a full model-crafted plan due to model access/quota issues. Below are relevant excerpts from your uploaded PDF(s) that address your query — use these as the authoritative source until model access is restored.")
    parts.append("")

    # Trip Summary
    summary = _extract_snippet_for_keywords(docs, ["trip summary", "summary", "overview"]) or docs[0].page_content[:500]
    parts.append("**Trip Summary (from PDF):**")
    parts.append(summary)
    parts.append("")

    # Day-wise Itinerary
    itinerary_snip = _extract_snippet_for_keywords(docs, ["day 1", "day 2", "day 3", "itinerary"]) or "(No explicit day-wise itinerary found in the retrieved excerpts.)"
    parts.append("**Day-wise Itinerary (from PDF):**")
    parts.append(itinerary_snip)
    parts.append("")

    # Hotels / Transport / Food / Budget sections: try to extract each
    hotel = _extract_snippet_for_keywords(docs, ["hotel", "resort", "accommodation"]) or "(No hotel information found in the retrieved excerpts.)"
    parts.append("**Hotels / Stay Options (from PDF):**")
    parts.append(hotel)
    parts.append("")

    transport = _extract_snippet_for_keywords(docs, ["flight", "train", "bus", "taxi", "transport"]) or "(No transport details found in the retrieved excerpts.)"
    parts.append("**Local Transport (from PDF):**")
    parts.append(transport)
    parts.append("")

    food = _extract_snippet_for_keywords(docs, ["restaurant", "food", "cuisine"]) or "(No food recommendations found in the retrieved excerpts.)"
    parts.append("**Food Recommendations (from PDF):**")
    parts.append(food)
    parts.append("")

    budget = _extract_snippet_for_keywords(docs, ["budget", "cost", "price", "estimate"]) or "(No budget estimates found in the retrieved excerpts.)"
    parts.append("**Approx Budget Tips (from PDF):**")
    parts.append(budget)
    parts.append("")

    if missing_sections:
        parts.append("**Missing Sections Detected:**")
        parts.append(", ".join(missing_sections))
        parts.append("")

    parts.append("**Note:** When model access/quota is restored I will convert these excerpts into a full professional itinerary, make suggested updates, and calculate revised costs. Lines labeled [Suggested] will indicate non-PDF recommendations.")

    return "\n\n".join(parts)

def retrieve_docs(question: str, k: int = 3) -> list[Document]:
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=get_embeddings(),
    )

    docs = vectorstore.similarity_search(question, k=k)
    return docs


def build_context(docs: list[Document]) -> str:
    if not docs:
        return ""

    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', 'unknown')}\n{doc.page_content[:1500]}"
        for doc in docs
    )


def has_travel_content(docs: list[Document], min_matches: int = 1) -> bool:
    if not docs:
        return False

    text_blob = "\n".join(doc.page_content.lower() for doc in docs)
    matches = sum(1 for keyword in TRAVEL_KEYWORDS if keyword in text_blob)
    return matches >= min_matches


def detect_missing_sections(docs: list[Document]) -> list[str]:
    if not docs:
        return list(SECTION_KEYWORDS.keys())

    text_blob = "\n".join(doc.page_content.lower() for doc in docs)
    missing = []
    for section, keywords in SECTION_KEYWORDS.items():
        if not any(keyword in text_blob for keyword in keywords):
            missing.append(section)
    return missing


def invoke_with_resilience(prompt: str) -> str:
    last_exc = None
    for model_name in get_chat_model_candidates():
        chat_model = build_chat_model(model_name)
        for attempt in range(2):
            try:
                response = chat_model.invoke(prompt)
                text = extract_response_text(response)
                if text:
                    return text
                return str(response)
            except Exception as exc:
                last_exc = exc
                message = str(exc).lower()
                retryable = "503" in message or "unavailable" in message or "high demand" in message
                if retryable and attempt == 0:
                    time.sleep(1.2)
                    continue
                break

    if last_exc:
        raise last_exc

    raise RuntimeError("No chat model could be invoked")

@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        docs = dedupe_docs(retrieve_docs(query))
        context = build_context(docs)

        if not context:
            return {
                "response": (
                    "### Travel Planner Status\n\n"
                    "I am ready to build your trip plan, but I need a travel PDF in the knowledge base first.\n\n"
                    "Please do the following:\n"
                    "- Upload a travel guide PDF from the sidebar.\n"
                    "- Ask a specific question such as destination, duration, or budget.\n"
                    "- Re-run your query after upload completes."
                )
            }

        if not has_travel_content(docs):
            return {
                "response": (
                    "### Travel Relevance Check\n\n"
                    "I reviewed the retrieved PDF content and it does not appear to contain travel-related information.\n\n"
                    "To generate a complete trip plan, please upload a document that includes details such as:\n"
                    "- Destinations or itinerary\n"
                    "- Transport or visa notes\n"
                    "- Hotel or stay options\n"
                    "- Local attractions, food, or budget guidance\n\n"
                    "Once a travel-focused PDF is uploaded, I will generate a professional plan immediately."
                )
            }

        missing_sections = detect_missing_sections(docs)
        missing_text = ", ".join(missing_sections) if missing_sections else "none"

        # Get the travel-copilot instructions and prepend to the generation prompt
        copilot_instructions = get_travel_copilot_instructions()
        memory_prompt = get_itinerary_memory_prompt()

        prompt = f"""
    {copilot_instructions}

    {memory_prompt}
    You are an expert travel planner.

    Use the PDF context as the primary source of truth.
        Travel relevance has already been validated. Continue only as a travel planner.
        If some sections are missing in the PDF ({missing_text}), supplement them using your general travel knowledge.
    When you supplement from your own knowledge, mark those lines with: [LLM Suggested].
    Do not invent PDF facts.

    Context:
    {context}

    Question:
    {query}

    Return the answer as a practical trip plan with these sections:
    1) Trip Summary
    2) Day-wise Itinerary
    3) Hotels / Stay Options
    4) Local Transport
    5) Food Recommendations
    6) Approx Budget Tips
    7) Important Notes

    If a section is unavailable in both PDF and your general travel knowledge, state that clearly.

    For missing sections, you must still provide practical suggestions using LLM knowledge:
    - Hotels / Stay Options: include suggested area, stay type, estimated rent per night, and a typical rating band.
    - Local Transport: include practical options, expected fare ranges, and when to use each.
    - Approx Budget Tips: provide a simple daily and trip-level estimate in a clear bullet breakdown.

    If the user does not specify duration, assume a 3-day trip and mention that assumption.

    Formatting requirements:
    - Use clean Markdown headings for each section.
    - Start with a short professional paragraph.
    - Use bullet points for actionable recommendations.
    - Keep each bullet crisp and practical.
    - End with a brief closing note.
    """.strip()

        answer = invoke_with_resilience(prompt)
        return {"response": answer}
    except Exception as exc:
        if 'docs' in locals() and docs:
            reason = classify_model_error(exc)
            detail_message = str(exc)
            if "free-tier quota" in reason.lower() or "quota exceeded" in reason.lower():
                reason += (
                    " Your Gemini project currently reports a free-tier quota limit of 0 for content generation. "
                    "This means the API key is valid, but the Google Cloud project does not currently have Gemini free-tier quota enabled for this model. "
                    "Please enable Gemini API quota in the Google Cloud project, use a different eligible project, or wait until quota is available."
                )

                try:
                    missing_sections = detect_missing_sections(docs)
                    extractive = make_extractive_response(docs, query, missing_sections)
                    return {
                        "response": (
                            extractive +
                            "\n\n---\n\n"
                            "**Debug note:** "
                            f"{reason}\n\n"
                            "If you want, use a different Google Cloud project or check the Gemini dashboard for quota status."
                        )
                    }
                except Exception:
                    # fallback to original excerpts if extractive construction fails
                    fallback = "\n\n".join(
                        f"From {doc.metadata.get('source', 'uploaded PDF')} page {doc.metadata.get('page', 'unknown')}:\n{doc.page_content[:700]}"
                        for doc in docs[:2]
                    )
                    return {
                        "response": (
                            "### Trip Planner Update\n\n"
                            "I could not complete the model-generated plan at the moment.\n\n"
                            f"Reason: {reason}\n\n"
                            "You can still use the following relevant excerpts from your uploaded PDF:\n\n"
                            f"{fallback}\n\n"
                            "Next step:\n"
                            "- Please retry after resolving model access or quota, and I will produce the full professional itinerary."
                        )
                    }

            # Generic fallback (non-quota reasons)
            fallback = "\n\n".join(
                f"From {doc.metadata.get('source', 'uploaded PDF')} page {doc.metadata.get('page', 'unknown')}:\n{doc.page_content[:700]}"
                for doc in docs[:2]
            )
            return {
                "response": (
                    "### Trip Planner Update\n\n"
                    "I could not complete the model-generated plan at the moment.\n\n"
                    f"Reason: {reason}\n\n"
                    "You can still use the following relevant excerpts from your uploaded PDF:\n\n"
                    f"{fallback}\n\n"
                    "Next step:\n"
                    "- Please retry after resolving model access or quota, and I will produce the full professional itinerary."
                )
            }

        raise HTTPException(status_code=500, detail=f"Failed to answer query: {exc}") from exc


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    
    try:
        result = await ingest_pdf(file)

        # Build a lightweight itinerary memory from upload result.
        # Heuristic: search chunk contents for likely sections.
        raw_chunks = result.get("chunk_contents", [])
        text_blob = "\n\n".join(raw_chunks)

        def find_section(keywords):
            for chunk in raw_chunks:
                lower = chunk.lower()
                if any(k in lower for k in keywords):
                    return chunk
            return None

        itinerary = {
            "source": result.get("filename"),
            "pages": result.get("pages"),
            "chunks": result.get("chunks"),
            "raw_text": text_blob[:20000],
            "trip_summary": find_section(["trip summary", "summary", "overview"]),
            "day_wise_itinerary": find_section(["day", "day 1", "day 2", "itinerary"]),
            "hotels": find_section(["hotel", "resort", "stay", "accommodation"]),
            "transport": find_section(["flight", "train", "bus", "taxi", "transport"]),
            "food": find_section(["food", "restaurant", "cuisine"]),
            "budget": find_section(["budget", "cost", "price", "estimate"]),
            "source_type": "pdf",
        }

        # Persist to in-memory copilot memory
        try:
            set_itinerary(itinerary)
        except Exception:
            # non-fatal; continue
            pass

        return {"message": "file processed successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}") from e


class ItineraryUpdateRequest(BaseModel):
    updates: Dict[str, Any]


@app.get("/itinerary/get")
async def itinerary_get():
    """Return the current in-memory itinerary or an empty object."""
    it = get_itinerary()
    return {"itinerary": it or {}}


@app.post("/itinerary/update")
async def itinerary_update(req: ItineraryUpdateRequest):
    """Apply partial updates to the in-memory itinerary and return the changed keys and current memory."""
    try:
        changed = update_itinerary_sections(req.updates)
        return {"changed": changed, "itinerary": get_itinerary()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update itinerary: {e}") from e
