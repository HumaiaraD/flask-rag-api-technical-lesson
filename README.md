# Technical Lesson — Building a Flask RAG API with Chroma and Ollama

## Introduction

A backend RAG API does more than send a user question to a model. It retrieves approved context, builds a structured prompt, calls a model, and returns an answer with source information users and developers can inspect.

In this lesson, you will build a local Flask API that uses **Chroma** as the vector database layer and **Ollama** as the local AI service. You will produce a `POST /api/ask` endpoint within a facilities operations scenario using **Identify → Assemble → Execute → Verify** with request validation, document seeding, vector retrieval, prompt construction, model generation, source attribution, and `curl` checks.

## Scenario

You are a junior backend developer on an internal tools team. The facilities operations team manages building access, visitor registration, room setup, office maintenance, and equipment repair requests.

Employees often ask natural-language questions such as:

> “Can I get building access after 7 PM?”

The approved information exists in facilities documents, but employees do not always know which policy to search for. Your task is to build a small RAG API that retrieves relevant facilities context from Chroma, asks a local model to answer using that context, and returns an answer with sources.

## Tools and Resources

- Python 3.10 or newer
- Visual Studio Code or another code editor
- Terminal or integrated terminal
- `pipenv`
- Flask
- ChromaDB
- Ollama installed and running locally
- An embedding model, such as `embeddinggemma`
- A generation model, such as `llama3.2`
- `curl` for manual API verification



## Instructions

### Set Up

Set up the local Ollama models:

```bash
ollama pull embeddinggemma
ollama pull llama3.2
ollama run llama3.2 "Hello"
```

Install the project dependencies in the Pipfile and enter the virtual environment:

```bash
pipenv install
pipenv shell
```

You will seed Chroma after completing the Chroma steps:

```bash
python seed_chroma.py
```

You will run the Flask API after completing the route:

```bash
flask --app app run --debug
```

You will test the endpoint with `curl`:

```bash
curl -i -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I get building access after 7 PM?"}'
```

### Step 1: Identify the RAG API goal and output contract

Next, you will define what the endpoint needs to solve and what information it should return.

#### Action

**Create a short planning note that identifies the user need, searchable content, endpoint route, expected response, and fallback behavior.**

#### Breakdown

Open `planning_notes.md`.

Start by identifying the user role:

```text
User or role:
Employees asking facilities operations questions.
```

Add the business problem:

```text
Business problem:
Employees need quick answers from approved facilities information instead of general AI guesses.
```

Add the user question that the backend will process:

```text
User question:
"Can I get building access after 7 PM?"
```

Add the content the backend can search:

```text
Searchable content:
Facilities documents about building access, badges, visitors, room setup, maintenance, and equipment repair.
```

Add the endpoint route:

```text
Endpoint route:
POST /api/ask
```

Add the expected successful response:

```text
Expected successful response:
A JSON object with an answer and a sources list.
```

Add the fallback behavior:

```text
Fallback behavior:
If retrieved context is missing or weak, return a safe answer that says there is not enough approved facilities context.
```

Add the verification goal:

```text
Verification goal:
The answer should use retrieved facilities context and include traceable source metadata.
```

Your completed planning note should look like this:

```text
User or role:
Employees asking facilities operations questions.

Business problem:
Employees need quick answers from approved facilities information instead of general AI guesses.

User question:
"Can I get building access after 7 PM?"

Searchable content:
Facilities documents about building access, badges, visitors, room setup, maintenance, and equipment repair.

Endpoint route:
POST /api/ask

Expected successful response:
A JSON object with an answer and a sources list.

Fallback behavior:
If retrieved context is missing or weak, return a safe answer that says there is not enough approved facilities context.

Verification goal:
The answer should use retrieved facilities context and include traceable source metadata.
```

You should have a clear output contract for the endpoint.

The output contract helps you design the code. A RAG endpoint is not only about generating text. It needs to return enough source information for a user, frontend, test suite, or future developer to inspect how the answer was produced.

#### Step Hint

Do not describe the feature as “ask AI a question.” Describe the full backend flow: question → retrieval → context → prompt → model response → source-backed JSON.

#### Step Feedback

This step is strong when it connects the endpoint to a real user need and names the expected request, response, and fallback behavior.

---

### Step 2: Assemble the starter files and responsibilities

Next, you will inspect the starter project and identify which file owns each part of the workflow.

#### Action

**Review the starter files and write a short note explaining each file's responsibility.**

#### Breakdown

The project contains these files:

```text
m5-flask-rag-api-lesson/
├── Pipfile
├── README.md
├── app.py
├── ai_client.py
├── chroma_store.py
├── documents.py
├── planning_notes.md
├── rag_service.py
├── seed_chroma.py
└── .gitignore
```

Use this responsibility map:

| File | Responsibility |
|---|---|
| `documents.py` | Stores the small facilities knowledge base used in this lesson. |
| `ai_client.py` | Sends embedding and generation requests to local Ollama models. |
| `chroma_store.py` | Creates the Chroma collection, seeds documents, and retrieves relevant results. |
| `seed_chroma.py` | Runs the seeding workflow from the terminal. |
| `rag_service.py` | Builds context, sources, prompts, fallback behavior, and the full RAG response. |
| `app.py` | Defines the Flask app and `POST /api/ask` route. |
| `planning_notes.md` | Gives you a place to write planning, verification, and reflection notes. |

You should be able to explain where each part of the RAG workflow belongs before you implement code.

This separation keeps the route from doing too much. The Flask route coordinates the request and response, but retrieval, prompt construction, and model calls live in focused helper files.

#### Step Hint

If a file starts to own too many jobs, the API becomes harder to debug. Keep route logic, retrieval logic, prompt logic, and model-service logic separate.

#### Step Feedback

This step is strong when you can explain how data moves through the files from user question to source-backed answer.

---

### Step 3: Execute one embedding request

Next, you will create the helper that converts one text input into one embedding vector.

#### Action

**In `ai_client.py`, complete `get_embedding()` so it calls Ollama and returns one embedding.**

#### Breakdown

Open `ai_client.py`.

Find the starter function:

```python
def get_embedding(text: str) -> List[float]:
    """Return one embedding vector for one text input."""
    # TODO: Step 3 - call ollama.embed(...) and return the first embedding.
    raise NotImplementedError("Complete get_embedding() in Step 3.")
```

Replace the TODO with this implementation:

```python
def get_embedding(text: str) -> List[float]:
    """Return one embedding vector for one text input."""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        return response["embeddings"][0]
    except Exception as exc:
        raise ModelServiceError(
            "The local AI model service is unavailable. "
            "Confirm Ollama is running and the embedding model has been pulled."
        ) from exc
```

This helper sends one text string to the local embedding model and returns the first embedding vector.

The response stores embeddings in a list because the API can return embeddings for one input or multiple inputs. In this lesson, each call sends one text value, so the vector you need is the first item.

#### Step Output

You should now have a reusable function that accepts one text string and returns one list of numbers.

#### Step Explanation

Embedding generation is the first bridge between natural language and vector search. Chroma will store document embeddings and compare them with a query embedding later.

#### Step Hint

Use the same embedding model for documents and user questions. Do not embed stored documents with one model and queries with a different model.

#### Step Feedback

This step is strong when `get_embedding()` has one clear job: convert text into one embedding vector and raise a helpful service error if Ollama is unavailable.

---

### Step 4: Execute one model response

Next, you will create the helper that sends one structured prompt to the generation model.

#### Action

**In `ai_client.py`, complete `generate_answer()` so it calls Ollama and returns generated text.**

#### Breakdown

Find the starter function:

```python
def generate_answer(prompt: str) -> str:
    """Return one generated answer for one prompt."""
    # TODO: Step 4 - call ollama.generate(...) and return the response text.
    raise NotImplementedError("Complete generate_answer() in Step 4.")
```

Replace the TODO with this implementation:

```python
def generate_answer(prompt: str) -> str:
    """Return one generated answer for one prompt."""
    try:
        response = ollama.generate(model=GENERATION_MODEL, prompt=prompt)
        return response["response"].strip()
    except Exception as exc:
        raise ModelServiceError(
            "The local AI model service is unavailable. "
            "Confirm Ollama is running and the generation model has been pulled."
        ) from exc
```

This helper sends a complete prompt to `llama3.2` and returns the generated response text.

#### Step Output

You should now have two focused AI client helpers:

```text
get_embedding(text) → embedding vector
generate_answer(prompt) → generated answer text
```

#### Step Explanation

The application will use embeddings for retrieval and generation for the final answer. Keeping both calls in `ai_client.py` prevents Ollama-specific code from being scattered across the Flask route and RAG service.

#### Step Hint

Do not pass the raw user question directly to `generate_answer()` in the final endpoint. The RAG service will build a structured prompt that includes instructions, context, the question, and response rules.

#### Step Feedback

This step is strong when `generate_answer()` returns clean text and leaves prompt design to the RAG service.

---

### Step 5: Assemble the persistent Chroma collection

Next, you will create the Chroma collection that stores facilities document embeddings and metadata.

#### Action

**In `chroma_store.py`, complete `get_collection()` so it creates or resets the lesson collection.**

#### Breakdown

Open `chroma_store.py`.

Find the starter function:

```python
def get_collection(reset: bool = False):
    """Return the lesson Chroma collection."""
    # TODO: Step 5 - create a PersistentClient and get or reset the collection.
    raise NotImplementedError("Complete get_collection() in Step 5.")
```

Replace the TODO with this implementation:

```python
def get_collection(reset: bool = False):
    """Return the lesson Chroma collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
```

This function creates a persistent Chroma client that stores data in the local `chroma_db/` folder. When `reset=True`, the old lesson collection is deleted before a fresh collection is created.

#### Step Output

You should now have a helper that returns the Chroma collection used by the lesson.

#### Step Explanation

A vector database stores embeddings and lets the backend retrieve similar items later. In this lesson, Chroma owns the vector storage and nearest-neighbor search step.

#### Step Hint

Do not commit the generated `chroma_db/` folder to GitHub. The `.gitignore` file already excludes it.

#### Step Feedback

This step is strong when the collection setup is reusable and the reset behavior lets you reseed the same documents during development.

---

### Step 6: Execute document seeding

Next, you will embed each facilities document and store it in Chroma with metadata.

#### Action

**In `chroma_store.py`, complete `seed_collection()`. Then run `seed_chroma.py`.**

#### Breakdown

Find the starter function:

```python
def seed_collection() -> int:
    """Store the facility documents, embeddings, and metadata in Chroma."""
    # TODO: Step 6 - embed each document and add it to the collection.
    raise NotImplementedError("Complete seed_collection() in Step 6.")
```

Replace the TODO with this implementation:

```python
def seed_collection() -> int:
    """Store the facility documents, embeddings, and metadata in Chroma."""
    collection = get_collection(reset=True)

    ids = [document["id"] for document in FACILITY_DOCUMENTS]
    documents = [
        f"{document['title']}. {document['text']}"
        for document in FACILITY_DOCUMENTS
    ]
    metadatas = [
        {
            "title": document["title"],
            "category": document["category"],
        }
        for document in FACILITY_DOCUMENTS
    ]
    embeddings = [get_embedding(document_text) for document_text in documents]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection.count()
```

Run the seeding script from inside your `pipenv shell`:

```bash
python seed_chroma.py
```

Your output should look like this:

```text
Seeded 6 facility documents into Chroma.
```

Exact timing will depend on your local machine and Ollama model.

#### Step Output

You should now have a local `chroma_db/` folder containing the seeded facilities collection.

#### Step Explanation

Seeding turns approved source text into searchable vector database records. Each record keeps its ID, document text, embedding, and metadata connected so later answers can include source information.

#### Step Hint

If this step fails, check that Ollama is running and that you pulled `embeddinggemma`.

#### Step Feedback

This step is strong when Chroma stores all six facilities documents and the terminal confirms the collection count.

---

### Step 7: Execute retrieval from Chroma

Next, you will query Chroma with a user question and normalize the raw Chroma output into readable Python dictionaries.

#### Action

**In `chroma_store.py`, complete `query_collection()`. Then run a one-line retrieval check.**

#### Breakdown

Find the starter function:

```python
def query_collection(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Retrieve the most relevant Chroma results for a user question."""
    # TODO: Step 7 - embed the question, query Chroma, and normalize the results.
    raise NotImplementedError("Complete query_collection() in Step 7.")
```

Replace the TODO with this implementation:

```python
def query_collection(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Retrieve the most relevant Chroma results for a user question."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = get_embedding(question)
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    ids = raw_results.get("ids") or [[]]
    documents = raw_results.get("documents") or [[]]
    metadatas = raw_results.get("metadatas") or [[]]
    distances = raw_results.get("distances") or [[]]

    results: List[Dict[str, Any]] = []

    for index, document_id in enumerate(ids[0]):
        metadata = metadatas[0][index] or {}
        distance = float(distances[0][index])
        score = 1 - distance

        results.append(
            {
                "id": document_id,
                "title": metadata.get("title", "Untitled source"),
                "category": metadata.get("category", "uncategorized"),
                "text": documents[0][index],
                "score": score,
            }
        )

    return results
```

Run a quick retrieval check:

```bash
python -c "from chroma_store import query_collection; print(query_collection('Can I get into the building after 7 PM?'))"
```

Your output should show a list of result dictionaries. The top result should usually be related to after-hours building access.

#### Step Output

You should now have retrieval output with:

- `id`
- `title`
- `category`
- `text`
- `score`

#### Step Explanation

Chroma returns nested results. The helper normalizes those results into a structure that the RAG service can use for context, source attribution, and relevance checks.

#### Step Hint

Do not return raw embeddings in API responses. Users need source text and metadata, not long vectors.

#### Step Feedback

This step is strong when the result structure is predictable and the top result matches the meaning of the user question.

---

### Step 8: Assemble retrieved context and source metadata

Next, you will prepare retrieved results for two different uses: prompt context and API sources.

#### Action

**In `rag_service.py`, complete `has_usable_context()`, `format_context()`, and `build_sources()`.**

#### Breakdown

Open `rag_service.py`.

Find `has_usable_context()`:

```python
def has_usable_context(results: List[Dict[str, Any]]) -> bool:
    """Return True when retrieval produced context strong enough to use."""
    # TODO: Step 8 - check whether the top result exists and meets the relevance threshold.
    raise NotImplementedError("Complete has_usable_context() in Step 8.")
```

Replace the TODO with this implementation:

```python
def has_usable_context(results: List[Dict[str, Any]]) -> bool:
    """Return True when retrieval produced context strong enough to use."""
    if not results:
        return False

    top_score = float(results[0].get("score", 0.0))
    return top_score >= MIN_RELEVANCE_SCORE
```

Find `format_context()`:

```python
def format_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved results as labeled context for the prompt."""
    # TODO: Step 8 - format each result with source metadata and source text.
    raise NotImplementedError("Complete format_context() in Step 8.")
```

Replace the TODO with this implementation:

```python
def format_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved results as labeled context for the prompt."""
    context_blocks = []

    for index, result in enumerate(results, start=1):
        context_blocks.append(
            "\\n".join(
                [
                    f"Source {index}",
                    f"ID: {result['id']}",
                    f"Title: {result['title']}",
                    f"Category: {result['category']}",
                    f"Text: {result['text']}",
                ]
            )
        )

    return "\\n\\n".join(context_blocks)
```

Find `build_sources()`:

```python
def build_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return source metadata for the API response."""
    # TODO: Step 8 - return user-readable source metadata without raw embeddings.
    raise NotImplementedError("Complete build_sources() in Step 8.")
```

Replace the TODO with this implementation:

```python
def build_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return source metadata for the API response."""
    return [
        {
            "id": result["id"],
            "title": result["title"],
            "category": result["category"],
            "score": round(float(result["score"]), 4),
        }
        for result in results
    ]
```

#### Step Output

You should now have helpers that:

```text
retrieved results → labeled prompt context
retrieved results → source metadata for JSON
retrieved results → usable-context decision
```

#### Step Explanation

Prompt context and API sources are related, but they are not the same output. The model needs labeled source text inside the prompt. The API user needs source metadata in the response.

#### Step Hint

Do not remove source IDs while formatting context. If the model answer is questionable, the user or developer needs to trace the answer back to a source.

#### Step Feedback

This step is strong when the retrieved context is labeled clearly and the returned sources do not expose raw embeddings.

---

### Step 9: Assemble the structured prompt

Next, you will build the prompt that tells the model how to use retrieved context.

#### Action

**In `rag_service.py`, complete `build_prompt()` with instructions, context, question, and response rules.**

#### Breakdown

Find the starter function:

```python
def build_prompt(question: str, results: List[Dict[str, Any]]) -> str:
    """Build a structured prompt from the user question and retrieved context."""
    # TODO: Step 9 - combine instructions, labeled context, user question, and response rules.
    raise NotImplementedError("Complete build_prompt() in Step 9.")
```

Replace the TODO with this implementation:

```python
def build_prompt(question: str, results: List[Dict[str, Any]]) -> str:
    """Build a structured prompt from the user question and retrieved context."""
    context = format_context(results)

    return f"""
You are a facilities operations assistant for employees.

Use only the approved facilities context below to answer the user's question.
If the context does not contain enough information, say:
"I do not have enough approved facilities context to answer that question."

Approved facilities context:
{context}

User question:
{question}

Response requirements:
- Answer in 2 to 4 sentences.
- Use a helpful and professional tone.
- Do not invent policies, form names, phone numbers, timelines, or approval steps.
- Base the answer only on the approved context.
""".strip()
```

This prompt gives the model boundaries. It labels the approved context, includes the user's question, and tells the model what to do if the context is not enough.

#### Step Output

You should now have a prompt builder that produces a complete model input.

#### Step Explanation

This is where prompt and context engineering become part of backend development. The backend is not only passing text to a model. It is assembling the instructions and context the model should use.

#### Step Hint

Do not pass retrieved chunks as an unlabeled blob. Label the context so the model and the developer can see which source text was included.

#### Step Feedback

This step is strong when the prompt includes instructions, labeled context, the user question, response requirements, and fallback behavior.

---

### Step 10: Execute the full RAG service

Next, you will connect retrieval, fallback behavior, prompt construction, generation, and source formatting.

#### Action

**In `rag_service.py`, complete `answer_question()`.**

#### Breakdown

Find the starter function:

```python
def answer_question(question: str) -> Dict[str, Any]:
    """Run retrieval, prompt construction, generation, and source formatting."""
    # TODO: Step 10 - coordinate the RAG workflow.
    raise NotImplementedError("Complete answer_question() in Step 10.")
```

Replace the TODO with this implementation:

```python
def answer_question(question: str) -> Dict[str, Any]:
    """Run retrieval, prompt construction, generation, and source formatting."""
    results = query_collection(question)

    if not has_usable_context(results):
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

    prompt = build_prompt(question, results)
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": build_sources(results),
    }
```

This function coordinates the RAG workflow:

```text
question → Chroma retrieval → relevance check → prompt → Ollama answer → source-backed response
```

#### Step Output

You should now have one service function that returns a response dictionary with:

- `answer`
- `sources`

#### Step Explanation

The service layer keeps the Flask route clean. The route should not need to know how to format context, build prompts, check retrieval quality, or call Ollama.

#### Step Hint

Keep fallback behavior before the model call. If retrieval does not provide useful context, the backend should not ask the model to guess.

#### Step Feedback

This step is strong when `answer_question()` coordinates the workflow without hiding source information or skipping the relevance check.

---

### Step 11: Build the Flask `/api/ask` route

Next, you will expose the RAG service through a Flask API endpoint.

#### Action

**In `app.py`, complete the `POST /api/ask` route.**

#### Breakdown

Open `app.py`.

Find the starter route:

```python
@app.post("/api/ask")
def ask():
    """Accept a question and return a source-backed RAG response."""
    # TODO: Step 11 - validate the request, call answer_question(), and return JSON.
    return jsonify({"message": "Complete /api/ask in Step 11."}), 501
```

Replace the TODO with this implementation:

```python
@app.post("/api/ask")
def ask():
    """Accept a question and return a source-backed RAG response."""
    payload = request.get_json(silent=True) or {}
    question = payload.get("question")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Question is required."}), 400

    try:
        response = answer_question(question.strip())
    except ModelServiceError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify(response), 200
```

This route reads the JSON request, validates the question, calls the RAG service, handles model-service errors, and returns structured JSON.

#### Step Output

You should now have a working endpoint with this contract:

```http
POST /api/ask
Content-Type: application/json
```

```json
{
  "question": "Can I get building access after 7 PM?"
}
```

Successful responses should include:

```json
{
  "answer": "...",
  "sources": [
    {
      "id": "FAC-102",
      "title": "After-Hours Building Access",
      "category": "access",
      "score": 0.8123
    }
  ]
}
```

### Step Explanation

The route coordinates the web request. It should validate input, call the service layer, and return a predictable response. It should not build prompts or query Chroma directly.

#### Step Hint

Use `/api/ask`, not `/ask`. Keeping API routes namespaced helps when a React frontend later uses `/` for the main page and `/api/...` for backend routes.

#### Step Feedback

This step is strong when the route returns `400` for blank input, `503` for local model-service problems, and `200` for completed RAG responses.

---

### Step 12: Verify the endpoint with `curl`

Next, you will seed Chroma, run Flask, and test the endpoint behavior.

#### Action

**Run the project and verify valid input, blank input, source metadata, and fallback behavior.**

#### Breakdown

Seed Chroma:

```bash
python seed_chroma.py
```

Start the Flask app:

```bash
flask --app app run --debug
```

Open a second terminal, enter the same project folder, and test a valid facilities question:

```bash
curl -i -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I get building access after 7 PM?"}'
```

Your response should have status `200` and a JSON body with `answer` and `sources`.

Example response pattern:

```json
{
  "answer": "Employees may enter the building after 7 PM only when after-hours access has been enabled on their badge. Requests should be submitted before the needed date and approved by the employee's manager.",
  "sources": [
    {
      "category": "access",
      "id": "FAC-102",
      "score": 0.8123,
      "title": "After-Hours Building Access"
    }
  ]
}
```

Exact answer wording and scores may vary by local model.

Test blank input:

```bash
curl -i -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": ""}'
```

Your response should have status `400`:

```json
{
  "error": "Question is required."
}
```

Test a question that may not be covered by the facilities context:

```bash
curl -i -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the company retirement plan contribution limits?"}'
```

If the top retrieval score is below the threshold, your response should use the safe fallback:

```json
{
  "answer": "I do not have enough approved facilities context to answer that question.",
  "sources": []
}
```

If your local model still retrieves a facilities source for this question, review the returned source and decide whether your relevance threshold needs adjustment.

#### Step Output

You should have evidence that the endpoint:

- accepts a valid JSON question,
- returns an answer and sources,
- rejects blank questions,
- uses a safe fallback when context is missing or weak,
- returns traceable source metadata.

#### Step Explanation

Manual API checks help you verify the same behaviors a frontend, test suite, or deployed service would depend on. The goal is not only that the model answers. The goal is that the backend returns predictable, source-backed API behavior.

#### Step Hint

Use `curl -i` while testing so you can inspect HTTP status codes as well as JSON response bodies.

#### Step Feedback

This step is strong when you can explain what happened for each request and whether the returned answer was grounded in the retrieved source.

---

### Step 13: Reflect on how the workflow prepares you for the lab

Next, you will connect this technical lesson to the independent lab.

#### Action

**Write a short reflection that explains what each layer owns in the RAG API.**

#### Breakdown

In `planning_notes.md`, answer these questions:

```text
1. What does Chroma own in this workflow?
2. What does the prompt builder own?
3. What does the Flask route coordinate?
4. Why does the API response include sources?
5. What would break if the route sent the user question directly to the model?
```

A completed reflection might look like this:

```text
Chroma owns vector storage and retrieval. It stores embedded facilities documents and returns the most relevant source text for a user question.

The prompt builder owns the instructions, labeled context, user question, response requirements, and fallback rule. It helps the model answer from approved source text instead of guessing.

The Flask route coordinates the HTTP request and response. It validates the JSON body, calls the RAG service, handles model-service errors, and returns structured JSON.

The API response includes sources so users and developers can inspect where the answer came from. If the route sent the user question directly to the model, the answer might sound confident but would not be grounded in approved facilities context.
```

You should have a reflection that explains the workflow in your own words.

#### Step Hint

Focus your reflection on responsibilities, not only tool names. The same pattern can apply to other tools later.

#### Step Feedback

This step is strong when it clearly separates retrieval, prompt construction, generation, routing, and source attribution.

---

## Considerations

### Common issues

| Issue | Why it matters | How to respond |
|---|---|---|
| Ollama is not running | Embedding and generation calls will fail. | Start Ollama and rerun the command. |
| Model not pulled | The Python client cannot use a missing local model. | Run `ollama pull embeddinggemma` and `ollama pull llama3.2`. |
| Chroma not seeded | Retrieval returns no useful context. | Run `python seed_chroma.py` before starting the API. |
| Missing metadata | Responses cannot show where the answer came from. | Keep IDs, titles, categories, and source text connected. |
| Weak threshold | Unrelated questions may still get answers. | Adjust `MIN_RELEVANCE_SCORE` after testing retrieval quality. |
| Overly strict threshold | Good questions may fall back too often. | Test known in-scope questions and tune the threshold. |
| Prompt lacks fallback rule | The model may guess when context is weak. | Include a clear “not enough context” instruction. |
| Route does too much work | The API becomes harder to debug and test. | Keep retrieval and prompt logic in service/helper files. |

### Decision point: what belongs in the route?

| Logic | Belongs in `app.py`? | Better location |
|---|---:|---|
| Read JSON request | Yes | `app.py` |
| Validate required `question` field | Yes | `app.py` |
| Query Chroma | No | `chroma_store.py` |
| Format context | No | `rag_service.py` |
| Build the prompt | No | `rag_service.py` |
| Call Ollama | No | `ai_client.py` |
| Return HTTP status and JSON | Yes | `app.py` |

A clean route is easier to test, debug, and connect to a frontend later.

---

## Complete Code Checkpoint

Use these completed files to check your work after you have built the project step by step.

### `ai_client.py`

```python
"""Ollama helpers for the Flask RAG API technical lesson."""

from typing import List

import ollama


EMBEDDING_MODEL = "embeddinggemma"
GENERATION_MODEL = "llama3.2"


class ModelServiceError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


def get_embedding(text: str) -> List[float]:
    """Return one embedding vector for one text input."""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        return response["embeddings"][0]
    except Exception as exc:
        raise ModelServiceError(
            "The local AI model service is unavailable. "
            "Confirm Ollama is running and the embedding model has been pulled."
        ) from exc


def generate_answer(prompt: str) -> str:
    """Return one generated answer for one prompt."""
    try:
        response = ollama.generate(model=GENERATION_MODEL, prompt=prompt)
        return response["response"].strip()
    except Exception as exc:
        raise ModelServiceError(
            "The local AI model service is unavailable. "
            "Confirm Ollama is running and the generation model has been pulled."
        ) from exc
```

### `chroma_store.py`

```python
"""Chroma setup, seeding, and retrieval helpers for the Flask RAG API lesson."""

from typing import Any, Dict, List

import chromadb

from ai_client import get_embedding
from documents import FACILITY_DOCUMENTS


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "facility_documents"
TOP_K = 3


def get_collection(reset: bool = False):
    """Return the lesson Chroma collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def seed_collection() -> int:
    """Store the facility documents, embeddings, and metadata in Chroma."""
    collection = get_collection(reset=True)

    ids = [document["id"] for document in FACILITY_DOCUMENTS]
    documents = [
        f"{document['title']}. {document['text']}"
        for document in FACILITY_DOCUMENTS
    ]
    metadatas = [
        {
            "title": document["title"],
            "category": document["category"],
        }
        for document in FACILITY_DOCUMENTS
    ]
    embeddings = [get_embedding(document_text) for document_text in documents]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection.count()


def query_collection(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Retrieve the most relevant Chroma results for a user question."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = get_embedding(question)
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    ids = raw_results.get("ids") or [[]]
    documents = raw_results.get("documents") or [[]]
    metadatas = raw_results.get("metadatas") or [[]]
    distances = raw_results.get("distances") or [[]]

    results: List[Dict[str, Any]] = []

    for index, document_id in enumerate(ids[0]):
        metadata = metadatas[0][index] or {}
        distance = float(distances[0][index])
        score = 1 - distance

        results.append(
            {
                "id": document_id,
                "title": metadata.get("title", "Untitled source"),
                "category": metadata.get("category", "uncategorized"),
                "text": documents[0][index],
                "score": score,
            }
        )

    return results
```

### `rag_service.py`

```python
"""RAG service helpers for retrieval, prompt construction, and answer generation."""

from typing import Any, Dict, List

from ai_client import generate_answer
from chroma_store import query_collection


MIN_RELEVANCE_SCORE = 0.25
FALLBACK_ANSWER = "I do not have enough approved facilities context to answer that question."


def has_usable_context(results: List[Dict[str, Any]]) -> bool:
    """Return True when retrieval produced context strong enough to use."""
    if not results:
        return False

    top_score = float(results[0].get("score", 0.0))
    return top_score >= MIN_RELEVANCE_SCORE


def format_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved results as labeled context for the prompt."""
    context_blocks = []

    for index, result in enumerate(results, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"Source {index}",
                    f"ID: {result['id']}",
                    f"Title: {result['title']}",
                    f"Category: {result['category']}",
                    f"Text: {result['text']}",
                ]
            )
        )

    return "\n\n".join(context_blocks)


def build_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return source metadata for the API response."""
    return [
        {
            "id": result["id"],
            "title": result["title"],
            "category": result["category"],
            "score": round(float(result["score"]), 4),
        }
        for result in results
    ]


def build_prompt(question: str, results: List[Dict[str, Any]]) -> str:
    """Build a structured prompt from the user question and retrieved context."""
    context = format_context(results)

    return f"""
You are a facilities operations assistant for employees.

Use only the approved facilities context below to answer the user's question.
If the context does not contain enough information, say:
"I do not have enough approved facilities context to answer that question."

Approved facilities context:
{context}

User question:
{question}

Response requirements:
- Answer in 2 to 4 sentences.
- Use a helpful and professional tone.
- Do not invent policies, form names, phone numbers, timelines, or approval steps.
- Base the answer only on the approved context.
""".strip()


def answer_question(question: str) -> Dict[str, Any]:
    """Run retrieval, prompt construction, generation, and source formatting."""
    results = query_collection(question)

    if not has_usable_context(results):
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

    prompt = build_prompt(question, results)
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": build_sources(results),
    }
```

### `app.py`

```python
"""Flask app for the local Chroma + Ollama RAG API lesson."""

from flask import Flask, jsonify, request

from ai_client import ModelServiceError
from rag_service import answer_question


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    @app.post("/api/ask")
    def ask():
        """Accept a question and return a source-backed RAG response."""
        payload = request.get_json(silent=True) or {}
        question = payload.get("question")

        if not isinstance(question, str) or not question.strip():
            return jsonify({"error": "Question is required."}), 400

        try:
            response = answer_question(question.strip())
        except ModelServiceError as exc:
            return jsonify({"error": str(exc)}), 503

        return jsonify(response), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
```

### `seed_chroma.py`

```python
"""Seed the local Chroma database for the Flask RAG API lesson."""

from chroma_store import seed_collection


def main() -> None:
    """Create a fresh Chroma collection and store the lesson documents."""
    count = seed_collection()
    print(f"Seeded {count} facility documents into Chroma.")


if __name__ == "__main__":
    main()
```

### `documents.py`

```python
"""Facility operations documents for the Flask RAG API lesson."""

from typing import Dict, List


FACILITY_DOCUMENTS: List[Dict[str, str]] = [
    {
        "id": "FAC-101",
        "title": "Replacing a Lost or Damaged Employee Badge",
        "category": "access",
        "text": (
            "Employees who lose or damage a badge should submit a badge replacement "
            "request through the facilities portal. Temporary badges are available at "
            "the front desk after identity verification."
        ),
    },
    {
        "id": "FAC-102",
        "title": "After-Hours Building Access",
        "category": "access",
        "text": (
            "Employees may enter the building after 7 PM only when after-hours access "
            "has been enabled on their badge. Requests should be submitted before the "
            "needed date and approved by the employee's manager."
        ),
    },
    {
        "id": "FAC-103",
        "title": "Conference Room Setup Requests",
        "category": "events",
        "text": (
            "Room setup requests for meetings, trainings, or client visits should be "
            "submitted at least two business days in advance. Include room name, head "
            "count, seating layout, audio needs, and any equipment requirements."
        ),
    },
    {
        "id": "FAC-104",
        "title": "Temperature and Maintenance Requests",
        "category": "maintenance",
        "text": (
            "Heating, cooling, lighting, plumbing, or furniture issues should be "
            "reported through a facilities maintenance ticket. Include the floor, room "
            "number, issue description, urgency, and a photo when helpful."
        ),
    },
    {
        "id": "FAC-105",
        "title": "Visitor Registration and Lobby Check-In",
        "category": "visitors",
        "text": (
            "Visitors must be registered before arrival. The host should add the guest "
            "name, company, visit date, and host contact information. Guests receive a "
            "temporary visitor badge at lobby check-in."
        ),
    },
    {
        "id": "FAC-106",
        "title": "Office Equipment Repair Requests",
        "category": "maintenance",
        "text": (
            "Broken shared equipment, including printers, monitors, projectors, and "
            "badge readers, should be reported with the asset name, location, error "
            "message, and a description of the problem."
        ),
    },
]
```
