# Planning Notes

Use this file for the short planning and verification notes in the lesson.

## Step 1: RAG API goal and output contract

User or role:


Business problem:


User question:


Searchable content:


Endpoint route:


Expected successful response:


Fallback behavior:


Verification goal:


## Step 12: Verification note

Functional output:


Input validation:


Retrieval quality:


Prompt grounding:


Source traceability:


Fallback behavior:


## Step 13: Reflection

What does Chroma own in this workflow?


What does the prompt builder own?


What does the Flask route coordinate?


Why does the API response include sources?


What would break if the route sent the user question directly to the model?


--------------------------------------------------------


In planning_notes.md, answer these questions:

What does Chroma own in this workflow?
What does the prompt builder own?
What does the Flask route coordinate?
Why does the API response include sources?
What would break if the route sent the user's question directly to the model?
A completed reflection might look like this:

Chroma owns vector storage and retrieval. It stores embedded facilities documents and returns the most relevant source text for a user's question.
The prompt builder owns the instructions, labeled context, user question, response requirements, and fallback rule. It helps the model answer from an approved source text instead of guessing.
The Flask route coordinates the HTTP request and response. It validates the JSON body, calls the RAG service, handles model-service errors, and returns structured JSON.
The API response includes sources so users and developers can inspect where the answer came from. If the route sent the user's question directly to the model, the answer might sound confident but would not be grounded in the approved facilities context.