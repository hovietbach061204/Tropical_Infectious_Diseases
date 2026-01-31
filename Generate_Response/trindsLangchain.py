import asyncio
import os
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from RAG.QdrantRetriever import get_retriever
from RAG.utils.queryVectorDB import process_item, get_cases_knowledge_graph, graph
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(model=os.getenv("CHAT_MODEL"), reasoning=None)

"""
ChatOllama utilizes the model's internal chat template (e.g., <|im_start|>system...).
This creates a harder boundary between your instructions (System) and the noisy data (User/Context), 
making the model much more obedient.
"""

template = """
You are an expert in tropical and infectious diseases. Provide an evidence-based diagnosis.

### INSTRUCTIONS
Analyze the context provided below to answer the question.
Output format - STRICTLY FOLLOW THIS:
Line 1: "Diagnosis: [specific disease name]"
Lines 2-4: "Explanation: [2-4 concise sentences]"

Rules:
- Be SPECIFIC: Use disease names like "Dengue fever", not "viral infection"
- PRIORITIZE patient symptoms over any other information
- Stop IMMEDIATELY after the explanation
- Do NOT output references, metadata, or disease details found in the context headers
- Do NOT use <think> tags

Example:
Diagnosis: Typhoid fever
Explanation: Rising fever with alternating constipation/diarrhea indicates typhoid. Medical testing is needed for confirmation.

### CONTEXT
<context>
{rag_documents}
</context>

### QUESTION
{question}

### FINAL REMINDER
Output ONLY the "Diagnosis:" and "Explanation:" lines as requested above.
"""

system_template = """
You are an expert in tropical and infectious diseases. Provide an evidence-based diagnosis.
### INSTRUCTIONS
Analyze the context provided below to answer the question.
Output format - STRICTLY FOLLOW THIS:
Line 1: "Diagnosis: [specific disease name]"
Lines 2-4: "Explanation: [2-4 concise sentences]"
Rules:
- Be SPECIFIC: Use disease names like "Dengue fever", not "viral infection"
- PRIORITIZE patient symptoms over any other information
- Stop IMMEDIATELY after the explanation
- Do NOT output references, metadata, or disease details found in the context headers
Example:
Diagnosis: Typhoid fever
Explanation: Rising fever with alternating constipation/diarrhea indicates typhoid. Medical testing is needed for confirmation.
### FINAL REMINDER
Output ONLY the "Diagnosis:" and "Explanation:" lines as requested above.
"""

human_template = """
### CONTEXT
<context>
{rag_documents}
</context>

### QUESTION
{question}
"""

concise_system_template = """
You are an expert in tropical and infectious diseases. Provide an evidence-based diagnosis.
Output format:
Diagnosis: [disease name]
Explanation: [2-3 sentences]

Rules:
- Use specific disease names
- Prioritize patient symptoms
- Stop after explanation
### FINAL REMINDER
Output ONLY the "Diagnosis:" and "Explanation:" lines as requested above.
"""

concise_human_template = """
Context:
{rag_documents}

Question: {question}
"""

# prompt = ChatPromptTemplate.from_template(template)
prompt = ChatPromptTemplate.from_messages([
    ("system", concise_system_template),
    ("human", concise_human_template),
])

chain = prompt | model


def generate_response(question: str) -> str:
    contexts_list = with_naive_kg(question)
    all_contexts = "\n------\n".join(contexts_list)
    result = chain.invoke({"rag_documents": all_contexts, "question": question})
    print(f"Retrieved {len(contexts_list)} contexts: {all_contexts}\n")
    print(f"Response: {result}")
    print(f"Result's content:\n{result.content}")
    return result.content

def generate_response_with_context(question: str, contexts: list[Any]) -> str:
    all_contexts = "\n------\n".join(contexts)
    result = chain.invoke({"rag_documents": all_contexts, "question": question})
    print(f"Retrieved {len(contexts)} contexts: {all_contexts}\n")
    print(f"Response: {result}")
    print(f"Result's content:\n{result.content}")
    return result.content


def demo():
    while True:
        print("\n\n-------------------------------")
        question = input(">>> ")
        print("\n\n")
        if question == "/bye":
            break

        retriever = get_retriever()
        hits = asyncio.run(retriever.query(question, k=3))

        # ___________________WITH NAIVE RAG + KNOWLEDGE GRAPH: (QDRANT + NEO4J)______________________
        item = asyncio.run(process_item(hits))
        ids = item.get("found_cases", [])
        contexts_list = item.get("context", [])
        print(f"Found cases: {ids}")
        print(f"Contexts length: {len(contexts_list)}")

        knowledge_graph_list = get_cases_knowledge_graph(graph, ids)
        print(f"Knowledge graph length: {len(knowledge_graph_list)}")
        print(f"Knowledge Graph List: {knowledge_graph_list}")
        final_list = []
        for index, context in enumerate(contexts_list):
            knowledge = knowledge_graph_list[index]
            if knowledge.get("has_info"):
                info_values = [
                    val for key, val in knowledge.items()
                    if val is not None and key not in ["case_id", "has_info"]
                ]
                main_info_block = "\n".join(info_values)
                final_context = f"""
MAIN INFORMATION:
{main_info_block}
---
ADDITIONAL CONTEXTS:
{context}
"""
            else:
                final_context = f"""
CONTEXTS FROM RAG:
{context}
"""
            final_list.append(final_context)

        all_contexts = "\n------\n".join(final_list)
        result = chain.invoke({"rag_documents": all_contexts, "question": question})
        print(f"Retrieved {len(contexts_list)} contexts: {all_contexts}\n")
        print(f"Response: {result}")
        print(f"Result's content:\n{result.content}")

        # ___________________WITH NAIVE RAG ONLY: (QDRANT)______________________
        # contexts_list = [hit.get("context", "") for hit in (hits or [])]
        # all_contexts = "\n---\n".join(contexts_list)
        # result = chain.invoke({"rag_documents": all_contexts, "question": question})
        # print(f"Retrieved {len(contexts_list)} contexts: {all_contexts}\n")
        # print(f"Response: {result}")
        # print(f"Result's content:\n{result.content}")

def with_naive_kg(question: str) -> list[Any] | None:
    retriever = get_retriever()
    hits = asyncio.run(retriever.query(question, k=3))
    # ___________________WITH NAIVE RAG + KNOWLEDGE GRAPH: (QDRANT + NEO4J)______________________
    item = asyncio.run(process_item(hits))
    ids = item.get("found_cases", [])
    contexts_list = item.get("context", [])

    knowledge_graph_list = get_cases_knowledge_graph(graph, ids)
    final_list = []
    for index, context in enumerate(contexts_list):
        knowledge = knowledge_graph_list[index]
        if knowledge.get("has_info"):
            info_values = [
                val for key, val in knowledge.items()
                if val is not None and key not in ["case_id", "has_info"]
            ]
            main_info_block = "\n".join(info_values)

            final_context = f"""
MAIN INFORMATION:
{main_info_block}
---
ADDITIONAL CONTEXTS:
{context}
"""
        else:
            final_context = f"""
CONTEXTS FROM RAG:
{context}
"""
        final_list.append(final_context)
        return final_list
    return None


def with_naive_only(question: str) -> list[Any]:
    retriever = get_retriever()
    hits = asyncio.run(retriever.query(question, k=3))
    # ___________________WITH NAIVE RAG ONLY: (QDRANT)______________________
    contexts_list = [hit.get("context", "") for hit in (hits or [])]
    return contexts_list


def main():
    user_question = "My 10 year old nephew visited the Thanda Safari with us last week. Now he is sick with a fever, skin rash and itching foot. what is happening to him?"
    print(generate_response(user_question))
    # demo()


if __name__ == "__main__":
    main()