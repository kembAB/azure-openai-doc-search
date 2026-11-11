# Agentic layer over the document search: the agent plans, uses tools, checks its own answer, and retries or refuses honestly.  # what this file is
# It reuses the core program (azure_openai_doc_search.py) as its tool set, so the agent and the plain pipeline can never drift apart.  # one source of truth
import contextlib  # a toolbox for redirecting output while a module loads
import io  # a toolbox for holding text in memory instead of printing it

with contextlib.redirect_stdout(io.StringIO()):  # load the core quietly, without running its printed demo
    import azure_openai_doc_search as core  # the core program: chunking, embeddings, search, answering, checks

def tool_search(question, top_k=2):  # TOOL 1: find the chunks that answer a question
    return core.search(question, top_k=top_k)  # hand the question to the core semantic search

def tool_answer(question, chunks):  # TOOL 2: build an answer from found chunks only
    return core.generate_answer(question, chunks)  # hand the question and chunks to the core answer builder

def tool_check(answer, chunks):  # TOOL 3: verify the answer invented nothing
    return core.is_grounded(answer, chunks)  # hand the answer and chunks to the core hallucination check

TOOLS = {"search": tool_search, "answer": tool_answer, "check": tool_check}  # the agent's tool belt, by name

def run_agent(question, wider_k=5):  # the agent loop: search, answer, verify, then retry wider or refuse honestly
    steps = []  # a written trace of every step the agent takes, so its reasoning can be inspected
    steps.append(f"PLAN: answer from the documents only, verify before replying. Question: {question}")  # step: state the plan
    chunks = TOOLS["search"](question, top_k=2)  # step 1: search narrow first, the two best chunks
    steps.append(f"SEARCH (top 2): found {len(chunks)} chunk(s)")  # record what the search returned
    if not chunks:  # nothing in the documents matches the question at all
        steps.append("DECISION: no source material, so no answer may be given")  # record the honest refusal
        return {"question": question, "answer": None, "status": "not_in_documents", "grounded": False, "sources": [], "steps": steps}  # refuse rather than invent
    question_words = set(core.words_of(question))  # the question's content words, for the coverage gate
    chunk_words = set()  # every content word the found chunks contain
    for chunk in chunks:  # gather the found chunks' words
        chunk_words |= set(core.words_of(chunk["text"]))  # add this chunk's words to the set
    coverage = len(question_words & chunk_words) / len(question_words) if question_words else 0  # the share of the question the sources actually cover
    steps.append(f"COVERAGE: {round(coverage * 100)}% of the question's content words appear in the found chunks")  # record the coverage measurement
    if coverage < 0.4:  # the sources touch the question's words too weakly to support an answer
        steps.append("DECISION: coverage under 40 percent, so no answer may be given")  # record the honest refusal
        return {"question": question, "answer": None, "status": "not_in_documents", "grounded": False, "sources": [], "steps": steps}  # refuse rather than stretch weak matches
    answer = TOOLS["answer"](question, chunks)  # step 2: build the answer from the found chunks
    steps.append(f"ANSWER DRAFT: {answer}")  # record the draft answer
    grounded = TOOLS["check"](answer, chunks)  # step 3: verify the draft against its sources
    steps.append(f"CHECK: grounded in sources = {grounded}")  # record the verification result
    if not grounded:  # the narrow search may have missed the chunk that backs a fact, so widen once
        steps.append("RETRY: widening the search and answering again")  # record the retry decision
        chunks = TOOLS["search"](question, top_k=wider_k)  # search again with more chunks
        answer = TOOLS["answer"](question, chunks)  # rebuild the answer from the wider set
        grounded = TOOLS["check"](answer, chunks)  # verify the new answer
        steps.append(f"CHECK AFTER RETRY: grounded in sources = {grounded}")  # record the second verification
    if not answer:  # the chunks held no sentence that answers the question
        steps.append("DECISION: sources do not answer this question, so no answer may be given")  # record the honest refusal
        return {"question": question, "answer": None, "status": "not_in_documents", "grounded": False, "sources": [], "steps": steps}  # refuse rather than stretch the sources
    status = "ok" if grounded else "unverified"  # an ungrounded answer is flagged, never presented as fact
    sources = sorted({chunk["source"] for chunk in chunks})  # the document names the answer rests on
    steps.append(f"DECISION: reply with status {status}, sources {sources}")  # record the final decision
    return {"question": question, "answer": answer, "status": status, "grounded": grounded, "sources": sources, "steps": steps}  # the full result, trace included

if __name__ == "__main__":  # run the demo only when this file is run directly
    DEMO_QUESTIONS = [  # three questions: two the documents answer, one they do not
        "How many vacation days do employees receive each year?",  # answerable: the leave policy
        "How many days do customers have to return a product?",  # answerable: the refund policy
        "What is the CEO's salary?",  # not in any document: the agent must refuse, not invent
    ]  # end of the demo questions
    for q in DEMO_QUESTIONS:  # run the agent on each question in turn
        result = run_agent(q)  # one full agent run
        print("=" * 70)  # a separator line between runs
        for step in result["steps"]:  # print the agent's trace, step by step
            print(" ", step)  # one trace step per line
        print(" FINAL ANSWER:", result["answer"] if result["answer"] else "(none - not in the documents)")  # print the outcome plainly
        print(" STATUS:", result["status"], "| GROUNDED:", result["grounded"], "| SOURCES:", result["sources"])  # print the verdict line
