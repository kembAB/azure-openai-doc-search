# Azure OpenAI in production: document processing and semantic search, with validation.  # the whole job in one program
# In production these same steps call Azure OpenAI. Here they run locally with the same shapes, so the code runs with no key.  # honest demo setup
import re  # a toolbox for finding patterns in text, like words and numbers
import math  # a toolbox for math formulas, used here to compare vectors

# ---------- PART 1: LOAD AND CHUNK THE DOCUMENTS ----------
DOCUMENTS = {  # the source documents, stored as a dictionary of name -> list of sentences
    "Leave Policy": [  # first document: the company leave policy
        "Full-time employees receive 20 vacation days each year.",  # sentence: the vacation fact
        "Employees may carry over up to 5 unused vacation days into the next year.",  # sentence: the carry-over rule
        "Sick leave is separate and provides 10 days per year.",  # sentence: the sick leave fact
    ],  # end of the leave policy sentences
    "Refund Policy": [  # second document: the customer refund policy
        "Customers may return any product within 30 days of purchase for a full refund.",  # sentence: the return window fact
        "Refunds are issued to the original payment method within 7 business days.",  # sentence: how refunds are paid
        "Opened software licenses cannot be returned.",  # sentence: the exception rule
    ],  # end of the refund policy sentences
    "Password Policy": [  # third document: the security password policy
        "Passwords must be changed every 90 days.",  # sentence: the password change fact
        "Each password must contain at least 12 characters with one number and one symbol.",  # sentence: the password rule
        "Accounts lock after 5 failed sign-in attempts.",  # sentence: the lockout rule
    ],  # end of the password policy sentences
}  # end of all documents

def chunk_documents(documents, size=2):  # a function that cuts documents into small pieces called chunks
    chunks = []  # an empty list that will hold every chunk
    for name, sentences in documents.items():  # loop over each document, one at a time
        for start in range(0, len(sentences), size):  # step through the sentences in groups of the chunk size
            piece = " ".join(sentences[start:start + size])  # join one group of sentences into one chunk of text
            chunks.append({"source": name, "text": piece})  # save the chunk with the name of its document
    return chunks  # hand back the full list of chunks

CHUNKS = chunk_documents(DOCUMENTS)  # run the chunking once and keep the result for the whole program

# ---------- PART 2: EMBEDDINGS - TURN TEXT INTO NUMBERS ----------
def words_of(text):  # a function that breaks text into single words
    return re.findall(r"[a-z0-9]+", text.lower())  # find every word or number, all in small letters

VOCABULARY = sorted({word for chunk in CHUNKS for word in words_of(chunk["text"])})  # every different word in all chunks, sorted

def embed(text):  # a function that turns text into a vector, meaning a list of numbers
    # In production this line calls Azure OpenAI: client.embeddings.create(model="text-embedding-3-small", input=text)  # the real Azure call
    counts = [0] * len(VOCABULARY)  # start with one zero for every word in the vocabulary
    for word in words_of(text):  # loop over each word in the text
        if word in VOCABULARY:  # only count words the program knows
            counts[VOCABULARY.index(word)] += 1  # add one to that word's position in the vector
    return counts  # hand back the finished vector

def cosine(first, second):  # a function that measures how similar two vectors are, from 0 to 1
    dot = sum(a * b for a, b in zip(first, second))  # multiply matching positions and add them up
    size_first = math.sqrt(sum(a * a for a in first))  # the length of the first vector
    size_second = math.sqrt(sum(b * b for b in second))  # the length of the second vector
    if size_first == 0 or size_second == 0:  # guard against dividing by zero
        return 0.0  # no overlap possible, so the similarity is zero
    return dot / (size_first * size_second)  # divide to get the final similarity score

for chunk in CHUNKS:  # loop over every chunk once, at the start
    chunk["vector"] = embed(chunk["text"])  # store each chunk's vector inside the chunk itself

# ---------- PART 3: SEMANTIC SEARCH - FIND THE RIGHT CHUNKS ----------
def search(question, top_k=2):  # a function that finds the chunks closest in meaning to a question
    question_vector = embed(question)  # turn the question into a vector the same way
    scored = []  # an empty list for chunks with their scores
    for chunk in CHUNKS:  # loop over every chunk
        score = cosine(question_vector, chunk["vector"])  # score this chunk against the question
        scored.append((score, chunk))  # keep the score together with the chunk
    scored.sort(key=lambda pair: pair[0], reverse=True)  # sort so the best score comes first
    return [chunk for score, chunk in scored[:top_k] if score > 0]  # return only the top chunks that scored above zero

# ---------- PART 4: GENERATE THE ANSWER FROM THE FOUND CHUNKS ONLY ----------
def generate_answer(question, chunks):  # a function that builds an answer using only the found chunks
    # In production this sends the chunks plus the question to an Azure OpenAI chat model, told to answer from the context only  # the real setup
    question_words = set(words_of(question))  # the question's words as a set, so repeats vanish
    best_sentences = []  # an empty list for the sentences that will form the answer
    for chunk in chunks:  # loop over the found chunks
        for sentence in chunk["text"].split(". "):  # look at each sentence inside the chunk
            overlap = len(question_words & set(words_of(sentence)))  # count how many question words the sentence shares
            best_sentences.append((overlap, sentence.strip()))  # keep the sentence with its overlap score
    best_sentences.sort(key=lambda pair: pair[0], reverse=True)  # sort so the most relevant sentence comes first
    chosen = [sentence for overlap, sentence in best_sentences[:2] if overlap > 0]  # take the best two sentences that truly overlap
    answer = ". ".join(chosen)  # join the chosen sentences into one answer text
    if answer and not answer.endswith("."):  # check the answer ends cleanly
        answer += "."  # add the closing period if it is missing
    return answer  # hand back the answer, built only from source sentences

# ---------- PART 5: VALIDATION - TEST QUESTIONS, ACCURACY AND HALLUCINATION CHECKS ----------
def numbers_in(text):  # a function that pulls every number out of a text
    return re.findall(r"\d+", text)  # return the numbers as a list of strings

def is_grounded(answer, chunks):  # a function that checks the answer invented nothing
    context = " ".join(chunk["text"] for chunk in chunks)  # put all found chunks into one context text
    for number in numbers_in(answer):  # loop over every number the answer uses
        if number not in numbers_in(context):  # a number in the answer but not in the sources is a hallucination
            return False  # fail the check and say so
    return True  # every number came from the sources, so the answer is grounded

TESTS = [  # the predefined test questions, each with the fact the answer must contain
    ("How many vacation days do employees receive each year?", "20"),  # test one: the vacation fact
    ("How many days do customers have to return a product?", "30"),  # test two: the return window fact
    ("How often must passwords be changed?", "90"),  # test three: the password fact
]  # end of the test list

print("=== DEMO: ASK THE DOCUMENTS A QUESTION ===")  # print the demo heading
demo_question = "How many vacation days do employees receive each year?"  # the demo question
found = search(demo_question)  # search the chunks for the demo question
for chunk in found:  # loop over the chunks the search found
    print("Found in", chunk["source"], "->", chunk["text"])  # print where the answer will come from
print("Answer:", generate_answer(demo_question, found))  # print the generated answer
print()  # print one blank line for spacing
print("=== VALIDATION: PREDEFINED TEST QUERIES CHECKED AGAINST THE SOURCES ===")  # print the validation heading
passed = 0  # a counter for the tests that pass, starting at zero
for question, expected in TESTS:  # loop over the test questions one at a time
    chunks = search(question)  # find the chunks for this test question
    answer = generate_answer(question, chunks)  # generate the answer for this test question
    accurate = expected in answer  # accuracy check: the expected fact must appear in the answer
    grounded = is_grounded(answer, chunks)  # hallucination check: no invented numbers allowed
    ok = accurate and grounded  # the test passes only when both checks pass
    passed += 1 if ok else 0  # add one to the counter when the test passed
    print("Q:", question)  # print the test question
    print("A:", answer)  # print the generated answer
    print("Expected fact present:", accurate, "| Grounded in sources:", grounded, "| TEST", "PASS" if ok else "FAIL")  # print the verdict
print("Validation:", passed, "of", len(TESTS), "tests passed - answers checked against source documents, no invented facts.")  # print the summary
