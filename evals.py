# Evals for the document agent: known-answer tests scored automatically, including questions the documents cannot answer.  # what this file is
# An eval is only honest if it can fail, so the suite includes a question with no source, where the correct behaviour is refusal.  # why refusal is tested
from agent import run_agent  # the agent under test

CASES = [  # each case: a question and the fact a correct answer must contain; None means the documents cannot answer it
    ("How many vacation days do employees receive each year?", "20"),  # leave policy: the vacation fact
    ("How many sick days do employees get?", "10"),  # leave policy: the sick leave fact, asked with different wording
    ("How many vacation days can be carried over?", "5"),  # leave policy: the carry-over rule
    ("How many days do customers have to return a product?", "30"),  # refund policy: the return window
    ("How long do refunds take to reach the customer?", "7"),  # refund policy: the refund timing
    ("Can opened software be returned?", "cannot"),  # refund policy: the exception rule
    ("How often must passwords be changed?", "90"),  # password policy: the change cycle
    ("How long must a password be?", "12"),  # password policy: the length rule
    ("How many failed sign-ins lock an account?", "5"),  # password policy: the lockout rule
    ("What is the CEO's salary?", None),  # no source exists: the agent must refuse
    ("What is the office Wi-Fi password?", None),  # no source exists: the agent must refuse
]  # end of the eval cases

def score(case, result):  # grade one agent result against its case
    question, expected = case  # unpack the case
    if expected is None:  # the refusal cases: passing means the agent gave no answer
        return result["answer"] is None and result["status"] == "not_in_documents"  # refused cleanly, as required
    if result["answer"] is None:  # an answerable question that got no answer fails
        return False  # missing answer, missing fact
    return expected in result["answer"] and result["grounded"]  # pass only when the fact is present AND nothing was invented

if __name__ == "__main__":  # run the suite when this file is run directly
    print("=== EVALS: DOCUMENT AGENT ===")  # the suite heading
    passed = 0  # the count of passing cases
    for case in CASES:  # run every case through the agent
        result = run_agent(case[0])  # one agent run per case
        ok = score(case, result)  # grade the run
        passed += 1 if ok else 0  # count the pass
        print(f"[{'PASS' if ok else 'FAIL'}] {case[0]}")  # print the case verdict
        print(f"       answer: {result['answer'] if result['answer'] else '(refused - not in the documents)'}")  # print what the agent said
    total = len(CASES)  # the number of cases in the suite
    print(f"EVAL RESULT: {passed} of {total} passed ({round(100 * passed / total)}%)")  # the summary line a reviewer reads first
