# A small REST API over the document agent: ask a question over HTTP, get a JSON answer with its sources and grounding status.  # what this file is
# Standard library only, like the rest of the project, so it runs anywhere Python runs: python3 api.py, then open the /ask route.  # how to run it
import json  # a toolbox for JSON, the language APIs speak
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # the standard library web server tools
from urllib.parse import urlparse, parse_qs  # toolboxes for reading the route and the query string
from agent import run_agent  # the agent that answers the questions

class Handler(BaseHTTPRequestHandler):  # the request handler: one class, one job per route
    def send_json(self, payload, status=200):  # a helper that replies with JSON and a status code
        body = json.dumps(payload).encode()  # turn the answer data into JSON bytes
        self.send_response(status)  # send the HTTP status line
        self.send_header("Content-Type", "application/json")  # declare the reply is JSON
        self.send_header("Content-Length", str(len(body)))  # declare the reply length
        self.end_headers()  # finish the headers
        self.wfile.write(body)  # send the body

    def do_GET(self):  # handle every GET request by its route
        route = urlparse(self.path).path  # the route part of the address, without the query
        query = parse_qs(urlparse(self.path).query)  # the query values, such as the question text
        if route == "/health":  # the health route: is the service up
            self.send_json({"status": "ok", "service": "azure-openai-doc-search"})  # reply that the service is alive
        elif route == "/ask":  # the ask route: one question in, one grounded answer out
            question = query.get("q", [""])[0]  # read the question from the query string
            if not question:  # a missing question is a client error, said plainly
                self.send_json({"error": "missing query parameter: q"}, status=400)  # reply with the error and 400
                return  # stop here, nothing more to do
            result = run_agent(question)  # run the full agent: search, answer, verify
            self.send_json({  # reply with the answer and the evidence behind it
                "question": result["question"],  # the question as received
                "answer": result["answer"],  # the answer, or null when the documents cannot answer
                "status": result["status"],  # ok, unverified, or not_in_documents
                "grounded": result["grounded"],  # whether every fact traces to a source
                "sources": result["sources"],  # the documents the answer rests on
            })  # end of the ask reply
        else:  # any other route does not exist here
            self.send_json({"error": "unknown route", "routes": ["/health", "/ask?q=..."]}, status=404)  # reply 404 with the real routes

    def log_message(self, format, *args):  # silence the default request logging so test output stays readable
        pass  # deliberately nothing

if __name__ == "__main__":  # run the server when this file is run directly
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)  # bind the server to this machine on port 8765
    print("API listening on http://127.0.0.1:8765  (routes: /health, /ask?q=...)")  # say where it lives
    server.serve_forever()  # serve until stopped
