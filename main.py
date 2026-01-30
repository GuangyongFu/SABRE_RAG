from flask import Flask, render_template, request, jsonify
from src.sabre_rag import query_rag

app = Flask(__name__)
@app.route("/")
def home():
    """Render the home page with the question form."""
    return render_template("index.html")
@app.route("/ask", methods=["POST"])
def ask_question():
    """Handle the question submitted from the form."""
    data = request.json  # Parse JSON payload
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "Please provide a valid question."})
    # Process the question (placeholder logic)
    try:
    	response=''
    	response = query_rag(question)
    	answer=response.response
    	answer=answer+'\n ---------------\n'
    	for node in response.source_nodes:
            if node.score > 0.75:
            	filename = node.metadata.get('file_name', 'Unknown')
            	page_label = node.metadata.get('page_label', 'Unknown')
            	answer=answer+'\n Data Source:\n'+f"Page: {page_label}, File: {filename}\n"
    except Exception as e:
    	answer = f"An error occurred:{str(e)}"
    return jsonify({"answer": answer})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
