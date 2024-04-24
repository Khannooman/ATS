from flask import Flask, request, jsonify
from resume_parsing import parse_resume
app = Flask(__name__)


@app.route("/resume_parser", methods = ["GET", "POST"])
def resume_parser():
    if request.method == "post":
        data = request.json
        resume = data["resume"]
        parse_data = parse_resume(resume)
        return jsonify(parse_data)
    return jsonify({"msg":"please upload your resume"})


if __name__ == "__main__":
    app.run(debug = True)

    


