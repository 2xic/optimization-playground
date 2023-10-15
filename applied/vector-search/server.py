from flask import Flask
from database import Chroma

database = Chroma()
app = Flask(__name__)

@app.route('/')
def hello():
    return "<br>".join(
        link_2_arxiv(i)
        for i in database.get_all()
    )

@app.route('/sim/<id>')
def sim(id):
    header = f"<h1>Similiar to {id}</h1>"
    similar_items = "<br>".join(
        link_2_arxiv(i)
        for i in database.get_sim_from_id(
            id
        )
    )
    return "<br>".join([
        header,
        f"<a href=\"https://arxiv.org/abs/{id}\">Arxiv</a>",
        similar_items
    ])


def link_2_arxiv(id):
    a = [
        f"<a href=\"/sim/{id}\">{id}</a>",
        "(similarity)"
        f"<a href=\"https://arxiv.org/abs/{id}\">Arxiv</a>",
    ]
    return "".join(a)

if __name__ == "__main__":
    app.run(port=3242)
