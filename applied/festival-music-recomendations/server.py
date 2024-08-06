from flask import Flask
from crawler import get_artist_scores

app = Flask(__name__)

artists = get_artist_scores()

@app.route("/")
def get_artists():
    # I'm to old to use templates
    html = [
        "<html>",
        '<div style="width: fit-content;">',
    ]
    for i in artists[:10]:
        html.append(f"""
        <div style="border-style: solid; margin: 5px"> 
            <h1>{i.name}</h1>      
            <h1>Score: {i.score}</h1>      
        </div>
        """)

    html.append("</div>")
    html.append("</html>")
    print("I return the data?")
    return "\n".join(html)

app.run(port=5001)
