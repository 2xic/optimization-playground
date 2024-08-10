from flask import Flask
from crawler import get_artist_scores
from shared import spotify_playlist_cleaner_endpoint
from dateutil import parser

app = Flask(__name__)

days_artists = get_artist_scores()

play_song_script =         f"""
        <script>
            function play_song(songId) {{
                fetch("{spotify_playlist_cleaner_endpoint}/play?song_id=" + songId, {{
                    method: "POST",
                }});
            }}
        </script>
        """

@app.route("/date/<date>")
def get_lineup_date(date):
    # I'm to old to use templates
    html = [
        "<html>",
        play_song_script,
        '<div style="width: fit-content;">',
    ]
    html.append(f"""
        <h1>{date}</h1>
    """)
    timestamp_day = date #int(parser.parse(date).timestamp())
    print(timestamp_day)
    print(days_artists.keys())
    for i in days_artists[timestamp_day][:10]:
        html.append(f"""
        <div style="border-style: solid; margin: 5px"> 
            <h1>{i.name}</h1>      
            <h2>Score: {i.score}</h2>      
            <h2>When: {i.hour}</h2>      
            <button onclick="play_song('{i.best_track.id}')">{i.best_track.name}</button>
        </div>
        """)

    html.append("</div>")
    html.append("</html>")
    return "\n".join(html)

@app.route("/global")
def get_all_artists_scores():
    # I'm to old to use templates
    html = [
        "<html>",
        play_song_script,
        '<div style="width: fit-content;">',
    ]
    for i in sorted(sum(days_artists.values(), []), key=lambda x: x.score):
        html.append(f"""
            <div style="border-style: solid; margin: 5px"> 
                <h1>{i.name}</h1>      
                <h2>Score: {i.score}</h2>      
                <h2>When: {i.datetime} {i.hour}</h2>      
                <button onclick="play_song('{i.best_track.id}')">{i.best_track.name}</button>
            </div>
            """)
    html.append("</div>")
    html.append("</html>")
    return "\n".join(html)

@app.route("/")
def get_artists():
    # I'm to old to use templates
    html = [
        "<html>",
        '<div style="width: fit-content;">',
        "<ul>"
    ]
    for artists_date in days_artists:
        html.append(f"""
            <li>
                    <a href="/date/{artists_date}">{artists_date}</a>
            </li>
        """)
    html.append("</ul>")
    html.append("</div>")
    html.append("</html>")
    return "\n".join(html)

app.run(port=5025)
