import base64
from io import BytesIO
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
"""
TODO: I think it should just be shared with the plot figure we have and 
that it outputs to bytes
"""

def plot_xy(y):
    fig = Figure()
    ax = fig.subplots()
    ax.plot(y)
    buf = BytesIO()
    fig.savefig(buf, format="png")

    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.cla()
    return f"<img src='data:image/png;base64,{data}'/>"
