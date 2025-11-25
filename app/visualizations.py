import plotly.express as px


def scatter_2d(df):
    return px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_data=["id", "text"],
        width=900,
        height=600
    )


def scatter_3d(df):
    return px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="cluster",
        hover_data=["id", "text"],
        width=900,
        height=700
    )
