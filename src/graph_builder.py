from pyvis.network import Network
import streamlit.components.v1 as components

def build_graph(df, topic_column="topic_id"):
    net = Network(height="600px", width="100%", notebook=False, bgcolor="#ffffff", font_color="black")

    topics = df[topic_column].unique()
    for t in topics:
        net.add_node(f"Topic {t}", label=f"Topic {t}", color="orange", shape="dot", size=20)

    for _, row in df.iterrows():
        net.add_node(row["test_case"], label=row["test_case"], color="lightblue", shape="box")
        net.add_edge(f"Topic {row[topic_column]}", row["test_case"])

    # Render inside Streamlit instead of saving HTML
    html = net.generate_html()
    components.html(html, height=600, scrolling=True)




