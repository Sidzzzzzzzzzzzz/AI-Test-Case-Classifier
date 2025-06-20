from pyvis.network import Network

def build_graph(df, topic_column="topic_id"):
    net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='black')

    # Add topic nodes (orange boxes)
    topics = df[topic_column].unique()
    for topic in topics:
        net.add_node(f"Topic {topic}", label=f"Topic {topic}", shape='box', color='orange')

    # Add test case nodes and connect them to their topic
    for idx, row in df.iterrows():
        test_case = row['test_case']
        topic = f"Topic {row[topic_column]}"
        truncated_label = (test_case[:50] + "...") if len(test_case) > 50 else test_case

        net.add_node(test_case, label=truncated_label, shape='ellipse', color='lightblue')
        net.add_edge(topic, test_case)

    # Save graph
    net.show('test_case_graph.html')


