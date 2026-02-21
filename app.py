import streamlit as st
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Skill Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# ENHANCED UI STYLING
# --------------------------------------------------
st.markdown("""
<style>

/* Main Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
}

/* Titles */
h1 {
    font-size: 42px;
    font-weight: 700;
}

h2, h3 {
    color: #e2e8f0;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-weight: 600;
    font-size: 16px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
}

/* Skill Tags */
.skill-tag {
    background: linear-gradient(90deg, #2563eb, #60a5fa);
    padding: 8px 16px;
    border-radius: 25px;
    display: inline-block;
    margin: 6px;
    color: white;
    font-size: 14px;
    font-weight: 500;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
}

/* Metric Cards */
.metric-card {
    background: #1e293b;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}

/* Text Area */
textarea {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background-color: #0f172a !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER SECTION
# --------------------------------------------------
st.markdown("""
<div style="text-align:center; padding: 20px 0;">
    <h1>🧠 AI Skill Intelligence System</h1>
    <p style="font-size:18px; color:#94a3b8;">
        Hybrid Skill Extraction using BiLSTM + Rule-Based Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
@st.cache_data
def load_dataset():
    return pd.read_csv("all_job_post.csv")

df = load_dataset()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("bilstm_skill_model.h5")

    with open("tokenizer_lstm.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("mlb.pkl", "rb") as f:
        mlb = pickle.load(f)

    return model, tokenizer, mlb

model, tokenizer_lstm, mlb = load_artifacts()

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s\+\#\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------------------------------
# MASTER SKILLS (UNCHANGED)
# --------------------------------------------------
MASTER_SKILLS = {
    "python","java","c","c++","c#","javascript","typescript","go","rust","scala","kotlin",
    "php","ruby","swift","r","matlab","perl","bash","powershell",
    "html","css","react","angular","vue","next.js","node.js","express","django","flask",
    "spring","spring boot","laravel","asp.net","jquery","bootstrap","tailwind",
    "sql","mysql","postgresql","oracle","mongodb","redis","cassandra","dynamodb",
    "firebase","sqlite","neo4j","mariadb","elasticsearch",
    "aws","azure","gcp","google cloud","amazon web services","microsoft azure",
    "ec2","s3","lambda","cloudformation","iam","rds","eks",
    "azure data factory","adf","azure devops","azure functions","blob storage",
    "bigquery","cloud run","cloud functions",
    "docker","kubernetes","terraform","ansible","jenkins","gitlab","github actions",
    "ci/cd","devops","helm","prometheus","grafana","nagios",
    "etl","elt","data pipeline","data pipelines","airflow","apache airflow",
    "spark","pyspark","hadoop","hive","pig","kafka","databricks","snowflake",
    "redshift","synapse","delta lake",
    "machine learning","deep learning","nlp","computer vision","pytorch","tensorflow",
    "keras","scikit-learn","xgboost","opencv","huggingface","bert","llm",
    "langchain","rag","genai","generative ai",
    "cyber security","penetration testing","ethical hacking","network security",
    "cryptography","siem","splunk","firewall","ids","ips","vulnerability assessment",
    "android","ios","flutter","react native","swiftui","kotlin android",
    "selenium","cypress","junit","pytest","automation testing","manual testing",
    "testng","postman","api testing",
    "git","github","bitbucket","jira","confluence","linux","unix","windows",
    "visual studio","vscode","intellij","eclipse",
    "tcp/ip","dns","http","https","vpn","routing","switching",
    "power bi","tableau","excel","data analysis","data visualization","looker",
    "microservices","rest api","graphql","soap","system design","distributed systems",
    "oop","data structures","algorithms"
}

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
def predict_skills(job_text):
    FIXED_THRESHOLD = 0.30
    job_text = clean_text(job_text)
    seq = tokenizer_lstm.texts_to_sequences([job_text])
    X_input = pad_sequences(seq, maxlen=150)
    pred = model.predict(X_input, verbose=0)[0]

    return [
        mlb.classes_[i]
        for i, val in enumerate(pred)
        if val > FIXED_THRESHOLD
    ]

# --------------------------------------------------
# RULE BASED
# --------------------------------------------------
def rule_based_cs_skills(job_text):
    job_text = clean_text(job_text)
    return sorted({skill for skill in MASTER_SKILLS if skill in job_text})

# --------------------------------------------------
# KNOWLEDGE GRAPH
# --------------------------------------------------
def visualize_graph(job_role, skills):

    G = nx.Graph()
    G.add_node(job_role)

    for skill in skills:
        G.add_node(skill)
        G.add_edge(job_role, skill)

    pos = nx.spring_layout(G, k=0.9)

    plt.figure(figsize=(16, 9))
    plt.gca().set_facecolor("#0f172a")

    nx.draw_networkx_nodes(G, pos, nodelist=[job_role],
                           node_size=8000, node_color="#ef4444")

    nx.draw_networkx_nodes(G, pos, nodelist=skills,
                           node_size=3500, node_color="#3b82f6")

    nx.draw_networkx_edges(G, pos, edge_color="#94a3b8", width=2)

    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white")

    plt.title("Skill Relationship Graph", fontsize=20, color="white")
    plt.axis("off")
    st.pyplot(plt)

# --------------------------------------------------
# SIDEBAR (UNCHANGED LOGIC)
# --------------------------------------------------
st.sidebar.header("⚙️ Configuration")

category_list = sorted(df["category"].unique())
if "Software Engineer" not in category_list:
    category_list.append("Software Engineer")

selected_category = st.sidebar.selectbox("Category", sorted(category_list))

if selected_category == "Software Engineer":
    job_titles = [
        "Frontend Engineer","Backend Engineer","Fullstack Engineer",
        "Mobile Engineer","iOS Developer","Android Developer",
        "Desktop Engineer","Cloud Engineer","DevOps Engineer",
        "Site Reliability Engineer (SRE)","Platform Engineer",
        "Systems Administrator","Data Engineer",
        "Machine Learning Engineer","AI Engineer",
        "Deep Learning Engineer","Data Scientist",
        "Embedded Systems Engineer","Firmware Engineer",
        "Systems Programmer","Game Developer",
        "Graphics Engineer","Security Engineer",
        "DevSecOps Engineer","QA Automation Engineer",
        "Penetration Tester","Software Architect"
    ]
else:
    df_filtered = df[df["category"] == selected_category]
    job_titles = sorted(df_filtered["job_title"].unique())

selected_job_title = st.sidebar.selectbox("Job Role", job_titles)
show_graph = st.sidebar.checkbox("Show Knowledge Graph", value=True)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.markdown("## 📌 Job Description")
job_desc = st.text_area("Paste Job Description Here", height=280)

# --------------------------------------------------
# PROCESS BUTTON
# --------------------------------------------------
if st.button("🚀 Extract Skills"):

    if job_desc.strip() == "":
        st.warning("Please enter job description.")
    else:
        bilstm_skills = predict_skills(job_desc)

        if selected_category == "Software Engineer":
            rule_skills = rule_based_cs_skills(job_desc)
        else:
            rule_skills = []

        final_skills = sorted(set(bilstm_skills + rule_skills))

        st.markdown("## 📊 Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h4>Selected Role</h4>
            <p style="font-size:18px;">{selected_job_title}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <h4>Total Skills Extracted</h4>
            <p style="font-size:22px;">{len(final_skills)}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🛠 Extracted Skills")

        if final_skills:
            for skill in final_skills:
                st.markdown(
                    f'<span class="skill-tag">{skill}</span>',
                    unsafe_allow_html=True
                )
        else:
            st.error("No skills detected.")

        if show_graph and final_skills:
            st.markdown("## 🕸 Knowledge Graph")
            visualize_graph(selected_job_title, final_skills)