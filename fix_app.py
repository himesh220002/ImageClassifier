import os

with open("app.py", "r") as f:
    lines = f.readlines()

new_lines = []
in_tabs = False
for i, line in enumerate(lines):
    if line.startswith("tab1, tab2, tab3 = st.tabs"):
        new_lines.extend([
            "# --- Main Dashboard ---\n",
            "is_cloud_mode = not os.path.exists(os.path.join(BASE_DIR, \"data\", \"raw\"))\n\n",
            "if is_cloud_mode:\n",
            "    st.success(\"☁️ **Cloud Inference Mode Active**: Model training is disabled on the cloud. Running global inference on pre-trained custom models.\")\n",
            "    st.markdown(\"### 🔄 MLOps Workflow\")\n",
            "    st.info(\"🖥️ **Local Data Ingestion & Training** ➡️ 📦 **Save `.pth` Model** ➡️ ☁️ **Deploy to Streamlit Cloud for Global Inference**\")\n",
            "    st.divider()\n",
            "    inference_container = st.container()\n",
            "else:\n",
            "    tab1, tab2, tab3 = st.tabs([\"📂 Data Ingestion\", \"⚙️ Model Training\", \"🧠 Model Inference\"])\n",
            "    inference_container = tab3\n\n",
            "    with tab1:\n"
        ])
        in_tabs = True
        continue
    
    if line.startswith("# --- Main Dashboard ---"):
        continue

    if in_tabs:
        if line.startswith("with tab1:"):
            continue # already added
        if line.startswith("# Initialize monitoring session state"):
            in_tabs = False
            new_lines.append(line)
        else:
            new_lines.append("    " + line if line.strip() else "\n")
    else:
        new_lines.append(line)

with open("app.py", "w") as f:
    f.writelines(new_lines)
