# Project: AI-Based Gene Summarization Agent

## Summary
Large-scale omics resources, such as the Human Protein Atlas (HPA), provide extensive data on gene and protein expression across tissues. However, navigating this data is cognitively demanding due to the lack of concise, integrated overviews of gene function, localization, and disease relevance. This project develops an AI-based summarization agent that generates expert-style gene descriptions from structured HPA data using a locally deployable large language model (LLM). Through advanced prompt engineering, the system produces concise, accurate summaries capturing tissue specificity, subcellular localization, and disease context. Automated and LLM-based evaluations confirm factual consistency and readability while minimizing AI hallucinations.

---

## Project Structure

### /scripts
Contains Python scripts and notebooks for data extraction, summarization, evaluation, and visualization:
- **`extract_data.py`**: Extracts gene data from HPA and organizes it into structured JSON format.
- **`llm_models.py`**: Implements baseline and improved summarization models using prompt engineering.
- **`evaluation.py`**: Evaluates summaries using automated metrics and LLM-based evaluations.
- **`visualization_utils.py`**: Contains functions for generating plots and visualizations of evaluation results.
- **`pipeline.ipynb`**: Main pipeline notebook that integrates data extraction, summarization, evaluation, and visualization.

### /shiny-app
Contains the Shiny app for interactive gene summarization and evaluation:
- **`app.py`**: Main application file for the Shiny app, providing an interactive interface for gene selection, summary generation, and evaluation visualization.

---

## How to Use

### **Step 1: Set Up Ollama**
This project uses a locally deployable large language model (LLM) powered by Ollama. Follow these steps:
1. Download and install Ollama from their [official website](https://ollama.ai/).
2. Pull the required model by running:
   ```bash
   ollama pull gpt-oss:20b
   ```
   This step is required only once to download the model. It might take 2-4 hours.
3. Start the Ollama server before running the app or pipeline:
   ```bash
   ollama serve
   ```
   Keep the server running in the background.

### **Step 2: Install Dependencies**
Ensure you have Python installed along with the required libraries. Install dependencies using:
```bash
pip install -r requirements.txt
```

### **Step 3: Run the Pipeline**
The pipeline is implemented in `pipeline.ipynb`. Open the notebook in Jupyter or VS Code and execute the cells sequentially to:
1. Extract gene data from HPA.
2. Generate summaries using baseline and improved models.
3. Evaluate summaries using automated metrics and LLM-based evaluations.
4. Visualize results.

### **Step 4: Explore Results**
- Summaries are saved in the data folder (e.g., `gene_summaries_baseline.json`, `gene_summaries_refined.json`).
- Evaluation results and plots are generated for comparison between models.

---

## Alternative Step 3/4: Run the Shiny App

### **Step 3: Run the App**
To launch the Shiny app for gene summarization:
```bash
python shiny-app/app.py
```
The app provides an interactive interface for selecting genes, generating summaries, and visualizing evaluation results. It might take 30s-1min to create and evaluate each summary.

### **Step 4: Explore Results**
- Refined summaries and evaluation results are displayed directly in the app.
- Plots visualize the evaluation metrics for better insights.

---

## Note
This project was performed under the Data-Driven Life Sciences (DDLS 2025) course by KTH Royal Institute of Technology.