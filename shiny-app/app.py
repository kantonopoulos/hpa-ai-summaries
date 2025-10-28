from shiny import App, Inputs, Outputs, Session, render, ui
import pandas as pd
from scripts.extract_data import extract_multiple_genes
from scripts.llm_models import concise_gene_summary, refined_gene_summary
from scripts.evaluation import evaluate_summary, evaluate_with_llm
from shiny import reactive
from plotnine import ggplot, aes, geom_bar

plot = None

# UI
app_ui = ui.page_fluid(
    ui.input_select("gene_id", "Select Gene ID:", []),
    ui.input_action_button("run", "Run"),
    ui.output_ui("output")
)

# Server
def server(input: Inputs, output: Outputs, session: Session):
    global plot

    hpa_df = pd.read_csv("data/hpa_genes.tsv", sep="\t")
    hpa_genes = hpa_df.iloc[:, 0].tolist()
    session.on_flushed(lambda: ui.update_select("gene_id", choices=hpa_genes))

    @render.ui
    @reactive.event(input.run)
    def output():
        global plot
        gene_id = input.gene_id()
        extracted_data = extract_multiple_genes([gene_id])
        gene_data = extracted_data.get(gene_id, {})

        if not gene_data:
            return ui.div("No data found for the selected gene.", class_="alert alert-warning")

        concise_summary = concise_gene_summary(gene_data)
        refined_summary = refined_gene_summary(concise_summary, gene_data)
        evaluation_results = evaluate_summary(gene_data, refined_summary)
        llm_evaluation = evaluate_with_llm(gene_data, refined_summary)
        evaluation_df = pd.DataFrame(evaluation_results)
        plot = (
            ggplot(evaluation_df, aes(x="metric", y="value")) +
            geom_bar(stat="identity")
        )

        return ui.div(
            ui.h4("AI-Summary:"),
            ui.p(refined_summary, class_="text-muted"),
            ui.h4("LLM Evaluation:"),
            ui.p(f"Score: {llm_evaluation['score']}, Comment: {llm_evaluation['comment']}", class_="text-muted"),
            ui.output_plot("evaluation_plot_output"),
            class_="card card-body bg-light"
        )

    @render.plot
    def evaluation_plot_output():
        global plot
        return plot

app = App(app_ui, server)
