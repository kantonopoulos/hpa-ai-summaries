from plotnine import ggsave, ggplot, aes, geom_violin, geom_jitter, geom_crossbar, position_dodge, theme_classic, labs, theme, element_text

def flatten_values(val):
    """Recursively flatten lists/dicts to final values."""
    if isinstance(val, dict):
        vals = []
        for v in val.values():
            vals.extend(flatten_values(v))
        return vals
    elif isinstance(val, list):
        vals = []
        for v in val:
            vals.extend(flatten_values(v))
        return vals
    else:
        return [val]

def get_group(field):
    """Group fields by prefix."""
    if field.startswith("RNA"):
        return "RNA"
    elif field.startswith("Cancer prognostics"):
        return "Cancer prognostics"
    elif field.startswith("Blood"):
        return "Blood"
    elif field.startswith("Brain"):
        return "Brain"
    elif field.startswith("Antibody"):
        return "Antibody"
    elif field.startswith("Reliability"):
        return "Reliability"
    elif field.startswith("Subcellular"):
        return "Subcellular"
    elif field.startswith("Secretome"):
        return "Secretome"
    elif field.startswith("CCD"):
        return "CCD"
    elif field.startswith("Protein class"):
        return "Protein class"
    elif field.startswith("Interactions"):
        return "Interactions"
    else:
        return "Other"
    
def plot_metrics(df):
    """
    Plot violin plots for metrics with jittered points using plotnine.
    Adds a thick black line for the median of each metric, split by Model.
    Returns the plot and a DataFrame with median values for each metric.
    """
    # Calculate median values for each metric and Model
    medians = df.groupby(['metric', 'Model'])['value'].median().reset_index()
    medians.rename(columns={'value': 'median_value'}, inplace=True)

    # Create the plot
    plot = (
        ggplot(df, aes(x='metric', y='value', fill='Model')) +
        geom_violin(position=position_dodge(0.8), alpha=0.5) +  # Adjust dodge width for violin plots
        geom_crossbar(
            data=medians,
            mapping=aes(
                x='metric', 
                y='median_value', 
                ymin='median_value', 
                ymax='median_value', 
                fill='Model'
            ),
            position=position_dodge(0.8),  # Split crossbars by Model
            color='black',
            width=0.5,
            size=1.2
        ) +
        theme_classic() +
        labs(x='', y='Metric Value') +
        theme(
            axis_text_x=element_text(rotation=90, hjust=0.5),
            legend_position='top',
            legend_title=element_text(size=10),
            legend_text=element_text(size=8)
        )
    )

    return plot, medians

def save_plot(plot, filename):
    ggsave(plot, filename=filename, width=6, height=6, units="in", dpi=300)