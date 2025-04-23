""" Visualize logit lens outputs 
    Currently doesn't support custom layer exploration
"""

import numpy as np
import plotly.express as px
import plotly.io as pio
import pandas as pd
from typing import List

pio.renderers.default = "plotly_mimetype+notebook_connected+notebook+colab"

class Metric:
    """ Enum to define different logit lens metrics """
    PROB = "Probability"
    KLD = "KL Divergence"
    PC = "Principal Component"
    ENT = "Entropy"
    RK = "Rank"

class LogitVis():
    
    def __init__(self, metric : Metric, metric_data : np.ndarray, input_words : List[str], words : List[List[str]]):
        """_summary_

        Args:
            metric (Metric): _description_
            metric_data (np.ndarray): _description_
            input_words (List[str]): _description_
            words (List[List[str]]): _description_
        """
    
        self.metric = metric
        self.metric_data = metric_data
        self.input_words = input_words
        self.words = words
        self.transposed_words = list(map(list, zip(*self.words)))
    
    def heatmap(self):
        """
        Create a heatmap plot for the metric

        Returns: a heatmap plot
        """
        
        fig = px.imshow(
            self.metric_data,
            x=list(range(len(self.words))),
            y=self.input_words,
            color_continuous_scale=px.colors.diverging.RdYlBu_r,
            text_auto=True,
            labels=dict(x="Layers", y="Input Tokens", color=self.metric),
        )
        fig.update_layout(title=f"{self.metric} Heatmap", xaxis_tickangle=-45)
        
        # Add transposed words
        fig.update_traces(text = self.transposed_words, 
                          texttemplate="%{text}")
        
        # Display more granular information on hover
        fig.update_traces(hovertemplate =
            "Input Token: %{y}<br>" +
            "Layer: %{x}<br>" +
            "Predicted Token: %{text}<br>" +
            f"{self.metric}: " + "%{z:.4f}" +
            "<extra></extra>")
        
        return fig

    def lineplotToken(self, token : int):
        """
        Plot a lineplot for the metric for a given input token

        Args:
            token (int): Position of the input token
            
        Returns: a line plot
        """
        
        # Create a dataframe of lines
        df = pd.DataFrame({
            "Layer" : list(range(len(self.words))),
            "Current Layer Prediction" : self.metric_data[token]
        })
        
        df_melted = df.melt(id_vars=["Layer"], var_name="Token Source", value_name=self.metric)
        
        # Create the plot
        fig = px.line(df_melted, 
                      x="Layer", 
                      y=self.metric, 
                      color="Token Source",
                      title=f"{self.metric} for token predicted after \"{self.input_words[token]}\"",
                      labels={self.metric: self.metric, "Layer": "Layer"})
        fig.update_traces(mode="lines+markers")
        fig.update_traces(text = self.transposed_words[token],
                          texttemplate="%{text}")
        fig.update_traces(hovertemplate =
            f"Input Token: {self.input_words[token]}<br>" +
            "Layer: %{x}<br>" +
            "Predicted Token: %{text}<br>" +
            f"{self.metric}: " + "%{y:.4f}<br>" +
            "<extra></extra>")
        
        return fig
        
    def lineplotLayer(self, layer : int):
        """
        Plot a lineplot for the metric for a given model layer

        Args:
            layer (int): layer
            
        Returns: a line plot
        """
        
        # Create a dataframe of lines
        df = pd.DataFrame({
            "Input Token" : self.input_words,
            "Current Token Prediction" : self.metric_data.T[layer]
        })
        
        df_melted = df.melt(id_vars=["Input Token"], var_name="Layer Source", value_name=self.metric)
        
        layer_name = layer if layer is not -1 else len(self.words) - 1
        
        fig = px.line(df_melted,
                      x="Input Token",
                      y=self.metric,
                      color="Layer Source",
                      title=f"{self.metric} for layer {layer_name}",
                      labels={self.metric: self.metric, "Token": "Token"})
        fig.update_traces(mode="lines+markers")
        fig.update_traces(text = self.transposed_words[layer],
                          texttemplate="%{text}")
        fig.update_traces(hovertemplate =
            "Input Token: {x}<br>" +
            f"Layer: {layer_name}<br>" +
            "Predicted Token: %{text}<br>" +
            f"{self.metric}: " + "%{y:.4f}<br>" +
            "<extra></extra>")
        
        return fig
        
        
        
