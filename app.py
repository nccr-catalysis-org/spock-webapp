#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gradio as gr

CSS = """
"""


with gr.Blocks(css=CSS) as block:
    gr.Markdown(
        """
    ## Volcano plots
    """
    )

block.launch()
