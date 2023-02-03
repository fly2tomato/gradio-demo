from transformers import pipeline
import gradio as gr


def hello(i):
    classifier = pipeline("sentiment-analysis")
    a = classifier(i)
    return a

iface = gr.Interface(fn=hello, inputs="text", outputs="text")
iface.launch()
