# -*- coding: utf-8 -*-

# Importing Dependancies

import gradio as gr
from transformers import pipeline

"""# Loading Model Name"""

model_name = "deepset/roberta-base-squad2"

"""# Get Predictions

"""

nlu = pipeline('question-answering', model=model_name, tokenizer=model_name)

def func(context, question):
  input = {
      'question':question,
      'context':context
  }
  res = nlu(input)
  return res["answer"]

descr = "This is a question and Answer Web app, you give it a context and ask it questions based on the context provided"

app = gr.Interface(fn=func, inputs=[gr.inputs.Textbox(lines=3, placeholder="put in your context here..."),"text"], outputs="text", title="Question Answer App", description=descr)

app.launch()

