import os
import gradio as gr
import torch
import numpy as np
from transformers import pipeline

name_list = ['microsoft/biogpt', 'google/flan-t5-xxl', 'facebook/galactica-1.3b', 'gpt2']

examples = [['COVID-19 is'],['We describe an 11-year-old previously healthy male who presented with eight days of fever']] 

print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

pipe_biogpt = pipeline("text-generation", model="microsoft/BioGPT-Large", device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})
pipe_flan_t5 = pipeline("text-generation", model="google/flan-t5-xxl", device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})
pipe_galactica = pipeline("text-generation", model="facebook/galactica-1.3b", device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})
pipe_gpt_2 = pipeline("text-generation", model="gpt2", device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})

title = "Compare generative biomedical LLMs!"
description = "**Disclaimer:** this demo was made for research purposes only and should not be used for medical purposes."

def inference(text):
  output_biogpt = pipe_biogpt(text, max_length=100)[0]["generated_text"]
  output_fla_t5 = pipe_flan_t5(text, max_length=100)[0]["generated_text"]
  output_galactica = pipe_galactica(text, max_length=100)[0]["generated_text"]
  output_gpt_2 = pipe_gpt_2(text, max_length=100)[0]["generated_text"]
  return [
      output_biogpt, 
      output_fla_t5,
      output_galactica,
      output_gpt_2
  ]

io = gr.Interface(
  inference,
  gr.Textbox(lines=3),
  outputs=[
    gr.Textbox(lines=3, label="BioGPT-Large"),
    gr.Textbox(lines=3, label="Flan-t5"),
    gr.Textbox(lines=3, label="Galactica 1.3B"),
    gr.Textbox(lines=3, label="GPT-2"),
  ],
  title=title,
  description=description,
  examples=examples
)
io.launch()