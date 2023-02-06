import re
import os

from cleantext import clean
import gradio as gr
from tqdm.auto import tqdm
from transformers import pipeline


checker_model_name = "textattack/roberta-base-CoLA"
corrector_model_name = "pszemraj/flan-t5-large-grammar-synthesis"

# pipelines
checker = pipeline(
    "text-classification",
    checker_model_name,
)

if os.environ.get("HF_DEMO_NO_USE_ONNX") is None:
    # load onnx runtime unless HF_DEMO_NO_USE_ONNX is set
    from optimum.pipelines import pipeline

    corrector = pipeline(
        "text2text-generation", model=corrector_model_name, accelerator="ort"
    )
else:
    corrector = pipeline("text2text-generation", corrector_model_name)


def split_text(text: str) -> list:
    # Split the text into sentences using regex
    sentences = re.split(r"(?<=[^A-Z].[.?]) +(?=[A-Z])", text)

    # Initialize a list to store the sentence batches
    sentence_batches = []

    # Initialize a temporary list to store the current batch of sentences
    temp_batch = []

    # Iterate through the sentences
    for sentence in sentences:
        # Add the sentence to the temporary batch
        temp_batch.append(sentence)

        # If the length of the temporary batch is between 2 and 3 sentences, or if it is the last batch, add it to the list of sentence batches
        if len(temp_batch) >= 2 and len(temp_batch) <= 3 or sentence == sentences[-1]:
            sentence_batches.append(temp_batch)
            temp_batch = []

    return sentence_batches


def correct_text(text: str, checker, corrector, separator: str = " ") -> str:
    # Split the text into sentence batches
    sentence_batches = split_text(text)

    # Initialize a list to store the corrected text
    corrected_text = []

    # Iterate through the sentence batches
    for batch in tqdm(
        sentence_batches, total=len(sentence_batches), desc="correcting text.."
    ):
        # Join the sentences in the batch into a single string
        raw_text = " ".join(batch)

        # Check the grammar quality of the text using the text-classification pipeline
        results = checker(raw_text)

        # Only correct the text if the results of the text-classification are not LABEL_1 or are LABEL_1 with a score below 0.9
        if results[0]["label"] != "LABEL_1" or (
            results[0]["label"] == "LABEL_1" and results[0]["score"] < 0.9
        ):
            # Correct the text using the text-generation pipeline
            corrected_batch = corrector(raw_text)
            corrected_text.append(corrected_batch[0]["generated_text"])
        else:
            corrected_text.append(raw_text)

    # Join the corrected text into a single string
    corrected_text = separator.join(corrected_text)

    return corrected_text


def update(text: str):
    text = clean(text[:4000], lower=False)
    return correct_text(text, checker, corrector)


with gr.Blocks() as demo:
    gr.Markdown("# <center>Robust Grammar Correction with FLAN-T5</center>")
    gr.Markdown(
        "**Instructions:** Enter the text you want to correct in the textbox below (_text will be truncated to 4000 characters_). Click 'Process' to run."
    )
    gr.Markdown(
        """Models:
    - `textattack/roberta-base-CoLA` for grammar quality detection
    - `pszemraj/flan-t5-large-grammar-synthesis` for grammar correction
    """
    )
    with gr.Row():
        inp = gr.Textbox(
            label="input",
            placeholder="PUT TEXT TO CHECK & CORRECT BROSKI",
            value="I wen to the store yesturday to bye some food. I needd milk, bread, and a few otter things. The store was really crowed and I had a hard time finding everyting I needed. I finaly made it to the check out line and payed for my stuff.",
        )
        out = gr.Textbox(label="output", interactive=False)
    btn = gr.Button("Process")
    btn.click(fn=update, inputs=inp, outputs=out)
    gr.Markdown("---")
    gr.Markdown(
        "- see the [model card](https://huggingface.co/pszemraj/flan-t5-large-grammar-synthesis) for more info"
    )
    gr.Markdown("- if experiencing long wait times, feel free to duplicate the space!")
demo.launch()
