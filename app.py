import gradio as gr
import re
import os
import tempfile as _TemporaryFileWrapper
from backend import add_text, get_combined_result

def get_file_name(file: _TemporaryFileWrapper):
    file_name = os.path.basename(file.name).split(".")[0]
    return re.sub(r"[^\w\s]", " ", file_name)



with gr.Blocks() as demo:
    gr.Markdown("<h1><center>PDF QUERY </center></h1>")
    chatbot = gr.Chatbot(
        height=500,
        show_copy_button=True,
        avatar_images=("images/user.jpg", "images/bot.png"),
    )

    with gr.Column(scale=1):
        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(
                    type="file",
                    label="Upload a PDF",
                    file_types=[".pdf"],
                    height=80,
                )
            with gr.Column(scale=2):
                doc_description = gr.Textbox(
                    placeholder="Write a brief description of the document", label="\n"
                )
            with gr.Column(scale=4):
                text = gr.Textbox(placeholder="Write a question and submit", label="\n")
        with gr.Row():
            btn = gr.Button(value="Submit")
            clear = gr.ClearButton([text, chatbot, doc_description], variant="primary")

        with gr.Row():
             with gr.Accordion("Additional Parameters", open=False):
                gr.Markdown("Look at me...")
                k_slider = gr.Slider(label="Number of documents in similarity search", minimum=2, maximum=10, step=1, interactive=True, value=4, )
        with gr.Row():
            docs_text = gr.TextArea(label="Similarity Search Results")

    file.upload(fn=get_file_name, inputs=file, outputs=[doc_description])
    file.clear(
        lambda _, __, ___: ([], "", ""),
        inputs=[chatbot, text, doc_description],
        outputs=[chatbot, text, doc_description],
    )
    response = (
        gr.on(
            triggers=[btn.click, text.submit],
            fn=add_text,
            inputs=[chatbot, text],
            outputs=[chatbot, text],
        )
        .then(
            fn=get_combined_result,
            inputs=[file, chatbot, doc_description, k_slider],
            outputs=[chatbot, docs_text],
        )
        # .then(fn=get_response, inputs=[file, chatbot, doc_description], outputs=chatbot)
    )

    response.then(lambda: gr.update(interactive=True), None, [text], queue=False)


demo.queue()
demo.launch()
