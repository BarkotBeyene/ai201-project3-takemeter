"""
TakeMeter deployed interface (stretch feature).

Loads the fine-tuned DistilBERT from a local directory and classifies
arbitrary r/television posts via a simple Gradio UI.

Usage:
  pip install gradio transformers torch
  python src/interface.py --model-dir ./results/checkpoint-best

The --model-dir should be the checkpoint directory downloaded from Colab
(or a HuggingFace model ID if you pushed it there).
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL2ID = {"analysis": 0, "reaction": 1, "prediction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

LABEL_DESCRIPTIONS = {
    "analysis":   "Evaluates writing/direction/themes with specific textual evidence.",
    "reaction":   "Immediate emotional response — verdict without argument.",
    "prediction": "Speculative forecast about future episodes or seasons.",
}


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def classify(text: str, tokenizer, model) -> tuple[str, dict[str, float]]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    scores = {ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)}
    predicted = max(scores, key=scores.__getitem__)
    return predicted, scores


def build_gradio_app(tokenizer, model):
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("Install gradio: pip install gradio")

    def predict(text):
        if not text.strip():
            return "—", {}, ""
        label, scores = classify(text, tokenizer, model)
        confidence = scores[label]
        desc = LABEL_DESCRIPTIONS[label]
        summary = f"**{label}** ({confidence:.1%} confidence)\n\n_{desc}_"
        return summary, scores, label

    with gr.Blocks(title="TakeMeter") as demo:
        gr.Markdown("## TakeMeter — r/television Discourse Classifier")
        gr.Markdown(
            "Paste a Reddit post or comment. TakeMeter classifies it as "
            "**analysis**, **reaction**, or **prediction**."
        )
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Post text",
                    placeholder="Paste a post from r/television...",
                    lines=6,
                )
                classify_btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                result_md = gr.Markdown(label="Result")
                confidence_bar = gr.Label(
                    label="Confidence scores",
                    num_top_classes=3,
                )

        classify_btn.click(
            fn=predict,
            inputs=text_input,
            outputs=[result_md, confidence_bar, gr.Textbox(visible=False)],
        )

        gr.Examples(
            examples=[
                ["That finale absolutely destroyed me. I was not ready. This show is one of the best things on TV right now."],
                ["The writing this season has been deliberately fragmented — every episode cuts away right before an emotional payoff, mirroring the protagonist's avoidance of grief. It's frustrating on purpose."],
                ["Calling it now: [character] is the mole. The show has been dropping hints since episode 3 — the phone call in the background, the look in the finale."],
            ],
            inputs=text_input,
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./results/checkpoint-best",
                        help="Path to fine-tuned model directory or HF model ID")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    tokenizer, model = load_model(args.model_dir)

    demo = build_gradio_app(tokenizer, model)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
