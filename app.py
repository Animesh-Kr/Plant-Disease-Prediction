import streamlit as st
import numpy as np
import json
import time
from pathlib import Path
from PIL import Image

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
)

# ── load model + labels (cached — only runs once per session) ─────────────────
@st.cache_resource
def load_model():
    """Load TFLite float16 model. Cached so it only loads once."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path="model_float16_quant.tflite")
    interp.allocate_tensors()
    ind  = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    return interp, ind, outd

@st.cache_resource
def load_labels():
    with open("class_indices.json") as f:
        raw = json.load(f)   # {"0": "Apple___Apple_scab", ...}
    # clean up display names
    return {int(k): v.replace("___", " — ").replace("_", " ")
            for k, v in raw.items()}

# ── inference ─────────────────────────────────────────────────────────────────
def predict(img: Image.Image, interp, ind, outd,
            class_map: dict, top_k: int = 5) -> list:
    """Resize, run TFLite inference, return top-k predictions."""
    img_size = ind["shape"][1]   # reads from model — always correct
    img      = img.convert("RGB").resize((img_size, img_size))
    arr      = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    interp.set_tensor(ind["index"], arr)
    interp.invoke()
    probs = interp.get_tensor(outd["index"])[0]   # (38,)
    top   = np.argsort(probs)[-top_k:][::-1]
    return [(class_map[i], float(probs[i])) for i in top]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌿 Plant Disease Detection")
st.markdown(
    "Upload a leaf image to identify the plant disease. "
    "Powered by **EfficientNetV2S** fine-tuned on the "
    "[PlantVillage dataset](https://arxiv.org/abs/1511.08060) "
    "(38 classes, 54,306 images)."
)

# sidebar — model info
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown("""
    **Architecture:** EfficientNetV2S  
    **Input size:** 384 × 384  
    **Classes:** 38 plant diseases  
    **Format:** TFLite float16  
    **Dataset:** PlantVillage (Hughes & Salathé, 2015)  
    """)
    st.markdown("---")
    st.markdown("**Supported plants:**")
    st.markdown(
        "Apple · Blueberry · Cherry · Corn · Grape · "
        "Orange · Peach · Pepper · Potato · Raspberry · "
        "Soybean · Squash · Strawberry · Tomato"
    )
    st.markdown("---")
    st.caption(
        "⚠️ Trained on lab images. May not generalise "
        "to field photos with occlusion or variable lighting."
    )

# load resources
with st.spinner("Loading model ..."):
    interp, ind, outd = load_model()
    class_map = load_labels()

# upload widget
uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Best results with clear, close-up photos of a single leaf.",
)

if uploaded is not None:
    img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Running inference ..."):
            t0      = time.perf_counter()
            results = predict(img, interp, ind, outd, class_map, top_k=5)
            elapsed = time.perf_counter() - t0

        top_class, top_conf = results[0]
        st.success(f"**{top_class}**")
        st.metric("Confidence", f"{top_conf*100:.1f}%",
                  help="Softmax probability for the top prediction.")
        st.caption(f"Inference time: {elapsed*1000:.0f} ms")

    # top-5 bar chart
    st.markdown("#### Top-5 predictions")
    labels = [r[0] for r in results]
    confs  = [r[1] * 100 for r in results]

    import pandas as pd
    chart_df = pd.DataFrame({"Disease": labels, "Confidence (%)": confs})
    st.bar_chart(chart_df.set_index("Disease"), height=260)

    # detailed table
    with st.expander("Full prediction table"):
        st.dataframe(
            chart_df.style.format({"Confidence (%)": "{:.2f}"}),
            use_container_width=True,
        )

    # disclaimer
    st.info(
        "This tool is for research and educational purposes only. "
        "Always consult an agronomist for crop management decisions.",
        icon="ℹ️",
    )

else:
    # placeholder when no image uploaded
    st.markdown("---")
    st.markdown("### 👆 Upload a leaf image to get started")
    st.markdown(
        "The model will predict the most likely disease and show "
        "confidence scores for the top 5 candidates."
    )
