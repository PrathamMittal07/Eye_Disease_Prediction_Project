import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import base64
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image
from collections import Counter
from recommendation import cnv, dme, drusen, normal

# Class names
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

@st.cache_resource
def load_models():
    try:
        mobilenet_model = tf.keras.models.load_model("mobilenetv3_model.keras")
        resnet_model = tf.keras.models.load_model("resnet50_model.keras")
        efficient_model = tf.keras.models.load_model("efficientnetb0_model.keras")
        return mobilenet_model, resnet_model, efficient_model
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

def preprocess_image(img_path, model_type):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        x = np.expand_dims(img_array, axis=0)
        if model_type == 'mobilenet':
            return mobilenet_preprocess(x)
        elif model_type == 'resnet':
            return resnet_preprocess(x)
        elif model_type == 'efficientnet':
            return efficientnet_preprocess(x)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        st.error(f"Image preprocessing failed: {str(e)}")
        st.stop()

def predict_with_all_models(img_path):
    mobilenet_model, resnet_model, efficient_model = load_models()
    x1 = preprocess_image(img_path, 'mobilenet')
    x2 = preprocess_image(img_path, 'resnet')
    x3 = preprocess_image(img_path, 'efficientnet')
    pred1 = mobilenet_model.predict(x1, verbose=0)[0]
    pred2 = resnet_model.predict(x2, verbose=0)[0]
    pred3 = efficient_model.predict(x3, verbose=0)[0]
    return pred1, pred2, pred3

def get_final_prediction(pred1, pred2, pred3):
    preds = [pred1, pred2, pred3]
    labels = [np.argmax(p) for p in preds]
    counter = Counter(labels)
    
    if len(counter) == 3:
        avg_confs = [np.mean([p[i] for p in preds]) for i in range(len(class_names))]
        final_idx = np.argmax(avg_confs)
        return final_idx, avg_confs[final_idx], preds
    else:
        final_idx = counter.most_common(1)[0][0]
        avg_conf = np.mean([p[final_idx] for p in preds])
        return final_idx, avg_conf, preds

def generate_pdf(pred_label, confidence, img_path):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, "OCT Prediction Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Prediction: {pred_label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(200, 10, f"Confidence: {confidence * 100:.2f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        if os.path.exists(img_path):
            pdf.image(img_path, x=30, y=50, w=150)
        
        tmp_pdf_path = os.path.join(tempfile.gettempdir(), "oct_report.pdf")
        pdf.output(tmp_pdf_path)
        
        with open(tmp_pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode()
        
        os.unlink(tmp_pdf_path)
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="oct_report.pdf">📄 Download PDF Report</a>'
        return href
    except Exception as e:
        st.error(f"Failed to generate PDF: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide", page_title="OCT Retinal Analysis")
    st.sidebar.title("Navigation")
    
    if st.sidebar.button("🔄 Clear Cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Identification"])

    if app_mode == "Home":
        st.markdown("## **OCT Retinal Disease Prediction**")
        st.markdown("Upload your retinal OCT image to get analysis results from multiple deep learning models.")
        st.markdown("Created by:")
        st.markdown(""" 
          - Aditya Malav-23BCE10319
                    """)

    elif app_mode == "About":
        st.markdown("### About This Application")
        st.markdown("""
        This app uses three state-of-the-art CNN models to classify OCT scans:
        - MobileNetV3
        - ResNet50
        - EfficientNetB0
        
        The system combines predictions from all models for more accurate results.
        """)

    elif app_mode == "Disease Identification":
        st.header("OCT Image Analysis")
        uploaded_file = st.file_uploader("Upload OCT scan (JPEG/PNG):", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    img_path = tmp_file.name
                
                st.image(img_path, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Processing..."):
                        pred1, pred2, pred3 = predict_with_all_models(img_path)
                        final_idx, final_conf, all_preds = get_final_prediction(pred1, pred2, pred3)
                        final_label = class_names[final_idx]

                    st.success(f"✅ Final Prediction: **{final_label}** ({final_conf * 100:.2f}% confidence)")

                    # Color coding background but keep text black for readability
                    confidence_color = "green" if final_conf >= 0.85 else "yellow" if final_conf >= 0.6 else "red"
                    st.markdown(
                        f'<div style="background:{confidence_color};color:black;padding:8px;border-radius:5px;text-align:center;">'
                        f'<b>Confidence Level:</b> {final_conf*100:.2f}%</div>', 
                        unsafe_allow_html=True
                    )

                    # Model predictions
                    st.subheader("Model Predictions Comparison")
                    cols = st.columns(3)
                    for col, name, pred in zip(cols, ['MobileNetV3', 'ResNet50', 'EfficientNetB0'], [pred1, pred2, pred3]):
                        idx = np.argmax(pred)
                        col.metric(label=name, value=class_names[idx], delta=f"{pred[idx]*100:.1f}%")

                    # Detailed probabilities
                    st.subheader("Detailed Class Probabilities")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    width = 0.25
                    x = np.arange(len(class_names))
                    
                    for i, (name, pred) in enumerate(zip(['MobileNetV3', 'ResNet50', 'EfficientNetB0'], [pred1, pred2, pred3])):
                        ax.bar(x + i*width, pred*100, width, label=name)
                    
                    ax.set_xticks(x + width)
                    ax.set_xticklabels(class_names)
                    ax.set_ylabel("Probability (%)")
                    ax.set_title("Model Confidence Across Classes")
                    ax.legend()
                    st.pyplot(fig)

                    # Clinical info
                    st.subheader(f"Clinical Information: {final_label}")
                    info_content = {
                        0: ("Choroidal Neovascularization (CNV)", cnv),
                        1: ("Diabetic Macular Edema (DME)", dme),
                        2: ("Drusen Deposits", drusen),
                        3: ("Normal Retina", normal)
                    }[final_idx]
                    
                    st.markdown(f"**{info_content[0]}**")
                    st.markdown(info_content[1])

                    # PDF Download
                    st.subheader("Download Report")
                    pdf_link = generate_pdf(final_label, final_conf, img_path)
                    if pdf_link:
                        st.markdown(pdf_link, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
            finally:
                if 'img_path' in locals() and os.path.exists(img_path):
                    os.unlink(img_path)

if __name__ == "__main__":
    main()
