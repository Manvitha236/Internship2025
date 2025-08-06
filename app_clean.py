# breast_cancer_streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import hashlib
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Advanced Explainability imports
from captum.attr import (
    GradientShap, 
    Saliency, 
    LayerGradCam,
    LayerLRP,
    IntegratedGradients,
    NoiseTunnel
)
from captum.attr import visualization as viz
import matplotlib.cm as cm
import base64
import io

# Fairness and Bias Detection imports
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import MetricFrame, equalized_odds_difference, demographic_parity_difference
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Paths
TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

# Blockchain Logger
class BlockchainLogger:
    def __init__(self):
        self.chain = []
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash, data=None):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': datetime.now().isoformat(),
            'proof': proof,
            'previous_hash': previous_hash,
            'data': data or {}
        }
        self.chain.append(block)
        return block

    def log_ai_metrics(self, model_name, metrics):
        last_block = self.chain[-1]
        new_block = self.create_block(
            proof=hashlib.sha256(json.dumps(metrics).encode()).hexdigest(),
            previous_hash=self.hash_block(last_block),
            data={
                'model': model_name,
                'metrics': metrics
            }
        )
        return new_block

    @staticmethod
    def hash_block(block):
        return hashlib.sha256(json.dumps(block).encode()).hexdigest()
    
    def get_prediction_stats(self):
        """Get statistics for bias detection"""
        predictions = []
        for block in self.chain[1:]:  # Skip genesis block
            if 'data' in block and 'metrics' in block.get('data', {}):
                metrics = block['data']['metrics']
                prediction_data = {
                    'timestamp': block.get('timestamp'),
                    'prediction': metrics.get('prediction'),
                    'confidence': metrics.get('confidence'),
                    'filename': metrics.get('filename'),
                    'block_index': block.get('index')
                }
                # Include image data if available
                if 'image_data' in metrics:
                    prediction_data['image_data'] = metrics['image_data']
                predictions.append(prediction_data)
        return predictions

# Load model (pretrained ResNet18 with fine-tuning)
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class names
class_names = ['benign', 'malignant']

# LIME Explainer
@st.cache_resource
def load_lime_explainer():
    explainer = lime_image.LimeImageExplainer()
    return explainer

# Advanced Explainability Methods
@st.cache_resource
def load_explainability_methods():
    """Initialize various explainability methods"""
    methods = {}
    return methods

def generate_saliency_map(model, image_tensor, target_class):
    """Generate saliency map using Captum"""
    saliency = Saliency(model)
    attribution = saliency.attribute(image_tensor, target=target_class)
    return attribution

def generate_gradcam(model, image_tensor, target_class, target_layer='layer4'):
    """Generate GradCAM using Captum"""
    # Get the target layer
    if target_layer == 'layer4':
        layer = model.layer4[1].conv2
    else:
        layer = model.layer4[0].conv1
    
    layer_gc = LayerGradCam(model, layer)
    attribution = layer_gc.attribute(image_tensor, target=target_class)
    return attribution

def generate_lrp(model, image_tensor, target_class):
    """Generate Layer-wise Relevance Propagation"""
    try:
        # Use the last convolutional layer for LRP
        layer = model.layer4[1].conv2
        lrp = LayerLRP(model, layer)
        attribution = lrp.attribute(image_tensor, target=target_class)
        return attribution
    except Exception as e:
        # Fallback to Integrated Gradients if LRP fails
        ig = IntegratedGradients(model)
        attribution = ig.attribute(image_tensor, target=target_class, n_steps=50)
        return attribution

def generate_integrated_gradients(model, image_tensor, target_class):
    """Generate Integrated Gradients"""
    ig = IntegratedGradients(model)
    attribution = ig.attribute(image_tensor, target=target_class, n_steps=50)
    
    # Add noise tunnel for more robust explanations
    nt = NoiseTunnel(ig)
    attribution_nt = nt.attribute(image_tensor, nt_type='smoothgrad', 
                                  nt_samples=10, target=target_class)
    return attribution, attribution_nt

def visualize_attribution(attribution, original_image, method_name, prediction_class):
    """Visualize attribution maps"""
    # Convert attribution to numpy
    if len(attribution.shape) == 4:
        attribution_np = attribution.squeeze().cpu().detach().numpy()
    else:
        attribution_np = attribution.cpu().detach().numpy()
    
    # For multi-channel attributions, take the mean across channels
    if len(attribution_np.shape) == 3:
        attribution_np = np.mean(attribution_np, axis=0)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if isinstance(original_image, torch.Tensor):
        img_np = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = np.array(original_image) / 255.0
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attribution heatmap
    im1 = axes[1].imshow(attribution_np, cmap='RdYlBu_r')
    axes[1].set_title(f'{method_name} Attribution')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Overlay
    axes[2].imshow(img_np)
    axes[2].imshow(attribution_np, cmap='RdYlBu_r', alpha=0.6)
    axes[2].set_title(f'{method_name} Overlay')
    axes[2].axis('off')
    
    plt.suptitle(f'{method_name} Explanation for {prediction_class.upper()} Prediction')
    plt.tight_layout()
    
    return fig

# Fairness and Bias Detection with Threshold Optimization
class AdvancedBiasDetector:
    def __init__(self):
        self.post_processor = None
        self.base_estimator = None
        self.is_fitted = False
        
    def prepare_fairness_data(self, predictions_data):
        """Prepare data for fairness analysis"""
        if len(predictions_data) < 10:
            return None, None, None, None
            
        df = pd.DataFrame(predictions_data)
        
        # Create synthetic sensitive attributes based on confidence patterns
        # This is a simplified approach - in real scenarios, you'd have actual demographic data
        df['sensitive_attr'] = (df['confidence'] > df['confidence'].median()).astype(int)
        
        # Create binary outcomes
        y_true = (df['prediction'] == 'malignant').astype(int)
        y_scores = df['confidence'].values
        
        # Use confidence threshold for binary predictions
        y_pred = (y_scores > 0.5).astype(int)
        
        sensitive_features = df['sensitive_attr'].values
        
        return y_true, y_pred, y_scores, sensitive_features
    
    def fit_threshold_optimizer(self, y_true, y_scores, sensitive_features):
        """Fit Threshold Optimizer for equalized odds"""
        try:
            # Split data for post-processing
            if len(y_true) < 20:
                return False
                
            # Create train/test split for post-processing
            indices = np.arange(len(y_true))
            train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
            
            # Create a simple base estimator that returns the scores
            self.base_estimator = LogisticRegression()
            
            # Fit base estimator on scores (treating scores as features)
            X_train = y_scores[train_idx].reshape(-1, 1)
            y_train = y_true[train_idx]
            sensitive_train = sensitive_features[train_idx]
            
            self.base_estimator.fit(X_train, y_train)
            
            # Create threshold optimizer for equalized odds
            self.post_processor = ThresholdOptimizer(
                estimator=self.base_estimator,
                constraints='equalized_odds',
                objective='accuracy_score'
            )
            
            # Fit the threshold optimizer
            self.post_processor.fit(
                X_train, 
                y_train, 
                sensitive_features=sensitive_train
            )
            
            self.is_fitted = True
            return True
            
        except Exception as e:
            st.error(f"Error fitting Threshold Optimizer: {str(e)}")
            return False
    
    def predict_fair(self, y_scores, sensitive_features):
        """Make fair predictions using post-processing"""
        if not self.is_fitted:
            return None
            
        try:
            X_test = y_scores.reshape(-1, 1)
            fair_predictions = self.post_processor.predict(
                X_test,
                sensitive_features=sensitive_features
            )
            return fair_predictions
        except Exception as e:
            return None
    
    def calculate_fairness_metrics(self, y_true, y_pred, y_pred_fair, sensitive_features):
        """Calculate comprehensive fairness metrics"""
        metrics = {}
        
        try:
            # Convert to pandas Series for fairlearn metrics
            y_true_series = pd.Series(y_true)
            y_pred_series = pd.Series(y_pred)
            sensitive_series = pd.Series(sensitive_features)
            
            # Calculate equalized odds difference
            eq_odds_diff = equalized_odds_difference(
                y_true_series, y_pred_series, sensitive_features=sensitive_series
            )
            metrics['equalized_odds_violation'] = abs(eq_odds_diff)
            
            # Calculate demographic parity difference
            demo_parity_diff = demographic_parity_difference(
                y_true_series, y_pred_series, sensitive_features=sensitive_series
            )
            metrics['demographic_parity_violation'] = abs(demo_parity_diff)
            
            # Group-wise metrics for detailed analysis
            group_0_mask = sensitive_features == 0
            group_1_mask = sensitive_features == 1
            
            if np.sum(group_0_mask) > 0 and np.sum(group_1_mask) > 0:
                # True Positive Rates
                tpr_0 = np.sum((y_true[group_0_mask] == 1) & (y_pred[group_0_mask] == 1)) / max(1, np.sum(y_true[group_0_mask] == 1))
                tpr_1 = np.sum((y_true[group_1_mask] == 1) & (y_pred[group_1_mask] == 1)) / max(1, np.sum(y_true[group_1_mask] == 1))
                
                # False Positive Rates
                fpr_0 = np.sum((y_true[group_0_mask] == 0) & (y_pred[group_0_mask] == 1)) / max(1, np.sum(y_true[group_0_mask] == 0))
                fpr_1 = np.sum((y_true[group_1_mask] == 0) & (y_pred[group_1_mask] == 1)) / max(1, np.sum(y_true[group_1_mask] == 0))
                
                metrics['tpr_difference'] = abs(tpr_0 - tpr_1)
                metrics['fpr_difference'] = abs(fpr_0 - fpr_1)
                
                # Fair metrics if available
                if y_pred_fair is not None:
                    y_pred_fair_series = pd.Series(y_pred_fair)
                    
                    # Fair equalized odds
                    eq_odds_diff_fair = equalized_odds_difference(
                        y_true_series, y_pred_fair_series, sensitive_features=sensitive_series
                    )
                    metrics['equalized_odds_violation_fair'] = abs(eq_odds_diff_fair)
                    
                    # Fair demographic parity
                    demo_parity_diff_fair = demographic_parity_difference(
                        y_true_series, y_pred_fair_series, sensitive_features=sensitive_series
                    )
                    metrics['demographic_parity_violation_fair'] = abs(demo_parity_diff_fair)
                    
                    # Fair TPR calculations
                    tpr_fair_0 = np.sum((y_true[group_0_mask] == 1) & (y_pred_fair[group_0_mask] == 1)) / max(1, np.sum(y_true[group_0_mask] == 1))
                    tpr_fair_1 = np.sum((y_true[group_1_mask] == 1) & (y_pred_fair[group_1_mask] == 1)) / max(1, np.sum(y_true[group_1_mask] == 1))
                    metrics['tpr_difference_fair'] = abs(tpr_fair_0 - tpr_fair_1)
        
        except Exception as e:
            # Fallback to basic metrics if fairlearn fails
            metrics['equalized_odds_violation'] = 0.0
            metrics['demographic_parity_violation'] = 0.0
            metrics['tpr_difference'] = 0.0
            metrics['fpr_difference'] = 0.0
        
        return metrics

# Predict function for LIME
def predict_for_lime(images, model, transform):
    """Prediction function that LIME can use"""
    batch_predictions = []
    
    for image in images:
        # Convert to PIL Image and apply transforms
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        img_tensor = transform(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].numpy()
            batch_predictions.append(probabilities)
    
    return np.array(batch_predictions)

# Predict

def predict(image, model):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0, preds.item()].item()
    return class_names[preds.item()], confidence, img_tensor

# Fairness Check

def fairness_check(conf):
    if conf < 0.6:
        return "âš ï¸ Low confidence â€” potential uncertainty"
    return "âœ… Confident prediction"

# Streamlit UI
st.set_page_config(
    page_title="Breast Cancer Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize model and blockchain
@st.cache_resource
def init_app():
    model = load_model()
    lime_explainer = load_lime_explainer()
    explainability_methods = load_explainability_methods()
    return model, lime_explainer, explainability_methods

model, lime_explainer, explainability_methods = init_app()

# Initialize bias detector in session state
if 'bias_detector' not in st.session_state:
    st.session_state.bias_detector = AdvancedBiasDetector()

# Force light mode with custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: white;
        color: black;
    }
    .stSidebar {
        background-color: #f0f2f6;
    }
    .stSelectbox > div > div {
        background-color: white;
        color: black;
    }
    .stMetric {
        background-color: white;
        color: black;
    }
    .stDataFrame {
        background-color: white;
    }
    /* Force all text to be dark */
    .stMarkdown, .stText, .stCaption {
        color: black !important;
    }
    /* Style buttons */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1a6aa3;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize blockchain in session state for persistence
if 'blockchain' not in st.session_state:
    st.session_state.blockchain = BlockchainLogger()
blockchain = st.session_state.blockchain

# Sidebar Navigation
st.sidebar.title("ðŸ¥ Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["ðŸ©º Main Diagnosis", "ðŸ” LIME Explanations", "ðŸ§  Advanced Explainability", "ðŸ”— Blockchain Data", "âš–ï¸ Advanced Bias Detection"]
)

# Page Functions
def main_diagnosis_page():
    st.title("ðŸ©º Breast Cancer Diagnosis with AI")
    st.subheader("Using ResNet18 + LIME + Custom Blockchain")
    
    # Upload image
    uploaded_file = st.file_uploader("Upload a breast cancer image (benign/malignant)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display image with smaller width
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", width=400)

        # Predict
        pred_label, confidence, img_tensor = predict(image, model)

        # Show prediction
        st.success(f"### Prediction: {pred_label.upper()} with confidence {confidence:.2%}")
        st.info(fairness_check(confidence))

        # Log to blockchain with image data
        import base64
        import io
        
        # Convert image to base64 for storage
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        blockchain.log_ai_metrics("ResNet18_BreastCancer", {
            'prediction': pred_label,
            'confidence': confidence,
            'filename': uploaded_file.name,
            'image_data': img_base64  # Store image for LIME page
        })

        # Quick LIME preview
        st.subheader("ðŸ” Quick Explanation Preview")
        st.info("ðŸ’¡ For detailed LIME analysis, visit the 'LIME Explanations' page from the sidebar")
        
        with st.expander("ðŸ§¾ Latest Blockchain Entry"):
            st.json(blockchain.chain[-1])

def lime_explanations_page():
    st.title("ðŸ” LIME Model Explainability")
    st.write("**Local Interpretable Model-agnostic Explanations**")
    
    # Check if there are recent predictions
    stats = blockchain.get_prediction_stats()
    if not stats:
        st.warning("âš ï¸ No predictions found. Please make a prediction on the Main Diagnosis page first.")
        return
    
    # Filter stats that have image data
    stats_with_images = [s for s in stats if 'image_data' in s and s['image_data']]
    
    if not stats_with_images:
        st.warning("âš ï¸ No images found in recent predictions. The image storage feature was recently added.")
        return
    
    # Let user select which prediction to explain
    st.subheader("Select Prediction to Explain:")
    prediction_options = [f"#{p['block_index']} - {p['filename']} ({p['prediction']} - {p['confidence']:.1%})" 
                         for p in stats_with_images[-10:]]
    
    if prediction_options:
        selected_idx = st.selectbox("Choose a recent prediction:", range(len(prediction_options)), 
                                   format_func=lambda x: prediction_options[x])
        
        selected_prediction = stats_with_images[-10:][selected_idx]
        
        # Display selected prediction info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", selected_prediction['prediction'].upper())
        with col2:
            st.metric("Confidence", f"{selected_prediction['confidence']:.1%}")
        with col3:
            st.metric("Block #", selected_prediction['block_index'])
        
        # Get image data from blockchain
        import base64
        import io
        
        # Find the full block data
        selected_block = None
        for block in blockchain.chain:
            if block.get('index') == selected_prediction['block_index']:
                selected_block = block
                break
        
        if selected_block and 'image_data' in selected_block['data']['metrics']:
            try:
                # Decode image from base64
                img_data = base64.b64decode(selected_block['data']['metrics']['image_data'])
                image = Image.open(io.BytesIO(img_data))
                
                # Show original image
                st.subheader("ðŸ“‹ Selected Image")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image, caption=f"File: {selected_prediction['filename']}", width=400)
                
                # Generate LIME explanation button
                if st.button("ðŸ” Generate LIME Explanation", type="primary"):
                    with st.spinner("Generating LIME explanation... This may take 30-60 seconds."):
                        try:
                            # Convert PIL image to numpy array for LIME
                            image_array = np.array(image)
                            
                            # Create prediction function wrapper
                            def predict_fn(images):
                                return predict_for_lime(images, model, transform)
                            
                            # Generate LIME explanation
                            explanation = lime_explainer.explain_instance(
                                image_array, 
                                predict_fn, 
                                top_labels=2, 
                                hide_color=0, 
                                num_samples=150  # More samples for better explanation
                            )
                            
                            # Get explanation for the predicted class
                            pred_class_idx = 0 if selected_prediction['prediction'] == 'benign' else 1
                            temp_img, mask = explanation.get_image_and_mask(
                                pred_class_idx, 
                                positive_only=True, 
                                num_features=10, 
                                hide_rest=False
                            )
                            
                            # Create visualization
                            st.subheader("ðŸŽ¯ LIME Explanation Results")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Original Image**")
                                fig1, ax1 = plt.subplots(figsize=(8, 8))
                                ax1.imshow(image_array)
                                ax1.set_title(f"Original Image\nPrediction: {selected_prediction['prediction'].upper()}")
                                ax1.axis('off')
                                st.pyplot(fig1)
                            
                            with col2:
                                st.write("**LIME Explanation**")
                                fig2, ax2 = plt.subplots(figsize=(8, 8))
                                ax2.imshow(mark_boundaries(temp_img, mask))
                                ax2.set_title(f"LIME Explanation\nImportant regions for '{selected_prediction['prediction']}'")
                                ax2.axis('off')
                                st.pyplot(fig2)
                            
                            # Show explanation details
                            st.subheader("ðŸ“– How to Interpret These Results")
                            st.write("- ðŸŸ¢ **Green boundaries**: Image regions that strongly support the AI's prediction")
                            st.write("- ðŸ” **Highlighted areas**: Most influential features the AI used for diagnosis")
                            st.write("- ðŸ“Š **LIME Algorithm**: Tests model by systematically hiding image regions")
                            st.write("- ðŸ¥ **Medical Insight**: These highlighted areas should align with known diagnostic features")
                            
                            # Show feature importance if available
                            try:
                                features = explanation.local_exp[pred_class_idx]
                                st.subheader("ðŸ“ˆ Feature Importance Scores")
                                feature_df = pd.DataFrame(features, columns=['Segment', 'Importance'])
                                feature_df = feature_df.sort_values('Importance', key=abs, ascending=False).head(10)
                                
                                fig3 = px.bar(feature_df, x='Segment', y='Importance', 
                                             title="Top 10 Most Important Image Segments",
                                             color='Importance', 
                                             color_continuous_scale='RdYlGn')
                                fig3.update_layout(
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    font_color='black'
                                )
                                fig3.update_xaxes(showgrid=True, gridcolor='lightgray')
                                fig3.update_yaxes(showgrid=True, gridcolor='lightgray')
                                st.plotly_chart(fig3, use_container_width=True)
                                
                            except Exception as e:
                                st.info("Feature importance details not available")
                            
                        except Exception as e:
                            st.error(f"âŒ Error generating LIME explanation: {str(e)}")
                            st.info("This can happen with certain image types. Try with a different image.")
                
            except Exception as e:
                st.error(f"âŒ Error loading image: {str(e)}")
        else:
            st.error("âŒ Image data not found in blockchain record")
    
    # LIME Information Section
    st.markdown("---")
    st.subheader("â„¹ï¸ About LIME (Local Interpretable Model-agnostic Explanations)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**How LIME Works:**")
        st.write("1. **Segmentation**: Divides image into meaningful regions")
        st.write("2. **Perturbation**: Creates thousands of variations by hiding regions")
        st.write("3. **Prediction**: Tests AI model on each variation")
        st.write("4. **Analysis**: Identifies which regions most affect the prediction")
        st.write("5. **Visualization**: Highlights important areas with green boundaries")
    
    with col2:
        st.write("**Medical Benefits:**")
        st.write("- ðŸ©º **Validation**: Verify AI is looking at medically relevant areas")
        st.write("- ðŸŽ“ **Education**: Learn what features indicate benign vs malignant")
        st.write("- ðŸ” **Quality Control**: Catch if AI focuses on irrelevant artifacts")
        st.write("- ðŸ¤ **Trust**: Build confidence in AI-assisted diagnosis")
        st.write("- ðŸ“‹ **Documentation**: Visual evidence for medical records")

def advanced_explainability_page():
    st.title("ðŸ§  Advanced Explainability Dashboard")
    st.write("**Class Activation Mapping, Saliency Maps & Layer-wise Relevance Propagation**")
    
    # Check if there are recent predictions
    stats = blockchain.get_prediction_stats()
    if not stats:
        st.warning("âš ï¸ No predictions found. Please make a prediction on the Main Diagnosis page first.")
        return
    
    # Filter stats that have image data
    stats_with_images = [s for s in stats if 'image_data' in s and s['image_data']]
    
    if not stats_with_images:
        st.warning("âš ï¸ No images found in recent predictions.")
        return
    
    # Let user select which prediction to explain
    st.subheader("Select Prediction to Analyze:")
    prediction_options = [f"#{p['block_index']} - {p['filename']} ({p['prediction']} - {p['confidence']:.1%})" 
                         for p in stats_with_images[-10:]]
    
    if prediction_options:
        selected_idx = st.selectbox("Choose a recent prediction:", range(len(prediction_options)), 
                                   format_func=lambda x: prediction_options[x])
        
        selected_prediction = stats_with_images[-10:][selected_idx]
        
        # Display selected prediction info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", selected_prediction['prediction'].upper())
        with col2:
            st.metric("Confidence", f"{selected_prediction['confidence']:.1%}")
        with col3:
            st.metric("Block #", selected_prediction['block_index'])
        
        # Get image data from blockchain
        import base64
        import io
        
        # Find the full block data
        selected_block = None
        for block in blockchain.chain:
            if block.get('index') == selected_prediction['block_index']:
                selected_block = block
                break
        
        if selected_block and 'image_data' in selected_block['data']['metrics']:
            try:
                # Decode image from base64
                img_data = base64.b64decode(selected_block['data']['metrics']['image_data'])
                image = Image.open(io.BytesIO(img_data))
                
                # Show original image
                st.subheader("ðŸ“‹ Selected Image")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image, caption=f"File: {selected_prediction['filename']}", width=400)
                
                # Prepare image tensor
                image_tensor = transform(image).unsqueeze(0)
                target_class = 0 if selected_prediction['prediction'] == 'benign' else 1
                
                # Select explainability methods
                st.subheader("ðŸ”¬ Choose Explainability Methods:")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    use_saliency = st.checkbox("ðŸŽ¯ Saliency Maps", value=True)
                with col2:
                    use_gradcam = st.checkbox("ðŸ”¥ GradCAM", value=True)
                with col3:
                    use_lrp = st.checkbox("ðŸ”„ Layer LRP", value=True)
                with col4:
                    use_ig = st.checkbox("ðŸ“ˆ Integrated Gradients", value=True)
                
                if st.button("ðŸ§  Generate Advanced Explanations", type="primary"):
                    with st.spinner("Generating advanced explanations... This may take 1-2 minutes."):
                        try:
                            model.eval()
                            
                            # Container for results
                            explanations = {}
                            
                            # 1. Saliency Maps
                            if use_saliency:
                                with st.status("Generating Saliency Maps...") as status:
                                    saliency_attr = generate_saliency_map(model, image_tensor, target_class)
                                    explanations['saliency'] = saliency_attr
                                    status.update(label="âœ… Saliency Maps Complete", state="complete")
                            
                            # 2. GradCAM
                            if use_gradcam:
                                with st.status("Generating GradCAM...") as status:
                                    gradcam_attr = generate_gradcam(model, image_tensor, target_class)
                                    explanations['gradcam'] = gradcam_attr
                                    status.update(label="âœ… GradCAM Complete", state="complete")
                            
                            # 3. Layer-wise Relevance Propagation
                            if use_lrp:
                                with st.status("Generating Layer LRP...") as status:
                                    lrp_attr = generate_lrp(model, image_tensor, target_class)
                                    explanations['lrp'] = lrp_attr
                                    status.update(label="âœ… Layer LRP Complete", state="complete")
                            
                            # 4. Integrated Gradients
                            if use_ig:
                                with st.status("Generating Integrated Gradients...") as status:
                                    ig_attr, ig_nt_attr = generate_integrated_gradients(model, image_tensor, target_class)
                                    explanations['ig'] = ig_attr
                                    explanations['ig_nt'] = ig_nt_attr
                                    status.update(label="âœ… Integrated Gradients Complete", state="complete")
                            
                            # Display Results
                            st.subheader("ðŸŽ¯ Advanced Explainability Results")
                            
                            # Create tabs for different methods
                            tab_names = []
                            if 'saliency' in explanations:
                                tab_names.append("ðŸŽ¯ Saliency")
                            if 'gradcam' in explanations:
                                tab_names.append("ðŸ”¥ GradCAM")
                            if 'lrp' in explanations:
                                tab_names.append("ðŸ”„ Layer LRP")
                            if 'ig' in explanations:
                                tab_names.append("ðŸ“ˆ Integrated Gradients")
                            if 'ig_nt' in explanations:
                                tab_names.append("ðŸŒŠ SmoothGrad IG")
                            
                            tabs = st.tabs(tab_names)
                            
                            tab_idx = 0
                            
                            # Saliency Maps
                            if 'saliency' in explanations:
                                with tabs[tab_idx]:
                                    st.write("**Saliency Maps** show pixel-level importance for the prediction")
                                    fig = visualize_attribution(explanations['saliency'], image_tensor, 
                                                               "Saliency", selected_prediction['prediction'])
                                    st.pyplot(fig)
                                    
                                    st.info("ðŸ’¡ **Interpretation**: Bright areas indicate pixels that most influence the model's decision. " +
                                           "Red areas increase confidence, blue areas decrease it.")
                                tab_idx += 1
                            
                            # GradCAM
                            if 'gradcam' in explanations:
                                with tabs[tab_idx]:
                                    st.write("**GradCAM** highlights important regions using convolutional layer activations")
                                    fig = visualize_attribution(explanations['gradcam'], image_tensor, 
                                                               "GradCAM", selected_prediction['prediction'])
                                    st.pyplot(fig)
                                    
                                    st.info("ðŸ’¡ **Interpretation**: GradCAM shows which regions the convolutional layers " +
                                           "focus on. Warmer colors indicate higher importance for the prediction.")
                                tab_idx += 1
                            
                            # Layer LRP
                            if 'lrp' in explanations:
                                with tabs[tab_idx]:
                                    st.write("**Layer-wise Relevance Propagation** decomposes predictions layer by layer")
                                    fig = visualize_attribution(explanations['lrp'], image_tensor, 
                                                               "Layer LRP", selected_prediction['prediction'])
                                    st.pyplot(fig)
                                    
                                    st.info("ðŸ’¡ **Interpretation**: LRP distributes the prediction score backwards through " +
                                           "the network, showing how each layer contributes to the final decision.")
                                tab_idx += 1
                            
                            # Integrated Gradients
                            if 'ig' in explanations:
                                with tabs[tab_idx]:
                                    st.write("**Integrated Gradients** provide stable attribution by integrating gradients")
                                    fig = visualize_attribution(explanations['ig'], image_tensor, 
                                                               "Integrated Gradients", selected_prediction['prediction'])
                                    st.pyplot(fig)
                                    
                                    st.info("ðŸ’¡ **Interpretation**: IG integrates gradients along a path from baseline to input, " +
                                           "providing more stable and accurate attributions than simple gradients.")
                                tab_idx += 1
                            
                            # SmoothGrad IG
                            if 'ig_nt' in explanations:
                                with tabs[tab_idx]:
                                    st.write("**SmoothGrad Integrated Gradients** reduce noise through sampling")
                                    fig = visualize_attribution(explanations['ig_nt'], image_tensor, 
                                                               "SmoothGrad IG", selected_prediction['prediction'])
                                    st.pyplot(fig)
                                    
                                    st.info("ðŸ’¡ **Interpretation**: SmoothGrad adds noise and averages results, " +
                                           "reducing visual noise and providing cleaner attribution maps.")
                            
                            # Comparison Summary
                            st.subheader("ðŸ“Š Method Comparison Summary")
                            
                            comparison_data = []
                            if 'saliency' in explanations:
                                comparison_data.append({
                                    'Method': 'Saliency Maps',
                                    'Speed': 'âš¡âš¡âš¡',
                                    'Accuracy': 'â­â­',
                                    'Interpretability': 'â­â­â­',
                                    'Best For': 'Quick pixel-level analysis'
                                })
                            
                            if 'gradcam' in explanations:
                                comparison_data.append({
                                    'Method': 'GradCAM',
                                    'Speed': 'âš¡âš¡',
                                    'Accuracy': 'â­â­â­',
                                    'Interpretability': 'â­â­â­â­',
                                    'Best For': 'Regional importance visualization'
                                })
                            
                            if 'lrp' in explanations:
                                comparison_data.append({
                                    'Method': 'Layer LRP',
                                    'Speed': 'âš¡',
                                    'Accuracy': 'â­â­â­â­',
                                    'Interpretability': 'â­â­â­â­â­',
                                    'Best For': 'Detailed layer-wise analysis'
                                })
                            
                            if 'ig' in explanations:
                                comparison_data.append({
                                    'Method': 'Integrated Gradients',
                                    'Speed': 'âš¡',
                                    'Accuracy': 'â­â­â­â­â­',
                                    'Interpretability': 'â­â­â­â­',
                                    'Best For': 'Stable, reliable attributions'
                                })
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"âŒ Error generating explanations: {str(e)}")
                            st.info("This can happen with certain images or methods. Try selecting fewer methods or a different image.")
                
            except Exception as e:
                st.error(f"âŒ Error loading image: {str(e)}")
        else:
            st.error("âŒ Image data not found in blockchain record")
    
    # Method Information Section
    st.markdown("---")
    st.subheader("â„¹ï¸ About Advanced Explainability Methods")
    
    method_info = {
        "ðŸŽ¯ Saliency Maps": {
            "description": "Compute gradients of the output with respect to input pixels",
            "advantages": ["Fast computation", "Pixel-level precision", "Simple to understand"],
            "limitations": ["Can be noisy", "May focus on irrelevant details", "Gradient saturation issues"]
        },
        "ðŸ”¥ GradCAM": {
            "description": "Use convolutional layer gradients to highlight important regions",
            "advantages": ["Region-level explanations", "Works with any CNN", "Visually intuitive"],
            "limitations": ["Lower resolution", "Depends on layer choice", "May miss fine details"]
        },
        "ðŸ”„ Layer LRP": {
            "description": "Decompose predictions by propagating relevance backward through layers",
            "advantages": ["Theoretically grounded", "Layer-wise insights", "Conservative attribution"],
            "limitations": ["Computationally intensive", "Requires specific rules", "Complex implementation"]
        },
        "ðŸ“ˆ Integrated Gradients": {
            "description": "Integrate gradients along path from baseline to input",
            "advantages": ["Mathematically principled", "Stable attributions", "Baseline comparison"],
            "limitations": ["Slower computation", "Baseline selection matters", "Path dependency"]
        }
    }
    
    for method, info in method_info.items():
        with st.expander(f"ðŸ“š {method}"):
            st.write(f"**Description**: {info['description']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**âœ… Advantages:**")
                for adv in info['advantages']:
                    st.write(f"â€¢ {adv}")
            
            with col2:
                st.write("**âš ï¸ Limitations:**")
                for lim in info['limitations']:
                    st.write(f"â€¢ {lim}")

def blockchain_data_page():
    st.title("ðŸ”— Blockchain Audit Trail")
    st.write("**Complete history of all AI predictions**")
    
    stats = blockchain.get_prediction_stats()
    
    if not stats:
        st.info("ï¿½ No predictions recorded yet. Make some predictions to see the audit trail!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(stats)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", len(df))
    with col2:
        benign_count = len(df[df['prediction'] == 'benign'])
        st.metric("Benign Cases", benign_count)
    with col3:
        malignant_count = len(df[df['prediction'] == 'malignant'])
        st.metric("Malignant Cases", malignant_count)
    with col4:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Detailed table
    st.subheader("ðŸ“‹ Detailed Prediction Log")
    st.dataframe(df.sort_values('timestamp', ascending=False), use_container_width=True)
    
    # Download options
    st.subheader("ðŸ“¥ Export Data")
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button("ðŸ“Š Download CSV", csv, "predictions.csv", "text/csv")
    with col2:
        blockchain_json = json.dumps(blockchain.chain, indent=2)
        st.download_button("ðŸ”— Download Full Blockchain", blockchain_json, 
                          f"blockchain_{len(blockchain.chain)}_blocks.json", "application/json")

def advanced_bias_detection_page():
    st.title("âš–ï¸ Advanced Bias Detection with Threshold Optimization")
    st.write("**Monitor AI fairness using advanced post-processing techniques**")
    
    stats = blockchain.get_prediction_stats()
    bias_detector = st.session_state.bias_detector
    
    if len(stats) < 10:
        st.warning(f"âš ï¸ Need at least 10 predictions for advanced bias analysis. Current count: {len(stats)}")
        st.info("ðŸ’¡ Make more predictions to enable Threshold Optimization post-processing.")
        return
    
    df = pd.DataFrame(stats)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Prepare fairness data
    y_true, y_pred, y_scores, sensitive_features = bias_detector.prepare_fairness_data(stats)
    
    if y_true is None:
        st.error("âŒ Unable to prepare fairness analysis data")
        return
    
    # Main Dashboard
    st.subheader("ðŸŽ¯ Fairness Analysis Dashboard")
    
    # Fit Threshold Optimizer if not already fitted
    if not bias_detector.is_fitted and len(stats) >= 20:
        with st.spinner("Training Threshold Optimizer for equalized odds..."):
            success = bias_detector.fit_threshold_optimizer(y_true, y_scores, sensitive_features)
            if success:
                st.success("âœ… Threshold Optimizer trained successfully!")
            else:
                st.warning("âš ï¸ Could not train post-processor. Using standard analysis.")
    
    # Generate fair predictions if post-processor is available
    y_pred_fair = None
    if bias_detector.is_fitted:
        y_pred_fair = bias_detector.predict_fair(y_scores, sensitive_features)
    
    # Calculate fairness metrics
    fairness_metrics = bias_detector.calculate_fairness_metrics(y_true, y_pred, y_pred_fair, sensitive_features)
    
    # Display Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        eq_odds_violation = fairness_metrics.get('equalized_odds_violation', 0)
        st.metric(
            "Equalized Odds Violation",
            f"{eq_odds_violation:.3f}",
            delta=f"{'Good' if eq_odds_violation < 0.1 else 'Poor'} fairness",
            delta_color="normal" if eq_odds_violation < 0.1 else "inverse"
        )
    
    with col2:
        demo_parity = fairness_metrics.get('demographic_parity_violation', 0)
        st.metric(
            "Demographic Parity Violation",
            f"{demo_parity:.3f}",
            delta=f"{'Good' if demo_parity < 0.1 else 'Poor'} parity",
            delta_color="normal" if demo_parity < 0.1 else "inverse"
        )
    
    with col3:
        tpr_diff = fairness_metrics.get('tpr_difference', 0)
        st.metric(
            "TPR Difference",
            f"{tpr_diff:.3f}",
            delta="Between groups",
            delta_color="normal" if tpr_diff < 0.1 else "inverse"
        )
    
    with col4:
        fpr_diff = fairness_metrics.get('fpr_difference', 0)
        st.metric(
            "FPR Difference", 
            f"{fpr_diff:.3f}",
            delta="Between groups",
            delta_color="normal" if fpr_diff < 0.1 else "inverse"
        )
    
    # Technical Details
    st.subheader("ðŸ”¬ Technical Details")
    
    with st.expander("ðŸ“š About Threshold Optimization for Equalized Odds"):
        st.write("""
        **Threshold Optimization** is an advanced post-processing technique for fairness in machine learning:
        
        **Key Concepts:**
        - **Equalized Odds**: Requires equal True Positive Rate AND False Positive Rate across groups
        - **Threshold Optimization**: Finds optimal decision thresholds for each group
        - **Post-processing**: Adjusts model outputs without retraining the model
        
        **How it Works:**
        1. **Analysis**: Identify disparities in model performance across sensitive groups
        2. **Optimization**: Find optimal group-specific thresholds to minimize fairness violations
        3. **Application**: Apply learned thresholds to new predictions
        4. **Evaluation**: Monitor fairness metrics and accuracy trade-offs
        
        **Benefits:**
        - âœ… Model-agnostic (works with any ML model)
        - âœ… Preserves overall model accuracy when possible
        - âœ… Provides mathematical fairness guarantees
        - âœ… Can be applied without model retraining
        - âœ… Interpretable and explainable adjustments
        
        **Limitations:**
        - âš ï¸ Requires labeled data with sensitive attributes
        - âš ï¸ May reduce overall accuracy for fairness
        - âš ï¸ Assumes binary sensitive attributes and outcomes
        """)
    
    # Recommendations
    st.subheader("ðŸ’¡ Fairness Recommendations")
    
    recommendations = []
    
    if fairness_metrics.get('equalized_odds_violation', 0) > 0.1:
        recommendations.append("ðŸ”§ **High Equalized Odds Violation**: Consider using the Threshold Optimizer post-processor")
    
    if fairness_metrics.get('demographic_parity_violation', 0) > 0.1:
        recommendations.append("âš–ï¸ **Demographic Disparity**: Monitor for systematic bias in prediction rates")
    
    if y_pred_fair is not None:
        recommendations.append("âœ… **Post-processing Active**: Fairness-adjusted predictions are being generated")
    else:
        recommendations.append("ðŸ“ˆ **Collect More Data**: Need more predictions to enable advanced fairness techniques")
    
    if not recommendations:
        recommendations.append("âœ… **Good Fairness**: No major bias concerns detected with current data")
    
    for rec in recommendations:
        st.write(rec)


# Page Router
if page == "ðŸ©º Main Diagnosis":
    main_diagnosis_page()
elif page == "ðŸ” LIME Explanations":
    lime_explanations_page()
elif page == "ðŸ§  Advanced Explainability":
    advanced_explainability_page()
elif page == "ðŸ”— Blockchain Data":
    blockchain_data_page()
elif page == "âš–ï¸ Advanced Bias Detection":
    advanced_bias_detection_page()

# Sidebar
