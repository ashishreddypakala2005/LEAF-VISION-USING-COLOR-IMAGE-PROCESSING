from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from src.dataset_utils import scan_dataset, list_dataset_images
from src.segmentation import analyze_leaf, make_pipeline_image

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
st.set_page_config(page_title="LeafVision | Disease Segmentation", page_icon="🌿", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#f7f7f1}.block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1500px}
section[data-testid="stSidebar"]{background:#142d25;border-right:1px solid #234638;width:280px!important;min-width:280px!important}
section[data-testid="stSidebar"] *{color:#eaf4ed!important}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"]{color:#52655a!important}
section[data-testid="stSidebar"] [data-testid="stMetricValue"]{color:#214d38!important}
.stApp{color:#26372e}
.stApp [data-testid="stMarkdownContainer"]{color:#26372e}
.stApp [data-testid="stMetricLabel"],.stApp [data-testid="stMetricValue"],.stApp [data-testid="stMetricDelta"]{color:#26372e!important}
.stApp [data-testid="stWidgetLabel"],.stApp [data-testid="stRadio"] label,.stApp [data-testid="stCheckbox"] label,.stApp [data-testid="stSlider"] label{color:#26372e!important}
.stApp [data-testid="stAlert"] *{color:#26372e!important}
.stApp [data-baseweb="tab"]{color:#26372e!important}
.stApp [data-testid="stCaptionContainer"]{color:#59685f}
section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricLabel"],
section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricLabel"] *,
section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"],
section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] *,
section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricDelta"],
section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricDelta"] *{color:#214d38!important;opacity:1!important}
.hero{background:linear-gradient(115deg,#17382d,#147b5a 58%,#3db58b);padding:34px 38px;border-radius:24px;color:white;margin:4px 0 24px;box-shadow:0 12px 35px #17382d20}
.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:700;color:#a8f0c8;margin-bottom:12px}
.hero h1{font-family:'Playfair Display',serif;font-size:clamp(30px,4vw,49px);line-height:1.1;color:#fff;margin:0 0 14px}
.hero p{font-size:15px;line-height:1.8;max-width:780px;color:#e3f5eb;margin:0}
.page-title{font-family:'Playfair Display',serif;font-size:29px;color:#1b3a2d;margin:4px 0}
.page-sub{color:#68796f;font-size:14px;margin-bottom:22px}
.kpi{background:white;border:1px solid #e2e9df;border-radius:16px;padding:17px 19px;min-height:110px}
.kpi .label{font-size:12px;color:#728075;margin-bottom:9px}.kpi .value{font-family:'Playfair Display',serif;font-size:20px;line-height:1.25;overflow-wrap:anywhere;color:#214d38;font-weight:700}.kpi .foot{font-size:11px;color:#849187;margin-top:4px}
.step,.panel{background:white;border:1px solid #e5e9df;border-radius:15px;padding:17px 19px;margin-bottom:14px}
.step{min-height:135px}.step .num{font-size:12px;color:#1d966b;font-weight:700;letter-spacing:1px}.step h4{color:#234635;margin:10px 0 7px;font-size:15px}.step p,.panel p{font-size:13px;line-height:1.7;color:#748078;margin:0}
.panel h3{font-size:17px;color:#214333;margin:0 0 10px}
div.stButton>button[kind="primary"]{background:#15966d;border:0;border-radius:10px;color:#fff;font-weight:700;min-height:44px}
div.stButton>button[kind="primary"] *{color:#fff!important}
div.stButton>button[kind="primary"]:hover{background:#087b57;color:white}
div[data-testid="stMetric"]{background:#fff;border:1px solid #e4e9df;padding:13px 15px;border-radius:14px}
.stTabs [data-baseweb="tab"]{background:#eef2eb;border-radius:9px 9px 0 0;padding:9px 16px}
.small{font-size:12px;color:#718076}hr{border-color:#e4e9df}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def get_dataset_stats(root):
    return scan_dataset(Path(root))

@st.cache_data(ttl=300, show_spinner=False)
def get_class_images(root, class_name):
    return list_dataset_images(Path(root), class_name)

stats = get_dataset_stats(str(DATASET))
with st.sidebar:
    st.markdown("## 🌿 LeafVision")
    st.caption("COLOR IMAGE PROCESSING LAB")
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Overview", "Leaf Analysis", "Segmentation Lab", "Dataset & Model", "Project Methodology"])
    st.markdown("---")
    st.markdown("### LIVE DATASET STATUS")
    st.metric("Images detected", f'{stats["image_count"]:,}')
    st.caption(f'{stats["class_count"]} class folders with images')
    st.markdown("---")
    st.caption("PlantVillage • OpenCV • HSV + LAB • Morphology")

def heading(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)

def choose_image(key):
    mode = st.radio("Image source", ["Upload image", "Choose from PlantVillage"], horizontal=True, key=key+"_source")
    if mode == "Upload image":
        f = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"], key=key+"_upload")
        return (Image.open(f).convert("RGB"), f.name) if f else (None,None)
    classes = [class_name for class_name, _ in stats["classes"] if class_name != "Unsorted (root)"]
    if not classes:
        st.warning("No PlantVillage images found. Put the extracted class folders inside this project's dataset folder.")
        return None,None
    class_name = st.selectbox("Choose a class", classes, key=key+"_class")
    images = get_class_images(str(DATASET), class_name)
    opts = [x["relative"] for x in images]
    chosen = st.selectbox("Select a PlantVillage sample", opts, key=key+"_sample")
    item = images[opts.index(chosen)]
    try: return Image.open(item["path"]).convert("RGB"), item["path"].name
    except Exception as e:
        st.error(f"Could not open image: {e}")
        return None,None

if page == "Overview":
    st.markdown("""<div class="hero"><div class="eyebrow">Academic Computer Vision Project</div><h1>Leaf Disease Segmentation</h1><p>A professional colour-image-processing system that isolates leaf tissue, detects visually abnormal colour regions, estimates candidate affected area, and presents each stage of the segmentation pipeline.</p></div>""", unsafe_allow_html=True)
    heading("Project at a glance","A quick overview of your local dataset and processing approach.")
    cols=st.columns(4)
    items=[("PlantVillage images",f'{stats["image_count"]:,}',"Scanned from local folder"),("Class folders",str(stats["class_count"]),"Detected from folder names"),("Processing","HSV + LAB","Colour-space analysis"),("Segmentation","Colour + Morphology","Classical image processing")]
    for c,(label,value,foot) in zip(cols,items):
        with c: st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="foot">{foot}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    heading("Processing architecture","From an input photograph to a candidate-region mask.")
    steps=[("01","Input image","Choose a PlantVillage sample or upload a leaf photo."),("02","Preprocessing","Prepare image and convert colour spaces."),("03","Colour analysis","Inspect HSV and CIELAB colour cues."),("04","Leaf mask","Estimate leaf pixels and suppress background."),("05","Region segmentation","Highlight yellow/brown candidate regions."),("06","Area estimation","Compare candidate pixels with estimated leaf pixels.")]
    for group in (steps[:3],steps[3:]):
        cols=st.columns(3)
        for c,(n,title,body) in zip(cols,group):
            with c: st.markdown(f'<div class="step"><div class="num">{n}</div><h4>{title}</h4><p>{body}</p></div>',unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
    st.info("The red regions are colour-based candidates, not a confirmed diagnosis. Lighting, shadows and natural leaf ageing can affect the output.")

elif page == "Leaf Analysis":
    heading("Leaf Analysis","Choose a leaf image, inspect colour features and review the candidate-region estimate.")
    img,name=choose_image("analysis")
    if img is not None:
        left,right=st.columns([1.05,1])
        with left:
            st.markdown('<div class="panel"><h3>Selected leaf image</h3></div>',unsafe_allow_html=True)
            st.image(img,caption=name,use_container_width=True)
        with right:
            st.markdown('<div class="panel"><h3>Analysis settings</h3></div>',unsafe_allow_html=True)
            sensitivity=st.slider("Disease detection sensitivity",0.60,1.40,1.00,0.05,key="analysis_sens")
            space=st.selectbox("Colour processing",["HSV + CIELAB","HSV only","CIELAB only"],key="analysis_space")
            run=st.button("Run leaf analysis →",type="primary",use_container_width=True)
        if run:
            st.session_state["analysis_result"]=analyze_leaf(np.array(img),sensitivity,space)
            st.session_state["analysis_name"]=name
        res=st.session_state.get("analysis_result")
        if res and st.session_state.get("analysis_name")==name:
            a,b,c=st.columns(3);a.metric("Estimated leaf pixels",f'{res["leaf_pixels"]:,}');b.metric("Candidate region pixels",f'{res["disease_pixels"]:,}');c.metric("Candidate area estimate",f'{res["affected_pct"]:.2f}%')
            st.image(res["overlay"],caption="Green outline: estimated leaf boundary · Red: candidate colour regions",use_container_width=True)

elif page == "Segmentation Lab":
    heading("Segmentation Lab","Tune the colour analysis and inspect each stage of the segmentation pipeline.")
    img,name=choose_image("lab")
    if img is not None:
        left,right=st.columns([1.05,.95],gap="large")
        with left:
            st.markdown('<div class="panel"><h3>Selected leaf image</h3></div>',unsafe_allow_html=True)
            st.image(img,caption=name,use_container_width=True)
        with right:
            st.markdown('<div class="panel"><h3>Analysis settings</h3></div>',unsafe_allow_html=True)
            sensitivity=st.slider("Disease detection sensitivity",.60,1.40,1.00,.05,key="lab_sens")
            space=st.selectbox("Colour space",["HSV + CIELAB","HSV only","CIELAB only"],key="lab_space")
            minimum=st.slider("Remove regions smaller than (pixels)",10,1500,120,10)
            st.caption("HSV/LAB colour cues are followed by morphological cleanup.")
            run=st.button("Run segmentation analysis →",type="primary",use_container_width=True)
        if run:
            st.session_state["lab_result"]=analyze_leaf(np.array(img),sensitivity,space,minimum)
            st.session_state["lab_name"]=name
        res=st.session_state.get("lab_result")
        if res and st.session_state.get("lab_name")==name:
            st.markdown("---");heading("Segmentation output","Inspect the masks, overlay, colour views and processing pipeline.")
            a,b,c=st.columns(3);a.metric("Leaf area",f'{res["leaf_pixels"]:,} px');b.metric("Candidate region",f'{res["disease_pixels"]:,} px');c.metric("Candidate area",f'{res["affected_pct"]:.2f}%')
            tabs=st.tabs(["Overlay","Leaf mask","Candidate mask","HSV / LAB","Pipeline"])
            with tabs[0]: st.image(res["overlay"],use_container_width=True)
            with tabs[1]: st.image(res["leaf_mask"],use_container_width=True)
            with tabs[2]: st.image(res["disease_mask"],use_container_width=True)
            with tabs[3]:
                x,y=st.columns(2);x.image(res["hsv_view"],caption="HSV colour view",use_container_width=True);y.image(res["lab_view"],caption="CIELAB colour view",use_container_width=True)
            with tabs[4]: st.image(make_pipeline_image(np.array(img),res),use_container_width=True)
            data=pd.DataFrame([{"image":name,"leaf_pixels":res["leaf_pixels"],"candidate_region_pixels":res["disease_pixels"],"candidate_area_percent":round(res["affected_pct"],4),"sensitivity":sensitivity,"colour_space":space}]).to_csv(index=False).encode()
            st.download_button("Download analysis summary (CSV)",data,file_name="leafvision_result.csv",mime="text/csv")

elif page == "Dataset & Model":
    heading("Dataset & Model","Explore the PlantVillage class folders found on your computer.")
    a,b,c=st.columns(3);a.metric("Images detected",f'{stats["image_count"]:,}');b.metric("Class folders",stats["class_count"]);c.metric("Image data size",f'{stats["size_mb"]:.1f} MB')
    if stats["classes"]:
        st.markdown('<div class="panel"><h3>Class-wise image count</h3></div>',unsafe_allow_html=True)
        df=pd.DataFrame(stats["classes"],columns=["Class folder","Image count"])
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("Download dataset report CSV",df.to_csv(index=False).encode(),file_name="plantvillage_class_counts.csv",mime="text/csv")
        with st.expander("Browse sample images"):
            class_name=st.selectbox("Choose class",[x[0] for x in stats["classes"]])
            samples=get_class_images(str(DATASET), class_name)[:12]
            cols=st.columns(4)
            for i,item in enumerate(samples):
                with cols[i%4]: st.image(str(item["path"]),caption=item["path"].name,use_container_width=True)
    else:
        st.warning("No images detected. Extract PlantVillage and put its class folders inside the project's dataset folder.")
        st.code("LeafVision_AI_Project/dataset/ClassName/image.jpg")
    st.info("PlantVillage typically provides image-level class labels, not pixel-level masks for every lesion. The highlighted regions are generated by image-processing thresholds.")

elif page == "Project Methodology":
    heading("Project Methodology","An explainable classical computer-vision workflow.")
    stages=[("1. Image input","Load a PlantVillage sample or user-provided JPG/PNG image."),("2. Colour-space conversion","Convert RGB to HSV and CIELAB to analyse colour characteristics."),("3. Leaf/background mask","Use colour and saturation thresholds to estimate the leaf region."),("4. Candidate-region extraction","Apply configurable yellow/brown colour thresholds in HSV and/or CIELAB."),("5. Morphological cleanup","Opening/closing reduces speckles and fills small gaps; connected components remove tiny regions."),("6. Area estimate and visualization","Count leaf and candidate pixels, calculate a percentage, and show masks and overlays.")]
    for title,body in stages: st.markdown(f'<div class="panel"><h3>{title}</h3><p>{body}</p></div>',unsafe_allow_html=True)
    st.markdown("### Limitations")
    st.warning("This is colour-based candidate segmentation, not a trained deep-learning model or a confirmed diagnosis. For validated IoU/Dice or disease severity, use suitable pixel-level ground-truth annotations.")
    st.markdown("### Run locally")
    st.code("py -m pip install -r requirements.txt\npy -m streamlit run app.py",language="powershell")

st.markdown("---")
st.markdown('<div class="small">LeafVision · Leaf Disease Segmentation using Color Image Processing · Academic prototype</div>',unsafe_allow_html=True)
