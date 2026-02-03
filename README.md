# ColorMood Predictor

## Project Overview
ColorMood Predictor is a research-backed machine learning project that predicts the emotional impact of colors as **Good, Medium, or Bad**. Users can input colors via RGB or HEX values, and the model interprets their emotional effect based on well-established color psychology principles. This project demonstrates end-to-end ML workflow including dataset creation, feature engineering, classification, and interactive deployment via Streamlit.

## *Idealogy*
Colors profoundly impact human emotions and decision-making in branding, web design, and user experience. This project bridges psychology and machine learning, enabling designers and businesses to make informed color choices for websites and applications.

## Dataset
- **Type:** Synthetic, research-informed  
- **Rows:** 5,500+  
- **Features:**
  - R, G, B: Red, Green, Blue channel values (0–255)
  - H, S, L: Hue, Saturation, Lightness derived from RGB
  - Brightness: Perceived brightness from RGB
  - Target: Emotional impact class (Good, Medium, Bad)  
  
- **Labeling Justification:** Based on peer-reviewed color psychology research:
  - **Blue/Green hues**   calming/positive → Good  
  - **High lightness / pastels**   generally positive or Good  
  - **Red hues / dark colors**  arousal/negative or Bad  
  - **Grays / low saturation**   neutral or Medium  

**Citations for labeling logic:**  
**:::Ou, L.-C., et al., 2004 : A study of color emotion and preference**

**:::Elliot, A. J., & Maier, M. A., 2012 : Color Psychology: Effects of Perceivable Color on Human Behavior**

**:::Hemphill, L., 1996 —: Adults’ color–emotion associations**

## ML Pipeline
1. **Data Preprocessing**  
   - Handle missing values  
   - Feature extraction: RGB → HSL + Brightness  
2. **Modeling**  
   - Multiclass classification using **Decision Tree Classifier**  
   - Labels: Good / Medium / Bad  
3. **Model Evaluation**  
   - Accuracy, confusion matrix, feature importance  
4. **Deployment**  
   - Interactive Streamlit app  
   - Users input colors via sliders or HEX picker  
   - Outputs predicted emotional impact and live color visualization