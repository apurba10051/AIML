**⚠️ Disclaimer: Proof of Concept (PoC)**
🚧 This project is developed as a Proof of Concept (PoC) and is intended for demonstration and experimentation purposes only.

While the current implementation focuses on showcasing the clinical trial ingestion, enrichment, AI scoring, it is not yet optimized for production-scale workloads.

----
## 🧪 Problem Statement
Clinical trials face several critical challenges:   

⏳ Slow Patient Recruitment: Matching patients to strict eligibility criteria delays trial timelines.    
❌ High Failure Rates: Many trials fail to show efficacy/safety despite high investments.    
📉 Poor Trial Designs: Flawed inclusion criteria or sample sizes undermine statistical validity.    
🔗 Data Silos: Difficulty integrating fragmented data (e.g., EHRs, registries) hinders insights.    
⚠️ Complex Outcome Prediction: Traditional methods struggle to model trial success due to multi-factorial dependencies.    

##  💡 Expected outcome
Our Clinical Trial Prediction System addresses these issues using -     
📊 Machine Learning: Predicts trial outcomes using historical data (design, interventions, demographics).    
🧠 NLP Techniques: Converts unstructured text (e.g., eligibility criteria) into structured, analyzable data.    
📈 Predictive Insights: Supports smarter trial design, resource allocation, and risk mitigation.    
🎯 Enables researchers, sponsors, and clinicians to make data-driven decisions, reducing costs and improving trial success rates.    

---

## 🚀 Overview

Every day, clinical trial data is pulled incrementally from public **Clinical Trial APIs** or from private **CTMS (Clinical Trial Management systems)** via a batch or streaming ETL process or API ingestion. The data is enriched with final status updates and additional insights captured by **CROs (Contract Research Organizations)**, enhancing both the **data completeness** and **clinical relevance**.

An integrated AI/ML layer scores each trial record using modern machine learning and deep neural networks. The enriched, intelligent data is then sent to **Salesforce**, where **care coordinators** use it to match patients based on:

- 🙍‍♂️ Patient **Symptoms**, **Age** and **Gender**
- 📍 Patient **Location**

###  🧑‍⚕️ Use Case Example 
A 65-year-old male patient with chronic back pain located in NYC is inquiring into this fictitious  system to find a suitable matching trial.     

#### Using the platform, the care coordinator:   
🩺 Inputs patient demographics and symptoms.    
🏥 Views recommended clinical trials near the patient’s location.     
🧬 Reviews AI-predicted match scores and detailed CRO enrichment (e.g., dropout rates, adverse events).         
📄 Selects and initiates the enrollment process for the top-ranked trial.      

## 🧪 Clinical Trial Data Enrichment & Recommendation Platform             

This project demonstrates an end-to-end data pipeline that automates **clinical trial ingestion, enrichment, prediction** and delivery to the User Interface.   
While this project is primarily designed for integration with Salesforce  Cloud (where care coordinators interact with enriched clinical trial data), a lightweight UI using Streamlit or Flask is provided for local testing and demonstration purposes.         
This allows users to interact with the AI model and view predictions without requiring a Salesforce setup.      
The Streamlit/Flask interface serves as a functional prototype and can be extended or replaced by a Salesforce UI in production deployments.    

---

## 🔄 Data Flow Architecture

This project follows a modular, end-to-end pipeline designed to ingest raw clinical trial data, process and clean it, engineer relevant features, train an explainable machine learning model, and prepare the system for deployment. The following components and architectural flow describe how each stage connects and contributes to the overall pipeline:

###  🧬 Key Components
        +--------------------+
        | ClinicalTrials.gov |
        |      API Query     |      🌐 --> [API] --> 📦 Data       
        +--------------------+
                  ↓
        +----------------------+
        |  fetch_data()        |     🔄 Dynamic Data Ingestion
        +----------------------+
                  ↓
        +------------------------+
        | parse_study_data()     |   📦 Structured Extraction
        | parse_criteria()       |
        +------------------------+
                  ↓
        +------------------------+
        | Pre-processing         |   ⚙️ Data Cleaning & Preprocessing   
        +------------------------+
                  ↓
        +------------------------+
        | Text Summarization     |  ✂️ Summarize Text
        +------------------------+
                  ↓
        +------------------------+
        | ClinicalBERT Embedding |  🧠 Biomedical Text Vectorization
        +------------------------+
                  ↓
        +------------------------+
        | PCA for Dim. Reduction |   🔽 Feature Compression
        +------------------------+
                  ↓
        +------------------------+
        | Feature Engineering    |  🔧 Feature Fusion (e.g., duration days, enrollment)
        +------------------------+
                  ↓
        +------------------------+
        | Class Balancing (SMOTE)|  ⚖️ Handle Imbalance
        +------------------------+
                  ↓
        +------------------------+
        | Model Training         |  🎯 XGBoost + Optuna Hyperparam Tuning
        +------------------------+
                  ↓
        +------------------------+
        | Model Evaluation       |  📈 Accuracy, F1, AUC, PR-AUC
        +------------------------+
                  ↓
        +------------------------+
        | xAI (SHAP, LIME)       |  🔍 LIME/SHAP + Feature Importance
        +------------------------+
                  ↓
        +------------------------+
        | Model Serialization    |  💾 Save Pipeline with joblib
        +------------------------+
                       ↓
        +------------------------+
        | Model Serving / API    |  🌐 Flask / FastAPI endpoint to serve predictions
        +------------------------+
                  ↓
        +------------------------+
        | UI Integration         |   🖥️ Flask/ Streamlit / Salesforce UI (Care Coordinator View)
        +------------------------+
                  ↓
        +------------------------+
        | Hosting & Deployment   |  ☁️ Cloud hosting (e.g., Heroku, GCP, AWS, Salesforce App)
        +------------------------+

---

## 🛠️ Technologies Used
  **Core programming language**   
    - Python
    - Salesforce UI (Not included here)
  
  **Important Libraries**    
    - Pandas: Data manipulation, analysis.       
    - NumPy: Numerical operations, arrays.      
    - Scikit-learn: Machine learning tools.      
    - Imbalanced-learn: Handles imbalanced data.      
    - PyTorch: Deep learning framework used for models.      
    - Transformers: Library for pre-trained models and NLP tasks.      
    - Matplotlib: Data visualization, creating plots.      
    - Seaborn: Statistical data visualization, enhancing plots.         
  
  **Data Preprocessing & Feature Engineering**     
    - RobustScaler: Scales numerical features for consistency.   
    - SimpleImputer: Fills missing values for data completeness.    
    - Class Balancing : SMOTE    
    - PCA: Reduces data dimensions for improved model performance.    
  
  **Machine Learning & Optimization**   
    - XGBoost: Prediction model, gradient boosting        
    - Optuna: Hyperparameter tuning, automation     
  
  **Natural Language Processing**  
    - Bio-ClinicalBERT: Clinical text embeddings    
    - T5-Small: Text summarization, conciseness   
    
  **Explainability Tools**   
    - SHAP: Interprets model predictions, feature importance.   
    - LIME: Local explanations, individual predictions.     
  
  **Metrics Used**    
    In this multi-class clinical trial status prediction task, we prioritize:   
    - ROC-AUC: For assessing overall discriminative power, robust against imbalance.   
    - F1-Score: For balancing false positives and false negatives in critical predictions.   
    - PR-AUC: For sensitivity to class imbalance and focus on positive prediction quality.   

**Deployment Tools**   
  - Streamlit: Interactive web application.    
  - pyngrok: Secure tunneling, public access.    

## 🚫 Limitations
  - Verified & clean data source     
  - Use ETL batch for large scale data intake from source to ML layer      
  - Normalize the data ans store in a RDBMS layer. (minimum required structure - Trials, Trial Locations,  Trial Co-ordinators, Patient Enrollment)      
  - Encoders for drug and disease information. Use Smile strings      
  - Distance calculation from Patient geo-code to nearest Trial location using google api      
  - (Beyond Salesforce UI) Build the streamlit/ flask UI with extensive data-entry validations for medical thesaurus from UMLS Metathesaurus: Unified Medical Language System (UMLS API)      
  - Fine tune embedding to improve accuracy       

### 📚 References & Acknowledgements

This project is informed and inspired by prior research in clinical trial prediction, natural language processing, and machine learning:      

**🔬 Key Literature**      
- Prediction of Clinical Trials Outcomes Based on Target Choice and Clinical Trial Design with Multi-Modal Artificial Intelligence - Alex Aliper, Roman Kudrin, et al. — DOI:10.1002/cpt.3008      
- Multimodal Clinical Trial Outcome Prediction with Large Language Models - Tianfan Fu, Huaxiu Yao      
- Metrics Reference Table - NIH/NLM - PMC Article Table 6      
- HINT: Hierarchical Interaction Network for Clinical-Trial-Outcome Predictions - ScienceDirect Link      
- https://github.com/futianfan/clinical-trial-outcome-prediction/blob/main/benchmark/README.md
- and **ALL** open sourcers  who helped me to build this knowledge.    


![Attribution Required](https://img.shields.io/badge/attribution-required-blue)

This code is open for academic and non-commercial exploration. Please acknowledge or cite the original author if using or adapting it. Do not present this work as your own without appropriate credit.


                                                                ----  THANK YOU -----
