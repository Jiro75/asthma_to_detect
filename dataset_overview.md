# Asthma Detection Dataset Overview

## 📊 Dataset Dimensions
* **Total Samples (Rows):** `1,211`
* **Total Features (Columns):** `113`
* **Numeric Columns:** `102` (Includes audio metrics like MFCCs, ZCR, and clinical metrics like Age, BMI, Heart Rate)
* **Categorical/Text Columns:** `11` (Includes Gender, Smoking Status, Severity, Diagnosis, etc.)
* **Missing Values:** `0` (All missing values have been mapped/cleaned during dataset generation)

---

## 🎯 Target Columns & Distributions

The dataset gives you several options for what you want to predict (your target variable):

### 1. Diagnosis (Binary Target)
Predicts whether the patient has an asthma/bronchial condition or not.
* **Negative** (COPD, Pneumonia, Healthy): `819`
* **Positive** (Asthma, Bronchial): `392`

### 2. Condition (Multi-class Target)
Specific condition prediction.
* **COPD:** `401`
* **Asthma:** `288`
* **Pneumonia:** `285`
* **Healthy:** `133`
* **Bronchial:** `104`

### 3. Severity (Multi-class/Ordinal Target)
How severe is the patient's condition?
* **Moderate:** `468`
* **Severe:** `405`
* **Mild:** `205`
*(Healthy patients don't have a severity listed, making up the remainder)*

---

## 📈 Feature Correlation Analysis

We analyzed the numeric features to see which ones are most strongly correlated with a **Positive Diagnosis** (Asthma/Bronchial). 

> **Tip:** A correlation closer to **1.0** means that as the feature value increases, the likelihood of a positive diagnosis strongly increases. A correlation closer to **-1.0** means that as the feature value increases, the likelihood of a positive diagnosis strongly decreases.

### Top Positive Correlations (Predictive of Asthma/Bronchial)
These features have the strongest positive linear relationship with an Asthma/Bronchial diagnosis:

1. **IgE_Level:** `+0.6308` (Highly predictive; IgE antibodies are tied to allergic asthma)
2. **Eosinophil_Count:** `+0.5772` (Strongly predictive; eosinophils indicate allergic inflammation)
3. **Wheezing_Frequency:** `+0.5238` (Classic clinical symptom of asthma/bronchial constriction)
4. **Allergy:** `+0.4597` (Asthma is frequently allergy-induced)
5. **Spectral_Contrast_2_mean:** `+0.3892` (Audio feature; specific frequency bands are prominent during wheezing)
6. **Spectral_Contrast_5_mean:** `+0.3202`
7. **Family_History:** `+0.3067`
8. **Spectral_Contrast_4_mean:** `+0.3047`
9. **Exercise_Induced_Symptoms:** `+0.3023`
10. **Spectral_Contrast_6_mean:** `+0.2704`

### Top Negative Correlations (Predictive of Other Conditions/Healthy)
These features have the strongest inverse relationship with Asthma/Bronchial (meaning higher values of these features usually point to COPD, Pneumonia, or Healthy lungs):

1. **Spectral_Contrast_7_mean:** `-0.4049` (Audio feature; high-frequency noise bands might be less prominent compared to COPD crackles)
2. **Age:** `-0.3520` (COPD patients tend to be significantly older than asthma patients on average)
3. **Spectral_Bandwidth_mean:** `-0.3377` (Audio feature; sounds are less "broadband/noisy" than pneumonia crackles)
4. **MFCC_1_mean:** `-0.3027` (Audio feature related to overall signal power and shape)
5. **Chroma_8_mean:** `-0.3015`
6. **RMS_Energy_mean:** `-0.2996` (Asthma/Bronchial recordings might have slightly lower overall acoustic energy on average compared to severe coughing fits in other diseases)
7. **RMS_Energy_std:** `-0.2986`
8. **Chroma_9_mean:** `-0.2643`
9. **Chroma_7_mean:** `-0.2554`
10. **Tonnetz_3_mean:** `-0.2455`

---

## 🩺 Example Clinical Features Summary
A quick look at the statistical spread of some key clinical features across all 1,211 patients:

| Feature | Average (Mean) | Min | Max | Standard Deviation |
| :--- | :--- | :--- | :--- | :--- |
| **Age** | 48.48 years | 5.0 | 90.0 | ± 17.34 |
| **BMI** | 25.34 | 15.0 | 42.0 | ± 4.85 |
| **Heart Rate** | 85.49 bpm | 55.0 | 135.0 | ± 14.60 |
| **Oxygen Saturation** | 91.46% | 78.7% | 100.0% | ± 4.27% |
| **Respiratory Rate** | 24.09 breaths/min | 12.0 | 45.0 | ± 6.06 |

---

## 🔊 Example Audio Features Summary
A look at a few of the 81 audio features extracted from the `.wav` files:

| Feature | Average (Mean) | Min | Max |
| :--- | :--- | :--- | :--- |
| **MFCC_1_mean** | -428.62 | -694.88 | -204.61 |
| **Spectral_Centroid_mean** | 208.17 | 24.36 | 1273.51 |
| **Tempo** | 51.90 bpm | 0.00 | 287.11 |
| **ZCR_mean** (Zero Crossing Rate) | ~ 0.00 | 0.00 | 0.04 |

---

## ⚙️ Data Extraction & Generation Methodology

The dataset was constructed through a combined process of audio feature extraction and medically-informed synthetic data generation. This process ensures a rich, multi-modal dataset suitable for machine learning.

### 1. Audio Feature Extraction
Audio features are extracted directly from the raw `.wav` recordings using the `librosa` Python library. Each audio file is standardized to a 6.0-second duration (padding with silence if shorter). We then compute various acoustic properties, including:
* **MFCCs & Deltas:** To capture the timbral shape and dynamic changes over time.
* **Spectral Features:** Centroid, bandwidth, rolloff, contrast, and flatness to quantify the "brightness", "noisiness", and frequency distribution of the lung sounds.
* **Energy & Rhythm:** Root Mean Square (RMS) energy for loudness, Zero Crossing Rate (ZCR) for noise, and Tempo.
* **Harmonic Content:** Chroma features and Tonnetz to map tonal properties.

### 2. Synthetic Clinical Feature Generation
Because real patient clinical data (like blood pressure, age, or BMI) was not originally paired with the open-source audio recordings, we synthesized these values to create a comprehensive tabular dataset. 
* We defined **Clinical Profiles** for each of the 5 conditions (Asthma, Bronchial, COPD, Pneumonia, Healthy) based on real-world medical statistics.
* For each patient, numeric features (e.g., Age, BMI, Heart Rate, IgE Levels) are sampled from condition-specific normal distributions (using a mean, standard deviation, and min/max limits).
* Categorical features (e.g., Smoking Status, Gender, Severity) are sampled using predefined probability weights tailored to the specific diagnosis.

This methodology successfully bridges the gap between raw audio classification and comprehensive clinical tabular modeling.

---

## 📋 Feature Dictionary (All 113 Columns)

### Metadata & Target Features
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **Patient_ID** | String | Unique anonymous identifier for the patient. |
| **Audio_File** | String | Name of the source `.wav` audio file. |
| **Diagnosis** | Categorical | Binary ground truth target (`Positive` for Asthma/Bronchial, `Negative` otherwise). |
| **Condition** | Categorical | Specific diagnosed condition (`Asthma`, `Bronchial`, `COPD`, `Healthy`, `Pneumonia`). |
| **Severity** | Categorical | Clinical severity level (`None`, `Mild`, `Moderate`, `Severe`). |

### Clinical Features
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **Age** | Numeric | Patient's age in years. |
| **Gender** | Categorical | Patient's gender (`Male`, `Female`). |
| **BMI** | Numeric | Body Mass Index. |
| **Blood_Pressure_Systolic** | Numeric | Systolic blood pressure (mmHg). |
| **Blood_Pressure_Diastolic** | Numeric | Diastolic blood pressure (mmHg). |
| **Heart_Rate** | Numeric | Heart rate in beats per minute (bpm). |
| **Glucose_Level** | Numeric | Blood glucose level (mg/dL). |
| **Cholesterol** | Numeric | Total cholesterol level (mg/dL). |
| **Oxygen_Saturation** | Numeric | Blood oxygen saturation level (%). |
| **Smoking_Status** | Categorical | Smoking history (`Never`, `Former`, `Current`). |
| **Family_History** | Binary | Indicator if the patient has a family history of respiratory issues (`1`=Yes, `0`=No). |
| **Physical_Activity** | Categorical | Level of physical activity (`Low`, `Moderate`, `High`). |
| **Allergy** | Binary | Indicator of known allergies (`1`=Yes, `0`=No). |
| **Air_Pollution_Index** | Numeric | Local air quality/pollution index exposure level. |
| **Exercise_Induced_Symptoms**| Binary | Indicator if symptoms are triggered by exercise (`1`=Yes, `0`=No). |
| **Respiratory_Rate** | Numeric | Breaths per minute. |
| **Cough_Frequency** | Numeric | Number of coughing episodes per day. |
| **Dust_Exposure_Level** | Categorical | Level of regular exposure to dust (`Low`, `Medium`, `High`). |
| **Peak_Expiratory_Flow** | Numeric | Maximum speed of expiration, indicating airway obstruction (L/min). |
| **FEV1_FVC_Ratio** | Numeric | Ratio of Forced Expiratory Volume in 1 sec to Forced Vital Capacity. |
| **Eosinophil_Count** | Numeric | Eosinophil white blood cell count (cells/µL); indicates allergic inflammation. |
| **IgE_Level** | Numeric | Immunoglobulin E level (IU/mL); indicates allergic response. |
| **Wheezing_Frequency** | Numeric | Number of wheezing episodes per week. |
| **Chest_Tightness** | Binary | Indicator of experiencing chest tightness (`1`=Yes, `0`=No). |
| **Nighttime_Symptoms** | Binary | Indicator of symptoms worsening at night (`1`=Yes, `0`=No). |
| **Medication_Use** | Categorical | Current medication regimen (`None`, `Inhaler_Only`, `Inhaler_and_Oral`, `Steroid`). |
| **Primary_Symptom** | Categorical | The most prominent symptom presented by the patient. |

### Audio Features
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **MFCC_1_mean** to **MFCC_13_mean** | Numeric | Mean Mel-frequency cepstral coefficients (spectral shape of the sound). |
| **MFCC_1_std** to **MFCC_13_std** | Numeric | Standard deviation of the MFCCs over the audio duration. |
| **Delta_MFCC_1_mean** to **13_mean**| Numeric | Mean of the first-order derivatives of MFCCs (captures dynamic changes). |
| **Spectral_Centroid_mean** / **_std** | Numeric | "Center of mass" of the frequency spectrum; indicates the "brightness" of sound. |
| **Spectral_Bandwidth_mean** / **_std**| Numeric | Width of the frequency band; indicates how "broadband" or noisy the sound is. |
| **Spectral_Rolloff_mean** / **_std** | Numeric | Frequency below which a specified percentage of total spectral energy lies. |
| **Spectral_Contrast_1_mean** to **7** | Numeric | Difference in amplitude between peaks and valleys in the frequency spectrum. |
| **Spectral_Flatness_mean** | Numeric | Measure of how noise-like a sound is, as opposed to tone-like. |
| **ZCR_mean** / **ZCR_std** | Numeric | Zero Crossing Rate: rate at which the signal changes sign (high for noisy sounds). |
| **RMS_Energy_mean** / **_std** | Numeric | Root Mean Square energy; a measure of overall loudness/volume. |
| **Chroma_1_mean** to **Chroma_12_mean**| Numeric | Pitch class profile, capturing harmonic and melodic content (12 pitches). |
| **Tonnetz_1_mean** to **Tonnetz_6_mean**| Numeric | Tonal centroid features representing harmonic relationships. |
| **Mel_Spec_mean, _std, _max, _min** | Numeric | Statistics of the Mel-scaled power spectrogram (in decibels). |
| **Tempo** | Numeric | Estimated tempo/rhythm of the sound (in beats per minute). |
| **Duration_sec** | Numeric | Duration of the audio clip in seconds. |
