# 🚘 AutoValue — Car Price Prediction

**AutoValue** is a machine learning project that predicts the **MSRP of a vehicle** based on its specifications such as horsepower, torque, manufacturer, body type, engine aspiration, drivetrain, and transmission.

The project uses **Linear Regression** for price prediction and provides an interactive **Streamlit web application** where users can enter vehicle specifications and receive an estimated price.

App link - https://9fh5hboqekdxbaoya6ugpv.streamlit.app/

---

## 🌟 Features

* 🚗 Predict vehicle MSRP using machine learning
* 📊 Linear Regression-based prediction model
* 🧹 Data cleaning and preprocessing pipeline
* 🔢 Converts horsepower and torque from text into numerical features
* 🔠 One-hot encoding for categorical variables
* 📈 Model evaluation using:

  * R² Score
  * RMSE
  * MAE
* 🧠 Feature importance analysis using a Decision Tree
* 📊 Interactive feature-importance visualization
* 🖥️ Interactive Streamlit interface
* 💾 Saved trained model using Pickle

---

## 🧠 Project Workflow

```text
Raw Vehicle Dataset
        │
        ▼
Data Cleaning & Preprocessing
        │
        ├── Handle missing values
        ├── Convert Horsepower → numeric
        ├── Convert Torque → numeric
        └── Convert Price → numeric
        │
        ▼
Feature Engineering
        │
        └── One-Hot Encoding
        │
        ▼
Train / Test Split
        │
        ▼
Linear Regression Model
        │
        ├── Model Evaluation
        └── Model Serialization
        │
        ▼
Saved Model
        │
        ▼
Streamlit Application
        │
        ▼
Estimated Vehicle Price
```

---

## 📂 Project Structure

```text
AutoValue/
│
├── app.py
├── Car_Price_Linear_Regression_clean.ipynb
├── car_data.csv
├── linear_model.pkl
├── feature_importance.xlsx
├── data_with_pred.xlsx
├── requirements.txt
└── README.md
```

### File Description

| File                                      | Description                                                                 |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| `app.py`                                  | Streamlit application for interactive price prediction                      |
| `Car_Price_Linear_Regression_clean.ipynb` | Data preprocessing, EDA, feature engineering, model training and evaluation |
| `car_data.csv`                            | Original vehicle dataset                                                    |
| `linear_model.pkl`                        | Trained Linear Regression model                                             |
| `feature_importance.xlsx`                 | Feature importance generated using a Decision Tree                          |
| `data_with_pred.xlsx`                     | Dataset containing model predictions                                        |
| `requirements.txt`                        | Python dependencies                                                         |
| `README.md`                               | Project documentation                                                       |

---

## 📊 Dataset

The dataset contains **1,610 vehicle records** and includes information such as:

* Make
* Model
* Year
* Trim
* MSRP
* Invoice Price
* Used/New Price
* Body Size
* Body Style
* Cylinders
* Engine Aspiration
* Drivetrain
* Transmission
* Horsepower
* Torque
* Highway Fuel Economy

The model does not use every original column. Features such as `Model`, `Year`, `Trim`, `Used/New Price`, and the original text versions of horsepower and torque are removed before training.

The target variable is:

```text
MSRP
```

The original dataset contains MSRP values ranging from approximately **$15,980 to $391,100**.

---

## 🧹 Data Preprocessing

Several preprocessing steps are performed before training.

### 1. Removing unnecessary columns

The following columns are removed because they are either not used by the model or contain too many missing values:

```python
[
    'Invoice Price',
    'Cylinders',
    'Highway Fuel Economy',
    'index',
    'Model',
    'Year',
    'Trim',
    'Used/New Price'
]
```

### 2. Horsepower conversion

The original horsepower values are stored as strings such as:

```text
697 hp @ 6000 rpm
```

They are converted into numerical values:

```text
697
```

The resulting feature is:

```text
Horsepower_no
```

### 3. Torque conversion

Torque values such as:

```text
663 ft-lbs. @ 2750 rpm
```

are converted into:

```text
663
```

The resulting feature is:

```text
Torque_no
```

### 4. Price conversion

MSRP and Used/New Price values are originally stored as strings containing `$` and commas.

For example:

```text
$242,000
```

is converted to:

```text
242000
```

### 5. One-Hot Encoding

Categorical variables are converted into numerical features using `pd.get_dummies()`.

The encoded categories include:

* Make
* Body Size
* Body Style
* Engine Aspiration
* Drivetrain
* Transmission

The final training dataset contains **36 features**.

---

## 🤖 Machine Learning Model

The project uses:

```text
Linear Regression
```

from Scikit-learn.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
```

The dataset is divided into:

```text
80% → Training
20% → Testing
```

with:

```python
random_state = 15
```

---

## 📈 Model Performance

Using the same preprocessing and train/test split defined in the notebook, the model achieves:

| Metric   |   Training |        Testing |
| -------- | ---------: | -------------: |
| R² Score |      0.896 |      **0.920** |
| RMSE     | $17,421.78 | **$16,534.59** |
| MAE      | $10,599.01 | **$11,090.09** |

### What this means

The testing **R² score of approximately 0.92** means that the model explains around **92% of the variance in vehicle MSRP** on the held-out test set.

The model's testing MAE is approximately:

```text
$11,090
```

meaning the average absolute prediction error is around $11K on this dataset.

---

## 🧠 Feature Importance

Feature importance is calculated separately using a **Decision Tree Classifier** with:

```python
criterion='entropy'
max_depth=10
random_state=15
```

The feature importance results are then displayed in the Streamlit application as an interactive Plotly bar chart.

The highest-ranked features in the generated feature-importance file include:

| Feature                        | Importance |
| ------------------------------ | ---------: |
| Horsepower_no                  |     0.2470 |
| Make_Ford                      |     0.1239 |
| Torque_no                      |     0.1226 |
| Engine Aspiration_Turbocharged |     0.0871 |
| Body Size_Large                |     0.0428 |

> **Note:** The feature importance is calculated using a Decision Tree to understand influential variables. The actual price prediction is performed by the Linear Regression model.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application called **AutoValue**.

The application allows users to enter:

### Performance

* Horsepower
* Torque

### Vehicle

* Make
* Body Size
* Body Style

### Powertrain

* Engine Aspiration
* Drivetrain
* Transmission

The supported input categories correspond to the features used during model training.

The application then converts the selected inputs into the same one-hot encoded feature structure expected by the trained model and generates the predicted MSRP.

The prediction is displayed as:

```text
Estimated Vehicle Price
$XX,XXX.XX
```

The app also provides an interactive visualization showing which features influence the model.

---

## 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* NumPy
* Pandas

### Machine Learning

* Scikit-learn
* Linear Regression
* Decision Tree

### Visualization

* Matplotlib
* Seaborn
* Plotly

### Deployment / UI

* Streamlit

### Model Serialization

* Pickle

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔮 Example Prediction

For example, a user can provide:

```text
Horsepower       → 300
Torque           → 400
Make             → BMW
Body Size        → Midsize
Body Style       → Sedan
Engine Aspiration→ Turbocharged
Drivetrain       → AWD
Transmission     → Automatic
```

The application converts these values into the required numerical feature representation and passes them to the trained Linear Regression model.

---

## 🔬 Model Input Representation

The application maintains a fixed feature order that must match the training data.

Numerical features:

```text
Horsepower_no
Torque_no
```

Categorical features are represented using one-hot encoding, for example:

```text
Make_BMW
Body Size_Midsize
Body Style_Sedan
Engine Aspiration_Turbocharged
Drivetrain_AWD
Transmission_automatic
```

All unused categorical features are set to `0`.

This ensures that the input format supplied to the saved model is consistent with the format used during training.

---

## 📁 Generated Outputs

The training notebook generates three important artifacts:

### `linear_model.pkl`

Serialized Linear Regression model used by the Streamlit application.

### `feature_importance.xlsx`

Contains the calculated feature importance scores used by the Streamlit visualization.

### `data_with_pred.xlsx`

Contains the original vehicle data along with:

```text
MSRP Prediction
```

which stores the model's predicted MSRP for each vehicle.

---

## 🚀 Future Improvements

Possible improvements for future versions include:

* [ ] Try more advanced regression models such as Random Forest, XGBoost and Gradient Boosting
* [ ] Add cross-validation
* [ ] Perform hyperparameter tuning
* [ ] Add prediction confidence / uncertainty estimates
* [ ] Include additional vehicle features
* [ ] Improve handling of unseen categorical values
* [ ] Add model comparison and evaluation dashboard
* [ ] Add charts for actual vs predicted prices
* [ ] Deploy the application publicly
* [ ] Add automated model retraining

---

## ⚠️ Disclaimer

AutoValue provides **estimated vehicle prices based on a machine learning model trained on the available dataset**.

The predicted price should not be considered an official market valuation or guaranteed selling price.

Actual vehicle prices may vary depending on factors such as location, condition, mileage, demand, dealer pricing, and market conditions.

---

## 👨‍💻 Author

**Vishwas Sonker**

B.Tech — Artificial Intelligence & Data Science

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub!

