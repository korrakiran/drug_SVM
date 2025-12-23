# Drug Classification

This notebook performs a supervised classification task to predict the type of drug (`Drug`) prescribed to a patient based on medical and demographic attributes.

---

## 1. Data Loading and Initial Inspection
- The dataset `drug.csv` is loaded into a pandas DataFrame (`df`).
- Initial inspection is carried out using:
  - `df.info()` to understand data types and non-null counts.
  - `df.isnull().sum()` to check for missing values.
  - `df.duplicated().sum()` to identify duplicate records.
- Result:
  - No missing values were found.
  - No duplicate rows were present.
- The distribution of the target variable (`Drug`) is examined using value counts to understand class balance.

---

## 2. Data Preprocessing
- All categorical (`object`) columns are converted into numerical form using **Label Encoding**:
  - Encoded columns: `Sex`, `BP`, `Cholesterol`, `Drug`.
- `sklearn.preprocessing.LabelEncoder` is applied column-wise.
- After encoding:
  - All features are numerical.
  - The dataset is suitable for machine learning algorithms.
- `df.info()` confirms successful preprocessing.

---

## 3. Feature Selection and Data Splitting
- Features (`X`): All columns except `Drug`.
- Target (`y`): `Drug`.
- The dataset is split into training and testing sets using:
  - `train_test_split`
  - `train_size = 0.5`
  - `random_state = 0`

---

## 4. Model Building and Evaluation (Support Vector Classifier)
Support Vector Machine (SVC) models are trained using different kernel functions and evaluated using accuracy.

### a. RBF Kernel
- Default RBF kernel and explicitly defined RBF kernel are tested.
- Accuracy achieved: **72.0%**
- Indicates moderate performance on this dataset.

### b. Linear Kernel
- SVC with a linear kernel is trained and evaluated.
- Accuracy achieved: **98.0%**
- Shows excellent performance and strong linear separability in the data.

### c. Polynomial Kernel
- SVC with a polynomial kernel is trained and evaluated.
- Accuracy achieved: **68.0%**
- Performance is lower compared to RBF and Linear kernels.

---

## 5. Conclusion
- The **Linear Kernel SVC** outperforms other kernels with an accuracy of **98.0%**.
- This suggests that the dataset is largely **linearly separable**.
- For this problem, a linear decision boundary is sufficient and most effective for predicting drug types.

---
