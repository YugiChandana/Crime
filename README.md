This project aims to predict whether an arrest will occur based on crime type, year, and state using a machine learning model.
It utilizes a **Random Forest Classifier** to analyze historical crime data.

---

## 🔧 Instructions

1. **Clone the repository**
   git clone https://github.com/YugiChandana/Crime.git
   cd Crime

2. 🚀 Usage
   Train the Model
   To train the Random Forest model using the dataset:
   python train.py

3. Run the App (use Flask):
   python app.py

4. 📊 Dataset Description:
   The dataset used is CRIME_DETAILS_RANDOMIZED.csv. It contains information on crimes committed across different states and years in India.

      Key columns:
   
        STATE/UT: Name of the state or union territory.

        YEAR: Year of the reported crime.

        CRIME HEAD: Category/type of the crime.

        ARRESTED: Whether an arrest was made (Yes/No).

   The data has been randomized for confidentiality and testing purposes.

✅ Results:
The Random Forest model achieved high accuracy on test data.

It was able to classify whether an arrest would occur based on state, year, and crime type.

The model showed robustness and consistency across different data splits.

![image](https://github.com/user-attachments/assets/0261bbea-20ab-49a9-99a0-0ca5a3e8ce98)

🙋‍♀️ Author
Yugi Chandana
B.Tech CSE Student
https://github.com/YugiChandana



