**Iris Flower Classification using Machine Learning**
**Overview**
This project demonstrates how to classify iris flower species (Setosa, Versicolor, Virginica) based on their sepal and petal measurements using a machine learning model. The model is built using the Random Forest Classifier from the scikit-learn library. The dataset used is the well-known Iris dataset, which is available in the scikit-learn library.

**Dataset**
The Iris dataset consists of 150 samples from three species of Iris flowers:
1.Iris Setosa
2.Iris Versicolor
3.Iris Virginica
**Each sample has four features:**
1.Sepal length (cm)
2.Sepal width (cm)
3.Petal length (cm)
4..Petal width (cm)
These features are used to classify the flower into one of the three species.

**Project Workflow**
The project follows these steps:
**Data Loading**: The Iris dataset is loaded using the scikit-learn library.
**Data Preprocessing: **The dataset is split into training and testing sets, and the features are scaled using StandardScaler to improve the performance of the machine learning model.
**Model Training:** A Random Forest Classifier is trained on the training dataset.
**Model Evaluation:** The trained model is evaluated using the test dataset. Performance metrics such as accuracy, confusion matrix, and classification report are generated.
**Visualization:** A confusion matrix is plotted to visualize the model’s performance, and feature importance is shown to highlight which features are most useful in the classification.

**Prerequisites**
To run this project, you need to have the following libraries installed:
1.Python 3.x
2.scikit-learn
3.numpy
4.pandas
5.matplotlib
6.seaborn
**CODE EXPLANATION**
1. Data Loading: The Iris dataset is loaded using load_iris() from the scikit-learn library.
from sklearn.datasets import load_iris
iris = load_iris()

2. Data Splitting: The dataset is split into training and testing sets using train_test_split().
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

3. Model Training: A Random Forest Classifier is trained on the scaled training data.
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

4. Model Evaluation: The model's accuracy and confusion matrix are calculated and printed.
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = model.predict(X_test_scaled)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

5. Visualization: A confusion matrix is plotted using Seaborn.
import seaborn as sns
sns.heatmap(conf_matrix, annot=True)
plt.show()
