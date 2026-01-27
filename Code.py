
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plot

temp_DF = pd.read_excel("enter_your_file_path", sheet_name="weight_of_evidence")

customer_DF = temp_DF.drop("Customer_IDs",axis = 1)
final_DF = customer_DF.drop("Churn",axis =1)
target_DF = temp_DF["Churn"]

sample_train,sample_test,target_train,target_test=train_test_split(
                                    final_DF,target_DF,random_state=42,test_size=0.3,stratify=target_DF)

Decision_tree = DecisionTreeClassifier(
                min_samples_leaf=150,
                min_samples_split=30,
                max_depth=4,
                criterion="gini")

Decision_tree.fit(sample_train,target_train)
prediction = Decision_tree.predict(sample_test)

plot.figure(figsize=(24,12))
plot_tree(Decision_tree,
          feature_names=final_DF.columns,
          class_names=["No","Yes"],
          filled=True,
          rounded=True,
          fontsize=10)
plot.title("Decision Tree For Gym Customer Churn",fontsize=10)
plot.show()

print(classification_report(target_test,prediction))

important_feature = pd.DataFrame({
        "Feature_name" : final_DF.columns,
        "Predicted_Values":Decision_tree.feature_importances_}).sort_values(by="Predicted_Values",ascending=False)
important_feature.to_excel("Important_Feature.xlsx",index=False)

confusion_matrix = pd.DataFrame({
            "Customer_IDs":sample_test.index,
            "Actual_Values":target_test.values,
            "Predicted_Values":prediction})
confusion_matrix.to_excel("confusion_matrix.xlsx",index=False)






