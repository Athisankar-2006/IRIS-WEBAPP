import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


# App title and description
st.set_page_config(page_title="🌸 Iris Flower Species Predictor", layout="centered")

st.title("🌸 Iris Flower Species Predictor")
st.markdown("""
Welcome to the Iris Flower Species Prediction App!  
Adjust the sliders in the sidebar to input flower measurements, and the model will predict the species using a Random Forest Classifier.
""")

@st.cache_data
def load_data():
    iris=load_iris()
    df=pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species']=iris.target
    return df,iris.target_names

df,target_name=load_data()

model=RandomForestClassifier()
model.fit(df.iloc[:,:-1],df['species'])


st.sidebar.title("Input features")
sepal_length=st.sidebar.slider("sepal length",float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max()))
sepal_width=st.sidebar.slider("sepal width",float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()))
petal_length=st.sidebar.slider("petal length",float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))
petal_width=st.sidebar.slider("petal width",float(df['petal width (cm)'].min()),float(df['petal width (cm)'].max()))

input_data=[[sepal_length,sepal_width,petal_length,petal_width]]


###prediction

prediction=model.predict(input_data)
prediction_proba=model.predict_proba(input_data)
prediction_species=target_name[prediction[0]]


#display the prediction

st.subheader("🔍prediction Result")
st.success(f"🌼The predicted species is: {prediction_species}")


##Display prediction probabilities

st.subheader("📈 Prediction Probabilities")
proba_df = pd.DataFrame(prediction_proba, columns=target_name)
st.dataframe(proba_df.style.format("{:.2%}"), use_container_width=True)

#show the model info

with st.expander("ℹ️About the Model"):
      st.markdown("""
    - **Model:** Random Forest Classifier  
    - **Dataset:** Iris flower dataset from `sklearn.datasets`  
    - **Features Used:** Sepal length, Sepal width, Petal length, Petal width  
    - **Target Classes:** Setosa, Versicolor, Virginica  
    """)
      


## footer

st.markdown("""---  
Made with ❤️ using [Streamlit](https://streamlit.io/)
""")
