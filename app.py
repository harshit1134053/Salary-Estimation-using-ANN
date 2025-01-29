{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "8c5df1b6-c6ec-40c3-9900-4d27cb812b8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st \n",
    "import numpy as np \n",
    "import tensorflow as tf\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder\n",
    "import pandas as pd \n",
    "import pickle\n",
    "from tensorflow.keras.models import load_model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "28553265-0bb3-462c-9c9d-7c7da2fedca0",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    }
   ],
   "source": [
    "model=tf.keras.models.load_model('model.h5')\n",
    "\n",
    "with open('label_encoder_gender.pkl','rb') as file:\n",
    "    label_encoder_gender = pickle.load(file)\n",
    "with open('onehot_encoder_geography.pkl','rb') as file:\n",
    "    onehot_encoder_geo = pickle.load(file)\n",
    "with open('scaler.pkl','rb') as file:\n",
    "    scaler = pickle.load(file)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a60d1044-ce39-45e2-98d8-78aefefe3a6b",
   "metadata": {},
   "source": [
    "### Streamlit app"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "29959575-1fa0-4375-bce6-8d76e2790ebb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 30ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\mishraha\\AppData\\Local\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:439: UserWarning: X does not have valid feature names, but OneHotEncoder was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "st.title('Customer Churn Prediction')\n",
    "\n",
    "#User Input\n",
    "geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])\n",
    "gender=st.selectbox('Gender',label_encoder_gender.classes_)\n",
    "age=st.slider('Age',18,92)\n",
    "balance=st.number_input('Balance')\n",
    "credit_score=st.number_input('Credit Score')\n",
    "estimated_salary= st.number_input('Estimated Salary')\n",
    "tenure = st.slider('Tenure',0,10)\n",
    "num_of_products = st.slider('Number of Products',1,4)\n",
    "has_cr_card = st.selectbox('Has Credit Card',[0,1])\n",
    "is_active_member = st.selectbox('Is Active Member',[0,1])\n",
    "\n",
    "#Prepare the input data \n",
    "input_data = pd.DataFrame({\n",
    "    'CreditScore':[credit_score],\n",
    "    'Gender':[label_encoder_gender.transform([gender])[0]],\n",
    "    'Age':[age],\n",
    "    'Tenure':[tenure],\n",
    "    'Balance':[balance],\n",
    "    'NumOfProducts':[num_of_products],\n",
    "    'HasCrCard':[has_cr_card],\n",
    "    'IsActiveMember':[is_active_member],\n",
    "    'EstimatedSalary':[estimated_salary]\n",
    "   \n",
    "})\n",
    "\n",
    "#One-hot encode Geography \n",
    "geo_encoded =onehot_encoder_geo.transform([[geography]]).toarray()\n",
    "geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))\n",
    "\n",
    "\n",
    "#Combine one-hot encoded columns with input data \n",
    "input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)\n",
    "\n",
    "#Scale the input data \n",
    "input_data_scaled = scaler.transform(input_data)\n",
    "\n",
    "#Predict churn\n",
    "prediction = model.predict(input_data_scaled)\n",
    "prediction_proba=prediction[0][0]\n",
    "\n",
    "st.write(f'Churn Probability is :{prediction_proba: .2f}')\n",
    "if prediction_proba > 0.5:\n",
    "    st.write('The customer is likely to churn.')\n",
    "else:\n",
    "    st.write('The customer is not likely to churn.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6c1e3a1b-8784-4972-8a13-9bfaa9c25d11",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5bb7a7d-039c-4e20-a1c1-3103e10b87b1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
