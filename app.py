import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# ── Load model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('churn_model_xgb.pkl')

model = load_model()

# ── Header ───────────────────────────────────────────────
st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer data below to predict churn risk.")
st.divider()

# ── Input form ───────────────────────────────────────────
st.subheader("Customer Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 35)
    membership_years = st.slider("Membership Years", 0, 10, 3)
    total_purchases = st.slider("Total Purchases", 0, 200, 50)
    customer_service_calls = st.slider("Customer Service Calls", 0, 20, 3)
    days_since_last_purchase = st.slider("Days Since Last Purchase", 0, 365, 30)
    cart_abandonment_rate = st.slider("Cart Abandonment Rate (%)", 0, 100, 40)
    email_open_rate = st.slider("Email Open Rate (%)", 0, 100, 30)
    returns_rate = st.slider("Returns Rate (%)", 0, 100, 10)
    discount_usage_rate = st.slider("Discount Usage Rate (%)", 0, 100, 30)
    login_frequency = st.slider("Login Frequency", 0, 30, 10)
    session_duration_avg = st.slider("Avg Session Duration (min)", 0, 120, 25)
    pages_per_session = st.slider("Pages Per Session", 0, 50, 8)

with col2:
    lifetime_value = st.number_input("Lifetime Value ($)", 0, 50000, 5000)
    average_order_value = st.number_input("Average Order Value ($)", 0, 5000, 200)
    credit_balance = st.number_input("Credit Balance ($)", 0, 10000, 1000)
    wishlist_items = st.slider("Wishlist Items", 0, 50, 5)
    product_reviews_written = st.slider("Product Reviews Written", 0, 50, 2)
    social_media_engagement = st.slider("Social Media Engagement Score", 0, 100, 30)
    mobile_app_usage = st.slider("Mobile App Usage (sessions/month)", 0, 100, 20)
    payment_method_diversity = st.slider("Payment Method Diversity", 1, 5, 2)
    gender = st.selectbox("Gender", ["Male", "Female"])
    country = st.selectbox("Country", ["USA", "UK", "Canada", "Australia", "Germany"])
    city = st.selectbox("City", ["New York", "London", "Toronto", "Sydney", "Berlin"])
    signup_quarter = st.selectbox("Signup Quarter", ["Q1", "Q2", "Q3", "Q4"])

st.divider()

# ── Predict button ────────────────────────────────────────
if st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary"):

    # Encode categoricals same way as training
    gender_enc = 0 if gender == "Female" else 1
    country_enc = ["Australia", "Canada", "Germany", "UK", "USA"].index(country)
    city_enc = ["Berlin", "London", "New York", "Sydney", "Toronto"].index(city)
    quarter_enc = ["Q1", "Q2", "Q3", "Q4"].index(signup_quarter)

    # Build input dataframe — columns must match training exactly
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender_enc,
        'Country': country_enc,
        'City': city_enc,
        'Membership_Years': membership_years,
        'Login_Frequency': login_frequency,
        'Session_Duration_Avg': session_duration_avg,
        'Pages_Per_Session': pages_per_session,
        'Cart_Abandonment_Rate': cart_abandonment_rate,
        'Wishlist_Items': wishlist_items,
        'Total_Purchases': total_purchases,
        'Average_Order_Value': average_order_value,
        'Days_Since_Last_Purchase': days_since_last_purchase,
        'Discount_Usage_Rate': discount_usage_rate,
        'Returns_Rate': returns_rate,
        'Email_Open_Rate': email_open_rate,
        'Customer_Service_Calls': customer_service_calls,
        'Product_Reviews_Written': product_reviews_written,
        'Social_Media_Engagement_Score': social_media_engagement,
        'Mobile_App_Usage': mobile_app_usage,
        'Payment_Method_Diversity': payment_method_diversity,
        'Lifetime_Value': lifetime_value,
        'Credit_Balance': credit_balance,
        'Signup_Quarter': quarter_enc
    }])

    # Predict
    prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    # ── Result display ────────────────────────────────────
    st.subheader("Prediction Result")

    if prob < 0.4:
        st.success(f"✅ Low Churn Risk — {prob*100:.1f}% probability")
    elif prob < 0.7:
        st.warning(f"⚠️ Medium Churn Risk — {prob*100:.1f}% probability")
    else:
        st.error(f"🚨 High Churn Risk — {prob*100:.1f}% probability")

    st.metric("Churn Probability", f"{prob*100:.1f}%")
    st.progress(float(prob))

    # ── SHAP explanation ──────────────────────────────────
    st.subheader("Why this prediction?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    shap_df = pd.DataFrame({
        'Feature': input_data.columns,
        'SHAP Value': shap_values[0],
        'Input Value': input_data.iloc[0].values
    })

    shap_df['Abs'] = shap_df['SHAP Value'].abs()
    shap_df = shap_df.sort_values('Abs', ascending=False).head(5)
    shap_df['Impact'] = shap_df['SHAP Value'].apply(
        lambda x: '🔴 Increases churn risk' if x > 0 else '🟢 Decreases churn risk'
    )

    st.dataframe(
        shap_df[['Feature', 'Input Value', 'Impact']].reset_index(drop=True),
        use_container_width=True
    )

st.divider()
st.caption("Built by MERIEM AZNAG | XGBoost + SHAP | AUC: 0.929")