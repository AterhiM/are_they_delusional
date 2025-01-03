import streamlit as st
import pandas as pd
import numpy as np
import json
import scipy.stats as stats

def set_page_config():
    st.set_page_config(
        page_title="Dream Partner Probability Calculator",
        page_icon="🔮",
        layout="wide"
    )

def calculate_anthropometric_probability(value, mean, std_dev=None):
    """Calculate probability using normal distribution"""
    if std_dev is None:
        std_dev = mean * 0.1
    z_score = abs(value - mean) / std_dev
    probability = 2 * (1 - stats.norm.cdf(z_score))
    return max(probability, 0.01)

def get_gender_specific_data(percentages_df, target_gender):
    """Extract gender-specific anthropometric data"""
    try:
        height_data = next(item for item in percentages_df 
                         if item['Indicator'] == 'Height')
        weight_data = next(item for item in percentages_df 
                         if item['Indicator'] == 'Weight')
        bmi_data = next(item for item in percentages_df 
                       if item['Indicator'] == 'BMI')
        
        # Get target gender values
        height = float(next(detail['Value'] for detail in height_data['Category Details'] 
                          if target_gender in detail['Category']))
        weight = float(next(detail['Value'] for detail in weight_data['Category Details'] 
                          if target_gender in detail['Category']))
        bmi = float(next(detail['Value'] for detail in bmi_data['Category Details'] 
                        if target_gender in detail['Category']))
        
        return {
            'height': height,
            'weight': weight,
            'bmi': bmi
        }
    except Exception as e:
        st.error(f"Error getting {target_gender} data: {str(e)}")
        return {
            'height': 1.70 if target_gender == 'Male' else 1.60,
            'weight': 75 if target_gender == 'Male' else 60,
            'bmi': 24
        }

def get_age_probability(age_gender_df, min_age, max_age, target_gender):
    """Calculate age probability for specific gender"""
    total_probability = 0
    for age_data in age_gender_df:
        age_range = age_data['Age Range']
        range_min, range_max = map(int, age_range.split('-')) if '-' in age_range else (80, 100)
        
        if range_min <= max_age and range_max >= min_age:
            # Get gender-specific percentage
            gender_percentage = next(
                detail['Percentage'] for detail in age_data['Gender Details']
                if detail['Gender'] == target_gender
            )
            total_probability += gender_percentage
    
    return total_probability / 100.0

def load_and_process_data(target_gender):
    """Load and process multiple datasets"""
    try:
        # Initialize final dataframe
        final_df = pd.DataFrame(columns=["Indicator", "Value/Distribution"])
        
        # Load all datasets
        with open("../data/final/age_gender_distribution.json", 'r') as f:
            age_gender_df = json.load(f)
        with open("../data/final/income_distribution.json", 'r') as f:
            income_df = json.load(f)
        with open("../data/final/percentages_result.json", 'r') as f:
            percentages_df = json.load(f)
        
        # Get gender-specific anthropometric data
        anthro_data = get_gender_specific_data(percentages_df, target_gender)
        
        # Store averages in final_df
        anthropometric_data = [
            ('Average Height (m)', anthro_data['height']),
            ('Average Weight (kg)', anthro_data['weight']),
            ('Average BMI', anthro_data['bmi'])
        ]
        
        for indicator, value in anthropometric_data:
            final_df = pd.concat([final_df, pd.DataFrame({
                'Indicator': [indicator],
                'Value/Distribution': [value]
            })], ignore_index=True)

        return final_df, age_gender_df, income_df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(columns=["Indicator", "Value/Distribution"]), None, None

def render_calculator(target_gender):
    """Render the calculator interface for specific gender"""
    st.title(f"🔮 Finding Your Dream {target_gender} Partner")
    st.write(f"Calculate the probability of finding your ideal {target_gender.lower()} partner based on your criteria!")
    
    # Load gender-specific data
    df, age_data, income_data = load_and_process_data(target_gender)
    
    # Age Range Selection
    st.subheader("🎯 Age Preference")
    age_min, age_max = st.slider(
        "Select preferred age range:",
        min_value=18,
        max_value=80,
        value=(25, 35),
        step=1
    )
    
    # Income Range Selection
    st.subheader("💰 Income Preference")
    try:
        income_data_list = []
        for item in income_data:
            for detail in item['Region Details']:
                if detail['Region'] == 'National (in MAD)':
                    income_data_list.append(detail['Income'])
        
        min_income = 6000 # min(income_data_list)
        max_income = 57000 # max(income_data_list)
        
        income_min = st.slider(
            "Select minimum preferred monthly income (in MAD):",
            min_value=int(min_income),
            max_value=int(max_income),
            value=int(min_income),
            step=500
        )
    except Exception as e:
        st.warning(f"Error setting up income slider: {str(e)}")
        income_min = st.slider(
            "Select preferred monthly income range (in MAD):",
            min_value=3000,
            max_value=50000,
            value=5000,
            step=500
        )

    # Physical Characteristics
    st.subheader("📏 Physical Characteristics")
    
    # Initialize default values
    height = 1.70 if target_gender == 'Male' else 1.60
    weight = 75 if target_gender == 'Male' else 60
    bmi = 24.0
    avg_height = height
    avg_weight = weight
    avg_bmi = bmi

    try:
        if not df.empty:
            avg_height = float(df[df['Indicator'] == 'Average Height (m)']['Value/Distribution'].values[0])
            avg_weight = float(df[df['Indicator'] == 'Average Weight (kg)']['Value/Distribution'].values[0])
            avg_bmi = float(df[df['Indicator'] == 'Average BMI']['Value/Distribution'].values[0])
        
        height = st.slider(
            f"Preferred height for {target_gender.lower()} partner (m):",
            min_value=1.50,
            max_value=2.00,
            value=float(avg_height),
            step=0.01,
            help=f"Average {target_gender.lower()} height in Morocco is {avg_height:.2f}m"
        )
        
        weight = st.slider(
            f"Preferred weight for {target_gender.lower()} partner (kg):",
            min_value=40,
            max_value=120,
            value=int(avg_weight),
            step=1,
            help=f"Average {target_gender.lower()} weight in Morocco is {avg_weight:.1f}kg"
        )
        
        bmi = st.slider(
            f"Preferred BMI for {target_gender.lower()} partner:",
            min_value=18.5,
            max_value=35.0,
            value=float(avg_bmi),
            step=0.1,
            help=f"Average {target_gender.lower()} BMI in Morocco is {avg_bmi:.1f}"
        )
    except Exception as e:
        st.error(f"Error setting up physical characteristic sliders: {str(e)}")
    
    # Calculate probabilities
    if True:
        # Age probability
        age_prob = get_age_probability(age_data, age_min, age_max, target_gender)
        
        # Income probability
        income_prob = 0.2  # Simplified for now
        
        # Physical characteristic probabilities
        try:
            height_prob = calculate_anthropometric_probability(height, avg_height)
            weight_prob = calculate_anthropometric_probability(weight, avg_weight)
            bmi_prob = calculate_anthropometric_probability(bmi, avg_bmi)
        except Exception as e:
            st.error(f"Error calculating physical probabilities: {str(e)}")
            height_prob = weight_prob = bmi_prob = 1.0
        
        # Calculate final probability using average instead of multiplication
        all_probs = [age_prob, income_prob, height_prob, weight_prob, bmi_prob]
        probability = np.mean(all_probs)  # Changed from np.prod to np.mean
        
        # Adjust thresholds for the new averaging method
        # Display results
        st.subheader("📊 Results")
        
        # Show individual probabilities
        st.info(f"Age range ({age_min}-{age_max}) probability: {age_prob:.1%}")
        st.info(f"Minimum Income ({income_min:,} MAD) probability: {income_prob:.1%}")
        st.info(f"Height ({height:.2f}m) probability: {height_prob:.1%}")
        st.info(f"Weight ({weight:.1f}kg) probability: {weight_prob:.1%}")
        st.info(f"BMI ({bmi:.1f}) probability: {bmi_prob:.1%}")
        
        # Adjust color thresholds for averaged probabilities
        if probability < 0.3:  # Adjusted from 0.1
            color = "red"
        elif probability < 0.5:  # Adjusted from 0.3
            color = "orange"
        else:
            color = "green"
            
        st.markdown(
            f"<h1 style='color: {color};'>Overall Match Score: {probability:.2%}</h1>",
            unsafe_allow_html=True
        )
        
        # Add interpretation with adjusted thresholds
        st.subheader("🔮 Interpretation")
        if probability < 0.3:
            st.write("Your criteria are quite selective! Consider being more flexible in your preferences.")
        elif probability < 0.5:
            st.write("You have a moderate match score. Some of your criteria align well with the population.")
        else:
            st.write("Your criteria align well with the population statistics. Good prospects!")
        
        # Updated disclaimer
        st.caption(
            "⚠️ Disclaimer: This calculator uses real demographic data from Morocco but makes "
            "simplified assumptions. The match score is an average of individual probabilities, "
            "providing a more balanced view of compatibility. Real relationships are complex "
            "and can't be reduced to statistics."
        )

def main():
    set_page_config()
    
    # Create sidebar for gender selection
    st.sidebar.title("🔍 Partner Search")
    looking_for = st.sidebar.radio(
        "I am looking for:",
        ["A Female Partner", "A Male Partner"],
        index=0
    )
    
    # Show appropriate calculator based on selection
    if looking_for == "A Female Partner":
        render_calculator("Female")
    else:
        render_calculator("Male")
    
    # Add credits/footer in sidebar
    st.sidebar.markdown("---")
    st.sidebar.caption("Made with ❤️ using Moroccan demographic data")

if __name__ == "__main__":
    main()
