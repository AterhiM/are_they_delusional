import streamlit as st
import pandas as pd
import numpy as np
import json
import scipy.stats as stats
import os
data_path = os.path.join(os.path.dirname(__file__), "../data/final")

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
        with open(os.path.join(data_path, "age_gender_distribution.json"), 'r') as f:
            age_gender_df = json.load(f)
        with open(f"{data_path}/income_distribution.json", 'r') as f:
            income_df = json.load(f)
        with open(f"{data_path}/percentages_result.json", 'r') as f:
            percentages_df = json.load(f)
        with open(f"{data_path}/religion_result.json", 'r') as f:
            religion_df = json.load(f)

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

        return final_df, age_gender_df, income_df, religion_df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(columns=["Indicator", "Value/Distribution"]), None, None, None

def get_religion_probability(religion_df, selected_religion):
    """Calculate probability for selected religion"""
    try:
        religion_data = religion_df[0]['Category Details']
        
        # Define mapping for selected religions to data categories
        religion_mapping = {
            "Sunni Muslim": "Sunni Muslims (%)",
            "Shia Muslim": "Shia Muslims (%)",
            "Jewish": "Jews (People)",
            "Christian": "Christians (Estimate)",
            "Other": "Baha'is and Other Groups"
        }
        
        # Get the corresponding category from the data
        category = religion_mapping.get(selected_religion)
        if not category:
            return 0.01
            
        # Find the matching category in the data
        for detail in religion_data:
            if detail['Category'] == category:
                population_str = detail['Population'].lower()
                
                # Handle different formats of population data
                if "99%" in population_str:
                    return 0.99
                elif "0.1%" in population_str:
                    return 0.001
                elif "3,500 people" in population_str:
                    return 3500 / 36.7e6  # Convert to percentage of total population
                elif "between 1,500 and 30,000" in population_str:
                    avg_christian = (1500 + 30000) / 2
                    return avg_christian / 36.7e6
                elif "less than 1%" in population_str:
                    return 0.005  # Assuming 0.5% for other groups
                
        return 0.01  # Default small probability if not found
        
    except Exception as e:
        st.error(f"Error calculating religion probability: {str(e)}")
        return 0.01

def calculate_income_probability(income, income_data):
    """
    Calculate probability of having a specific income based on income distribution
    
    Args:
        income (float): Income value to check
        income_data (list): List of dictionaries containing income distribution data
        
    Returns:
        float: Probability of having that income or higher
    """
    try:
        # Extract national income values
        incomes = [quintile["Region Details"][0]["Income"] for quintile in income_data]
        
        # Calculate mean and standard deviation
        mean = np.mean(incomes)
        std = np.std(incomes)
        
        # Calculate probability using normal distribution
        # We use 1 - CDF because we want probability of income being higher than target
        probability = 1 - stats.norm.cdf(income, mean, std)
        
        return max(probability, 0.01)  # Ensure minimum probability of 1%
    except Exception as e:
        st.error(f"Error calculating income probability: {str(e)}")
        return 0.01

def render_calculator(target_gender):
    """Render the calculator interface for specific gender"""
    st.title(f"🔮 Are You Delusional about Your Dream {target_gender} Partner?")
    st.write(
        f"Let's calculate how *realistic* your expectations are for finding your dream "
        f"{target_gender.lower()} partner in Morocco... *statistically speaking*."
    )
    
    # Load gender-specific data
    df, age_data, income_data, religion_df = load_and_process_data(target_gender)
    
    # Age Range Selection
    st.subheader("🎯 Age Preferences")
    st.caption("Because age is just a number (until you do the math)")
    age_min, age_max = st.slider(
        "Your ideal age range:",
        min_value=18,
        max_value=80,
        value=(25, 35),
        step=1
    )
    
    # Income Range Selection
    st.subheader("💰 Financial Expectations")
    st.caption("Let's see if your financial expectations could be met..realisticaly!")
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
    st.subheader("📏 The 'Shallow' Section")
    st.caption("Numbers don't lie, but they might hurt your feelings")
    
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
    
    # Religion Selection
    st.subheader("🕌 Spiritual Alignment")
    st.caption("Because faith can move mountains, but statistics are harder to budge")
    religion_options = [
        "Sunni Muslim",
        "Shia Muslim",
        "Jewish",
        "Christian",
        "Other"
    ]
    selected_religion = st.selectbox(
        "Your preferred religious background:",
        religion_options,
        help="Based on actual religious distribution in Morocco"
    )

    # Add Search Button
    st.markdown("---")
    search_clicked = st.button("🔍 Calculate Your Delusion Score", use_container_width=True)

    # Calculate and display results only if search is clicked
    if search_clicked:
        # Age probability
        age_prob = get_age_probability(age_data, age_min, age_max, target_gender)
        
        # Income probability
        income_prob = calculate_income_probability(income_min, income_data)
        
        # Physical characteristic probabilities
        try:
            height_prob = calculate_anthropometric_probability(height, avg_height)
            weight_prob = calculate_anthropometric_probability(weight, avg_weight)
            bmi_prob = calculate_anthropometric_probability(bmi, avg_bmi)
        except Exception as e:
            st.error(f"Error calculating physical probabilities: {str(e)}")
            height_prob = weight_prob = bmi_prob = 1.0

        # Religion probability
        religion_prob = get_religion_probability(religion_df, selected_religion)
        
        # Calculate final probability using product and normalize to 0-1 scale
        all_probs = [age_prob, income_prob, height_prob, weight_prob, bmi_prob, religion_prob]
        raw_probability = np.prod(all_probs)
        match_probability = min(raw_probability / 0.3, 1.0)  # Normalize to 0-1 scale
        
        # Calculate delusion score as inverse of probability
        delusion_score = 1 - match_probability
        
        # Display results
        st.subheader("📊 The Reality Check")
        
        # Show individual probabilities with sarcastic labels
        st.info(f"Age compatibility (because time waits for no one): {age_prob:.1%}")
        st.info(f"Financial reality check: {income_prob:.1%}")
        st.info(f"Height hopes and dreams: {height_prob:.1%}")
        st.info(f"Weight expectations: {weight_prob:.1%}")
        st.info(f"BMI reality: {bmi_prob:.1%}")
        st.info(f"Religious alignment chances: {religion_prob:.1%}")

        # Overall score with adjusted messages based on delusion score ranges
        if delusion_score > 0.967:  # > 96.7% delusional
            color = "red"
            message = "Highly Delusional"
        elif delusion_score > 0.833:  # 83.3-96.7% delusional
            color = "orange"
            message = "Mildly Delusional"
        elif delusion_score > 0.5:  # 50-83.3% delusional
            color = "yellow"
            message = "Mildly Delusional"
        elif delusion_score > 0:  # 0-50% delusional
            color = "lightgreen"
            message = "Surprisingly Reasonable"
        else:  # 0% delusional
            color = "green"
            message = "Surprisingly Reasonable"
            
        st.markdown(
            f"<h1 style='color: {color};'>Delusion Score: {delusion_score:.2%} ({message})</h1>",
            unsafe_allow_html=True
        )
        
        # Add interpretation with sarcastic touch
        st.subheader("🔮 Professional Opinion")
        if delusion_score > 0.967:
            st.write("Your standards are so high, they need oxygen masks! Maybe consider returning to Earth?")
        elif delusion_score > 0.833:
            st.write("Your standards are in the stratosphere. Time for a reality check!")
        elif delusion_score > 0.5:
            st.write("You're walking the fine line between optimistic and delusional. At least you're not alone!")
        elif delusion_score > 0:
            st.write("Well, well... looks like someone's being surprisingly reasonable! There might be hope for you.")
        else:
            st.write("Your criteria are actually quite reasonable! Who would have thought?")
        
        # Updated disclaimer
        st.caption(
            "⚠️ Disclaimer: This calculator uses real Moroccan demographic data but doesn't account for "
            "love, chemistry, or your charming personality. Remember, statistics don't care about your "
            "feelings, but someone out there might!"
        )

def main():
    set_page_config()
    
    # Create sidebar for gender selection
    st.sidebar.title("🔍 Reality Check Settings")
    looking_for = st.sidebar.radio(
        "I'm hoping to find:",
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
    st.sidebar.caption("*Made with ❤️ and some cold, hard to find statistics of Morocco*")

if __name__ == "__main__":
    main()
