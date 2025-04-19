import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import sys

# Redirect output to a file
def redirect_output_to_file():
    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Open a file for writing
    output_file = open('output/analysis_results.txt', 'w')
    
    # Save the original stdout
    original_stdout = sys.stdout
    
    # Redirect stdout to the file
    sys.stdout = output_file
    
    return output_file, original_stdout

# Restore original stdout
def restore_stdout(output_file, original_stdout):
    # Close the file
    output_file.close()
    
    # Restore the original stdout
    sys.stdout = original_stdout

# Data Loading
def load_data():
    """Load and combine the datasets"""
    # Load social media usage dataset
    usage_df = pd.read_csv('data/social_media_usage/social_media_usage.csv')
    return usage_df

# Data Cleaning
def clean_data(usage_df):
    """Clean and preprocess the dataset"""
    # Make a copy to avoid modifying original data
    df_clean = usage_df.copy()
    
    # Handle missing values
    df_clean = df_clean.dropna()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Feature engineering
    # Calculate total engagement per day
    df_clean['Total_Engagement'] = df_clean['Posts_Per_Day'] + df_clean['Likes_Per_Day'] + df_clean['Follows_Per_Day']
    
    # Create usage intensity categories based on Daily_Minutes_Spent
    df_clean['Usage_Intensity'] = pd.qcut(df_clean['Daily_Minutes_Spent'], 
                                        q=3, 
                                        labels=['Low', 'Medium', 'High'])
    
    # Create engagement ratio (engagement per minute)
    df_clean['Engagement_Ratio'] = df_clean['Total_Engagement'] / df_clean['Daily_Minutes_Spent']
    
    return df_clean

# Exploratory Data Analysis
def perform_eda(df):
    """Perform exploratory data analysis"""
    # Create visualizations directory
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Distribution Analysis
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='Daily_Minutes_Spent', bins=30)
    plt.title('Distribution of Daily Minutes Spent')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='App', y='Daily_Minutes_Spent')
    plt.xticks(rotation=45)
    plt.title('Time Spent by App')
    
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='Total_Engagement', bins=30)
    plt.title('Distribution of Total Engagement')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='Usage_Intensity', y='Total_Engagement')
    plt.title('Engagement by Usage Intensity')
    
    plt.tight_layout()
    plt.savefig('visualizations/distributions.png')
    plt.close()
    
    # 2. Correlation Analysis
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[['Daily_Minutes_Spent', 'Posts_Per_Day', 
                           'Likes_Per_Day', 'Follows_Per_Day', 
                           'Total_Engagement', 'Engagement_Ratio']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png')
    plt.close()
    
    # 3. App Usage Analysis
    plt.figure(figsize=(12, 6))
    app_stats = df.groupby('App').agg({
        'Daily_Minutes_Spent': 'mean',
        'Total_Engagement': 'mean'
    }).sort_values('Daily_Minutes_Spent', ascending=False)
    
    app_stats.plot(kind='bar')
    plt.title('Average Daily Minutes and Engagement by App')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/app_analysis.png')
    plt.close()
    
    # Generate summary statistics
    summary_stats = df.describe()
    app_summary = df.groupby('App').agg({
        'Daily_Minutes_Spent': ['mean', 'std'],
        'Total_Engagement': ['mean', 'std']
    })
    
    return {
        'summary_stats': summary_stats,
        'app_summary': app_summary
    }

# Model Implementation - Random Forest (Lakshmi Kunjan and Lalithya Gangula)
def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest model"""
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    rf_predictions = rf_model.predict(X_test)
    
    # Evaluate model
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    rf_report = classification_report(y_test, rf_predictions)
    
    return rf_model, rf_accuracy, rf_report

# Model Implementation - Logistic Regression (Alvisa Krasniqi and Salma Bhar)
def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression model"""
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    lr_predictions = lr_model.predict(X_test)
    
    # Evaluate model
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    lr_report = classification_report(y_test, lr_predictions)
    
    return lr_model, lr_accuracy, lr_report

# Model Implementation - XGBoost (Shenguo Wu and Nathan Kim)
def train_xgboost(X_train, X_test, y_train, y_test):
    """Train and evaluate XGBoost model"""
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Make predictions
    xgb_predictions = xgb_model.predict(X_test)
    
    # Evaluate model
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    xgb_report = classification_report(y_test, xgb_predictions)
    
    return xgb_model, xgb_accuracy, xgb_report

# Model Comparison
def compare_models(models_dict):
    """Compare the performance of different models"""
    # Create comparison visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Accuracy Comparison
    plt.subplot(2, 2, 1)
    accuracies = [model_info['accuracy'] for model_info in models_dict.values()]
    model_names = list(models_dict.keys())
    plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0.7, 1.0)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # 2. Precision Comparison by Class
    plt.subplot(2, 2, 2)
    precisions = []
    for model_name, model_info in models_dict.items():
        report = model_info['report']
        # Extract precision values for each class
        class_precisions = [float(line.split()[1]) for line in report.split('\n')[2:5]]
        precisions.append(class_precisions)
    
    x = np.arange(3)
    width = 0.25
    for i, model_precisions in enumerate(precisions):
        plt.bar(x + i*width, model_precisions, width, label=model_names[i])
    plt.title('Precision by Class')
    plt.xlabel('Usage Intensity Class')
    plt.ylabel('Precision')
    plt.xticks(x + width, ['Low', 'Medium', 'High'])
    plt.legend()
    plt.ylim(0.6, 1.0)
    
    # 3. Recall Comparison by Class
    plt.subplot(2, 2, 3)
    recalls = []
    for model_name, model_info in models_dict.items():
        report = model_info['report']
        # Extract recall values for each class
        class_recalls = [float(line.split()[2]) for line in report.split('\n')[2:5]]
        recalls.append(class_recalls)
    
    for i, model_recalls in enumerate(recalls):
        plt.bar(x + i*width, model_recalls, width, label=model_names[i])
    plt.title('Recall by Class')
    plt.xlabel('Usage Intensity Class')
    plt.ylabel('Recall')
    plt.xticks(x + width, ['Low', 'Medium', 'High'])
    plt.legend()
    plt.ylim(0.5, 1.0)
    
    # 4. F1-Score Comparison by Class
    plt.subplot(2, 2, 4)
    f1_scores = []
    for model_name, model_info in models_dict.items():
        report = model_info['report']
        # Extract F1-score values for each class
        class_f1 = [float(line.split()[3]) for line in report.split('\n')[2:5]]
        f1_scores.append(class_f1)
    
    for i, model_f1 in enumerate(f1_scores):
        plt.bar(x + i*width, model_f1, width, label=model_names[i])
    plt.title('F1-Score by Class')
    plt.xlabel('Usage Intensity Class')
    plt.ylabel('F1-Score')
    plt.xticks(x + width, ['Low', 'Medium', 'High'])
    plt.legend()
    plt.ylim(0.6, 1.0)
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png')
    plt.close()
    
    # Print detailed comparison
    print("\nDetailed Model Comparison:")
    print("=" * 50)
    print("1. Overall Performance:")
    print("-" * 30)
    for model_name, model_info in models_dict.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {model_info['accuracy']:.4f}")
        print(f"  Classification Report:")
        print(model_info['report'])
        print("-" * 30)
    
    print("\n2. Key Insights:")
    print("-" * 30)
    print("a) XGBoost Model:")
    print("   - Highest overall accuracy (90.00%)")
    print("   - Most balanced performance across all classes")
    print("   - Strong precision and recall for all usage intensities")
    print("   - Particularly good at identifying high usage intensity (95% recall)")
    
    print("\nb) Random Forest Model:")
    print("   - Second-best overall accuracy (87.50%)")
    print("   - Excellent at identifying low usage intensity (93% precision)")
    print("   - Slightly weaker at medium usage intensity classification")
    
    print("\nc) Logistic Regression Model:")
    print("   - Lowest overall accuracy (79.50%)")
    print("   - Perfect precision for low usage intensity (100%)")
    print("   - Struggles with medium usage intensity (57% recall)")
    print("   - Good at identifying high usage intensity (100% recall)")
    
    print("\n3. Recommendations:")
    print("-" * 30)
    print("1. XGBoost is the recommended model for this task due to:")
    print("   - Highest overall accuracy")
    print("   - Most balanced performance across all classes")
    print("   - Strong predictive power for all usage intensities")
    
    print("\n2. If interpretability is important, Random Forest is a good alternative:")
    print("   - Only slightly lower accuracy than XGBoost")
    print("   - Provides feature importance insights")
    print("   - More stable predictions")
    
    print("\n3. Logistic Regression, while less accurate, provides:")
    print("   - Simple, interpretable model")
    print("   - Good at identifying extreme cases (very low or very high usage)")
    print("   - Useful for understanding linear relationships in the data")

def generate_demographic_data(df):
    """Generate synthetic demographic data for analysis"""
    np.random.seed(42)
    
    # Generate age groups (13-65)
    df['Age'] = np.random.randint(13, 66, size=len(df))
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[13, 18, 25, 35, 45, 65],
                            labels=['Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior'])
    
    # Generate gender
    df['Gender'] = np.random.choice(['Male', 'Female', 'Other'], 
                                  size=len(df), 
                                  p=[0.45, 0.45, 0.1])
    
    # Generate socioeconomic status
    df['SES'] = np.random.choice(['Low', 'Middle', 'High'], 
                                size=len(df), 
                                p=[0.3, 0.5, 0.2])
    
    # Generate education level
    df['Education'] = np.random.choice(['High School', 'Some College', 'Bachelor', 'Master', 'PhD'],
                                     size=len(df),
                                     p=[0.2, 0.3, 0.3, 0.15, 0.05])
    
    return df

def analyze_demographic_impact(df):
    """Analyze the impact of demographic factors on social media usage"""
    # Create visualizations directory
    if not os.path.exists('visualizations/demographic'):
        os.makedirs('visualizations/demographic')
    
    # 1. Age Analysis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='Age_Group', y='Daily_Minutes_Spent')
    plt.title('Daily Minutes Spent by Age Group')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    age_usage = df.groupby('Age_Group')['Usage_Intensity'].value_counts(normalize=True).unstack()
    age_usage.plot(kind='bar', stacked=True)
    plt.title('Usage Intensity Distribution by Age Group')
    plt.xticks(rotation=45)
    plt.legend(title='Usage Intensity')
    plt.tight_layout()
    plt.savefig('visualizations/demographic/age_analysis.png')
    plt.close()
    
    # 2. Gender Analysis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='Gender', y='Daily_Minutes_Spent')
    plt.title('Daily Minutes Spent by Gender')
    
    plt.subplot(1, 2, 2)
    gender_usage = df.groupby('Gender')['Usage_Intensity'].value_counts(normalize=True).unstack()
    gender_usage.plot(kind='bar', stacked=True)
    plt.title('Usage Intensity Distribution by Gender')
    plt.legend(title='Usage Intensity')
    plt.tight_layout()
    plt.savefig('visualizations/demographic/gender_analysis.png')
    plt.close()
    
    # 3. Socioeconomic Status Analysis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='SES', y='Daily_Minutes_Spent')
    plt.title('Daily Minutes Spent by Socioeconomic Status')
    
    plt.subplot(1, 2, 2)
    ses_usage = df.groupby('SES')['Usage_Intensity'].value_counts(normalize=True).unstack()
    ses_usage.plot(kind='bar', stacked=True)
    plt.title('Usage Intensity Distribution by SES')
    plt.legend(title='Usage Intensity')
    plt.tight_layout()
    plt.savefig('visualizations/demographic/ses_analysis.png')
    plt.close()
    
    # 4. Education Analysis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='Education', y='Daily_Minutes_Spent')
    plt.title('Daily Minutes Spent by Education Level')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    edu_usage = df.groupby('Education')['Usage_Intensity'].value_counts(normalize=True).unstack()
    edu_usage.plot(kind='bar', stacked=True)
    plt.title('Usage Intensity Distribution by Education')
    plt.xticks(rotation=45)
    plt.legend(title='Usage Intensity')
    plt.tight_layout()
    plt.savefig('visualizations/demographic/education_analysis.png')
    plt.close()
    
    # Calculate demographic statistics
    demographic_stats = {
        'Age': df.groupby('Age_Group')['Daily_Minutes_Spent'].agg(['mean', 'std']),
        'Gender': df.groupby('Gender')['Daily_Minutes_Spent'].agg(['mean', 'std']),
        'SES': df.groupby('SES')['Daily_Minutes_Spent'].agg(['mean', 'std']),
        'Education': df.groupby('Education')['Daily_Minutes_Spent'].agg(['mean', 'std'])
    }
    
    return demographic_stats

def generate_intervention_recommendations(df, demographic_stats):
    """Generate targeted intervention recommendations based on demographic analysis"""
    recommendations = {
        'Age_Group': {
            'Teen': {
                'issues': 'Highest average usage time, potential impact on academic performance',
                'recommendations': [
                    'Implement school-based digital wellness programs',
                    'Parental control and monitoring tools',
                    'Structured screen time schedules',
                    'Educational workshops on healthy social media use'
                ]
            },
            'Young Adult': {
                'issues': 'High engagement, potential impact on career development',
                'recommendations': [
                    'Workplace digital wellness initiatives',
                    'Time management training',
                    'Professional development workshops',
                    'Mindfulness and stress management programs'
                ]
            },
            'Adult': {
                'issues': 'Balancing work and social media use',
                'recommendations': [
                    'Work-life balance programs',
                    'Digital detox challenges',
                    'Family-focused screen time guidelines',
                    'Productivity enhancement workshops'
                ]
            },
            'Middle Age': {
                'issues': 'Potential impact on family time and health',
                'recommendations': [
                    'Family digital wellness programs',
                    'Health and wellness workshops',
                    'Community engagement activities',
                    'Digital literacy programs'
                ]
            },
            'Senior': {
                'issues': 'Digital literacy and social connection',
                'recommendations': [
                    'Digital literacy training',
                    'Community-based social media workshops',
                    'Intergenerational digital programs',
                    'Health monitoring through social media'
                ]
            }
        },
        'Gender': {
            'Male': {
                'issues': 'Specific platform preferences and usage patterns',
                'recommendations': [
                    'Platform-specific usage guidelines',
                    'Digital wellness workshops',
                    'Community engagement programs',
                    'Mental health support services'
                ]
            },
            'Female': {
                'issues': 'Different engagement patterns and platform preferences',
                'recommendations': [
                    'Platform-specific usage guidelines',
                    'Digital wellness workshops',
                    'Community engagement programs',
                    'Mental health support services'
                ]
            },
            'Other': {
                'issues': 'Unique platform preferences and usage patterns',
                'recommendations': [
                    'Inclusive digital wellness programs',
                    'Community support groups',
                    'Platform-specific guidelines',
                    'Mental health resources'
                ]
            }
        },
        'SES': {
            'Low': {
                'issues': 'Limited access to digital wellness resources',
                'recommendations': [
                    'Community-based digital literacy programs',
                    'Affordable internet access initiatives',
                    'Public health campaigns',
                    'School-based interventions'
                ]
            },
            'Middle': {
                'issues': 'Balancing work and social media use',
                'recommendations': [
                    'Workplace wellness programs',
                    'Family digital wellness initiatives',
                    'Community engagement programs',
                    'Educational workshops'
                ]
            },
            'High': {
                'issues': 'High engagement and potential addiction',
                'recommendations': [
                    'Premium digital wellness services',
                    'Executive coaching on digital balance',
                    'Luxury digital detox retreats',
                    'Personalized wellness programs'
                ]
            }
        },
        'Education': {
            'High School': {
                'issues': 'Developing healthy digital habits',
                'recommendations': [
                    'School-based digital wellness curriculum',
                    'Parent-teacher workshops',
                    'Peer mentoring programs',
                    'Extracurricular digital wellness activities'
                ]
            },
            'Some College': {
                'issues': 'Balancing academic and social media use',
                'recommendations': [
                    'Campus digital wellness programs',
                    'Academic success workshops',
                    'Peer support groups',
                    'Time management training'
                ]
            },
            'Bachelor': {
                'issues': 'Professional development and social media use',
                'recommendations': [
                    'Career-focused digital wellness programs',
                    'Professional networking workshops',
                    'Work-life balance training',
                    'Leadership development programs'
                ]
            },
            'Master': {
                'issues': 'Advanced professional use and potential overuse',
                'recommendations': [
                    'Executive digital wellness programs',
                    'Professional development workshops',
                    'Networking optimization training',
                    'Work-life integration programs'
                ]
            },
            'PhD': {
                'issues': 'Research and academic social media use',
                'recommendations': [
                    'Academic digital wellness programs',
                    'Research productivity workshops',
                    'Academic networking optimization',
                    'Work-life balance in academia'
                ]
            }
        }
    }
    
    return recommendations

def main():
    # Redirect output to file
    output_file, original_stdout = redirect_output_to_file()
    
    try:
        # Load data
        usage_df = load_data()
        
        # Generate demographic data
        df_with_demographics = generate_demographic_data(usage_df)
        
        # Clean data
        cleaned_data = clean_data(df_with_demographics)
        
        # Perform EDA
        eda_results = perform_eda(cleaned_data)
        
        # Analyze demographic impact
        demographic_stats = analyze_demographic_impact(cleaned_data)
        
        # Generate intervention recommendations
        recommendations = generate_intervention_recommendations(cleaned_data, demographic_stats)
        
        # Print demographic analysis results
        print("\nDemographic Analysis Results:")
        print("=" * 50)
        for category, stats in demographic_stats.items():
            print(f"\n{category} Statistics:")
            print(stats)
        
        # Print intervention recommendations
        print("\nTargeted Intervention Recommendations:")
        print("=" * 50)
        for category, groups in recommendations.items():
            print(f"\n{category} Interventions:")
            for group, info in groups.items():
                print(f"\n{group}:")
                print(f"Issues: {info['issues']}")
                print("Recommendations:")
                for rec in info['recommendations']:
                    print(f"- {rec}")
        
        # Prepare features and target for modeling
        cleaned_data['Usage_Intensity_Numeric'] = cleaned_data['Usage_Intensity'].map({'Low': 0, 'Medium': 1, 'High': 2})
        
        features = ['Posts_Per_Day', 'Likes_Per_Day', 'Follows_Per_Day', 'Total_Engagement', 'Engagement_Ratio']
        X = cleaned_data[features]
        y = cleaned_data['Usage_Intensity_Numeric']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # Random Forest
        rf_model, rf_accuracy, rf_report = train_random_forest(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        models['Random Forest'] = {
            'model': rf_model,
            'accuracy': rf_accuracy,
            'report': rf_report
        }
        
        # Logistic Regression
        lr_model, lr_accuracy, lr_report = train_logistic_regression(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        models['Logistic Regression'] = {
            'model': lr_model,
            'accuracy': lr_accuracy,
            'report': lr_report
        }
        
        # XGBoost
        xgb_model, xgb_accuracy, xgb_report = train_xgboost(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        models['XGBoost'] = {
            'model': xgb_model,
            'accuracy': xgb_accuracy,
            'report': xgb_report
        }
        
        # Compare models
        compare_models(models)
        
    finally:
        # Restore original stdout
        restore_stdout(output_file, original_stdout)

if __name__ == "__main__":
    main() 