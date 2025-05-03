import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from zipfile import ZipFile

st.set_page_config(
    page_title="Dataset Visualizer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS to improve aesthetics
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stSidebar {
        padding-top: 2rem;
    }
    .visualization-container {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.title("📊 Dataset Visualizer")
st.write("Upload your CSV or Excel file to generate automatic visualizations and analysis")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

# Function to detect data types
def detect_data_types(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Try to convert potential datetime columns that were parsed as objects
    for col in categorical_cols.copy():
        try:
            pd.to_datetime(df[col])
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except:
            pass
    
    return numeric_cols, categorical_cols, datetime_cols

# Function to generate summary statistics
def generate_summary(df):
    st.header("📝 Dataset Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Numeric Columns", len(df.select_dtypes(include=['number']).columns))
    with col4:
        st.metric("Missing Values", df.isna().sum().sum())
    
    # Show first few rows
    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Missing values heatmap
    st.subheader("Missing Values")
    if df.isna().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.success("No missing values in the dataset!")
    
    # Show data types and summary statistics
    st.subheader("Data Types")
    dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
    st.dataframe(dtypes_df, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include='all').T, use_container_width=True)

# Function to generate visualizations for numeric columns
def visualize_numeric(df, numeric_cols):
    st.header("📈 Numeric Data Visualization")
    
    # Correlation heatmap if there are multiple numeric columns
    if len(numeric_cols) > 1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                   square=True, linewidths=0.5, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Individual column visualizations
    for col in numeric_cols:
        st.subheader(f"Analysis of '{col}'")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            st.pyplot(fig)
            
        with col2:
            # Box plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(y=df[col].dropna(), ax=ax)
            plt.title(f"Box Plot of {col}")
            plt.tight_layout()
            st.pyplot(fig)
            
        # Q-Q plot to check normality
        fig, ax = plt.subplots(figsize=(10, 6))
        from scipy import stats
        stats.probplot(df[col].dropna(), plot=ax)
        plt.title(f"Q-Q Plot of {col}")
        plt.tight_layout()
        st.pyplot(fig)

# Function to visualize categorical columns
def visualize_categorical(df, categorical_cols):
    st.header("📊 Categorical Data Visualization")
    
    for col in categorical_cols:
        st.subheader(f"Analysis of '{col}'")
        
        # Calculate value counts
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'Count']
        
        # Display the counts
        st.dataframe(value_counts, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            top_n = min(10, len(value_counts))  # Show top 10 categories at most
            sns.barplot(x='Count', y=col, data=value_counts.head(top_n), ax=ax)
            plt.title(f"Frequency of {col} Categories")
            plt.tight_layout()
            st.pyplot(fig)
            
            if top_n < len(value_counts):
                st.info(f"Showing top {top_n} out of {len(value_counts)} categories")
        
        with col2:
            # Pie chart
            fig, ax = plt.subplots(figsize=(10, 6))
            if len(value_counts) > 10:
                # Group small categories into "Others"
                top_cats = value_counts.head(9)[col].tolist()
                others_count = value_counts[~value_counts[col].isin(top_cats)]['Count'].sum()
                
                pie_data = value_counts[value_counts[col].isin(top_cats)].copy()
                pie_data = pd.concat([pie_data, pd.DataFrame({col: ['Others'], 'Count': [others_count]})], 
                                    ignore_index=True)
                
                pie_data.plot.pie(y='Count', labels=pie_data[col], autopct='%1.1f%%', ax=ax)
                plt.title(f"Percentage of {col} Categories (Top 9 + Others)")
            else:
                value_counts.plot.pie(y='Count', labels=value_counts[col], autopct='%1.1f%%', ax=ax)
                plt.title(f"Percentage of {col} Categories")
            
            plt.tight_layout()
            st.pyplot(fig)

# Function to visualize datetime columns
def visualize_datetime(df, datetime_cols):
    if not datetime_cols:
        return
        
    st.header("📅 Time Series Visualization")
    
    for col in datetime_cols:
        st.subheader(f"Time Analysis of '{col}'")
        
        # Ensure datetime format
        if df[col].dtype != 'datetime64[ns]':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                st.warning(f"Could not convert '{col}' to datetime format")
                continue
        
        # Line chart of frequency over time
        df_ts = df.copy()
        df_ts['year'] = df_ts[col].dt.year
        df_ts['month'] = df_ts[col].dt.month
        df_ts['day'] = df_ts[col].dt.day
        
        # Group by year-month and count
        time_counts = df_ts.groupby([df_ts[col].dt.year, df_ts[col].dt.month]).size().reset_index()
        time_counts.columns = ['Year', 'Month', 'Count']
        time_counts['Date'] = pd.to_datetime(time_counts['Year'].astype(str) + '-' + 
                                           time_counts['Month'].astype(str) + '-01')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(time_counts['Date'], time_counts['Count'], marker='o')
        plt.title(f"Frequency Over Time ({col})")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Distribution by day of week
        fig, ax = plt.subplots(figsize=(10, 6))
        day_of_week = df_ts[col].dt.day_name()
        day_of_week_counts = day_of_week.value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 
                                                               'Thursday', 'Friday', 'Saturday', 'Sunday'])
        sns.barplot(x=day_of_week_counts.index, y=day_of_week_counts.values, ax=ax)
        plt.title(f"Distribution by Day of Week ({col})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# Function to visualize relationships between variables
def visualize_relationships(df, numeric_cols, categorical_cols):
    if not numeric_cols or (len(numeric_cols) + len(categorical_cols) < 2):
        return
        
    st.header("🔄 Relationship Analysis")
    
    # Allow users to select variables to compare
    st.subheader("Custom Relationship Plots")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Select X variable", df.columns)
    
    with col2:
        y_var = st.selectbox("Select Y variable", df.columns, index=min(1, len(df.columns)-1))
    
    if x_var == y_var:
        st.warning("Please select different variables for X and Y")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot for numeric vs numeric
            if x_var in numeric_cols and y_var in numeric_cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=df[x_var], y=df[y_var], ax=ax)
                plt.title(f"{x_var} vs {y_var}")
                plt.tight_layout()
                st.pyplot(fig)
            
            # Box plot for categorical vs numeric
            elif (x_var in categorical_cols and y_var in numeric_cols) or \
                 (x_var in numeric_cols and y_var in categorical_cols):
                
                cat_var = x_var if x_var in categorical_cols else y_var
                num_var = y_var if y_var in numeric_cols else x_var
                
                # If too many categories, show only top 10
                unique_cats = df[cat_var].nunique()
                if unique_cats > 10:
                    top_cats = df[cat_var].value_counts().nlargest(10).index
                    filtered_df = df[df[cat_var].isin(top_cats)]
                    st.info(f"Showing only top 10 categories out of {unique_cats}")
                else:
                    filtered_df = df
                
                fig, ax = plt.subplots(figsize=(12, 6))
                if cat_var == x_var:
                    sns.boxplot(x=cat_var, y=num_var, data=filtered_df, ax=ax)
                else:
                    sns.boxplot(x=num_var, y=cat_var, data=filtered_df, ax=ax)
                plt.title(f"{cat_var} vs {num_var}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Contingency table for categorical vs categorical
            elif x_var in categorical_cols and y_var in categorical_cols:
                # Create contingency table
                contingency = pd.crosstab(df[x_var], df[y_var])
                st.write("Contingency Table:")
                st.dataframe(contingency)
                
                # Visualize as heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', ax=ax)
                plt.title(f"Contingency Table: {x_var} vs {y_var}")
                plt.tight_layout()
                st.pyplot(fig)
        
        with col2:
            # Add appropriate chart for selected variable types
            if x_var in numeric_cols and y_var in numeric_cols:
                # Add regression line
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.regplot(x=df[x_var], y=df[y_var], scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
                plt.title(f"Regression: {x_var} vs {y_var}")
                
                # Calculate and display correlation
                corr = df[[x_var, y_var]].corr().iloc[0, 1]
                plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                           fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                st.pyplot(fig)
            
            elif (x_var in categorical_cols and y_var in numeric_cols) or \
                 (x_var in numeric_cols and y_var in categorical_cols):
                
                cat_var = x_var if x_var in categorical_cols else y_var
                num_var = y_var if y_var in numeric_cols else x_var
                
                # If too many categories, show only top 10
                unique_cats = df[cat_var].nunique()
                if unique_cats > 10:
                    top_cats = df[cat_var].value_counts().nlargest(10).index
                    filtered_df = df[df[cat_var].isin(top_cats)]
                else:
                    filtered_df = df
                
                fig, ax = plt.subplots(figsize=(12, 6))
                if cat_var == x_var:
                    sns.violinplot(x=cat_var, y=num_var, data=filtered_df, ax=ax)
                else:
                    sns.violinplot(x=num_var, y=cat_var, data=filtered_df, ax=ax)
                plt.title(f"Distribution of {num_var} by {cat_var}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            elif x_var in categorical_cols and y_var in categorical_cols:
                # Mosaic plot
                from statsmodels.graphics.mosaicplot import mosaic
                
                # If too many categories, show only top 5 for each
                x_unique = df[x_var].nunique()
                y_unique = df[y_var].nunique()
                
                if x_unique > 5 or y_unique > 5:
                    top_x_cats = df[x_var].value_counts().nlargest(5).index
                    top_y_cats = df[y_var].value_counts().nlargest(5).index
                    filtered_df = df[df[x_var].isin(top_x_cats) & df[y_var].isin(top_y_cats)]
                    st.info(f"Showing only top 5 categories for each variable due to high cardinality")
                else:
                    filtered_df = df
                
                fig, ax = plt.subplots(figsize=(12, 8))
                mosaic(filtered_df, [x_var, y_var], ax=ax)
                plt.title(f"Mosaic Plot: {x_var} vs {y_var}")
                plt.tight_layout()
                st.pyplot(fig)

# Function to create a downloadable report
def create_downloadable_report(df):
    st.header("📥 Download Analysis Report")
    
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w') as zip_file:
        # Add dataset statistics
        stats_buffer = io.StringIO()
        stats_buffer.write("# Dataset Analysis Report\n\n")
        stats_buffer.write(f"## Basic Information\n")
        stats_buffer.write(f"* Number of rows: {df.shape[0]}\n")
        stats_buffer.write(f"* Number of columns: {df.shape[1]}\n")
        stats_buffer.write(f"* Missing values: {df.isna().sum().sum()}\n\n")
        
        stats_buffer.write("## Column Information\n")
        for col in df.columns:
            stats_buffer.write(f"### {col}\n")
            stats_buffer.write(f"* Data type: {df[col].dtype}\n")
            stats_buffer.write(f"* Missing values: {df[col].isna().sum()} ({df[col].isna().sum() / len(df) * 100:.2f}%)\n")
            if pd.api.types.is_numeric_dtype(df[col]):
                stats_buffer.write(f"* Min: {df[col].min()}\n")
                stats_buffer.write(f"* Max: {df[col].max()}\n")
                stats_buffer.write(f"* Mean: {df[col].mean()}\n")
                stats_buffer.write(f"* Median: {df[col].median()}\n")
                stats_buffer.write(f"* Standard deviation: {df[col].std()}\n")
            else:
                stats_buffer.write(f"* Unique values: {df[col].nunique()}\n")
                top_values = df[col].value_counts().nlargest(5)
                stats_buffer.write("* Top 5 values:\n")
                for value, count in top_values.items():
                    stats_buffer.write(f"  * {value}: {count} ({count/len(df)*100:.2f}%)\n")
            stats_buffer.write("\n")
        
        zip_file.writestr("dataset_analysis.md", stats_buffer.getvalue())
        
        # Add visualizations
        numeric_cols, categorical_cols, datetime_cols = detect_data_types(df)
        
        # Missing values heatmap
        if df.isna().sum().sum() > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)
            plt.title("Missing Values Heatmap")
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            zip_file.writestr("visualizations/missing_values.png", img_buffer.getvalue())
            plt.close()
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(correlation, dtype=bool))
            sns.heatmap(correlation, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                      square=True, linewidths=0.5)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            zip_file.writestr("visualizations/correlation_heatmap.png", img_buffer.getvalue())
            plt.close()
        
        # Individual column visualizations
        os.makedirs("temp_viz", exist_ok=True)
        
        # Numeric columns
        for col in numeric_cols:
            # Histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            zip_file.writestr(f"visualizations/histogram_{col}.png", img_buffer.getvalue())
            plt.close()
            
            # Box plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=df[col].dropna())
            plt.title(f"Box Plot of {col}")
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            zip_file.writestr(f"visualizations/boxplot_{col}.png", img_buffer.getvalue())
            plt.close()
        
        # Categorical columns
        for col in categorical_cols:
            if df[col].nunique() > 50:
                continue  # Skip columns with too many categories
                
            # Bar chart
            plt.figure(figsize=(12, 6))
            value_counts = df[col].value_counts().nlargest(15)
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f"Frequency of {col} Categories")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            zip_file.writestr(f"visualizations/barchart_{col}.png", img_buffer.getvalue())
            plt.close()
    
    st.download_button(
        label="Download Analysis Report (ZIP)",
        data=buffer.getvalue(),
        file_name="dataset_analysis.zip",
        mime="application/zip"
    )

# Main app logic
if uploaded_file is not None:
    try:
        # Determine file type and read
        file_extension = uploaded_file.name.split('.')[-1]
        
        if file_extension.lower() == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension.lower() in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()
        
        # Check if file is empty
        if df.empty:
            st.error("The uploaded file is empty. Please upload a file with data.")
            st.stop()
        
        # Display success message
        st.success(f"Successfully loaded '{uploaded_file.name}' with {df.shape[0]} rows and {df.shape[1]} columns!")
        
        # Detection of data types
        numeric_cols, categorical_cols, datetime_cols = detect_data_types(df)
        
        # Show tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📝 Summary", 
            "📈 Numeric", 
            "📊 Categorical", 
            "📅 Time Series", 
            "🔄 Relationships",
            "📥 Download"
        ])
        
        with tab1:
            generate_summary(df)
        
        with tab2:
            if numeric_cols:
                visualize_numeric(df, numeric_cols)
            else:
                st.info("No numeric columns found in the dataset.")
        
        with tab3:
            if categorical_cols:
                visualize_categorical(df, categorical_cols)
            else:
                st.info("No categorical columns found in the dataset.")
        
        with tab4:
            if datetime_cols:
                visualize_datetime(df, datetime_cols)
            else:
                st.info("No datetime columns found in the dataset.")
        
        with tab5:
            visualize_relationships(df, numeric_cols, categorical_cols)
        
        with tab6:
            import os
            create_downloadable_report(df)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
else:
    # Example section when no file is uploaded
    st.info("📤 Upload a CSV or Excel file to get started.")
    
    # Show example features
    st.header("✨ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Automatic Visualizations")
        st.markdown("- Histograms and density plots")
        st.markdown("- Box plots and scatter plots")
        st.markdown("- Bar charts and pie charts")
        st.markdown("- Time series analysis")
    
    with col2:
        st.markdown("### 📝 Data Analysis")
        st.markdown("- Statistical summaries")
        st.markdown("- Correlation analysis")
        st.markdown("- Missing value detection")
        st.markdown("- Distribution analysis")
    
    with col3:
        st.markdown("### 📥 Export Options")
        st.markdown("- Download complete analysis")
        st.markdown("- All visualizations included")
        st.markdown("- Comprehensive report")
        st.markdown("- Ready for presentations")
    
    # Example image
    st.image("https://raw.githubusercontent.com/streamlit/example-app-interactive-table/master/screenshot.png", 
            caption="Example visualization (not from your data)")
    
    # Sample datasets section
    st.header("🧪 Don't have a dataset?")
    st.markdown("Try one of these sample datasets:")
    
    sample_datasets = {
        "Iris Flower Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "Titanic Passenger Data": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "Boston Housing Prices": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    }
    
    for name, url in sample_datasets.items():
        st.markdown(f"- [{name}]({url})")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, pandas, matplotlib, and seaborn")

# Run the app with: streamlit run app.py
