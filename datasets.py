import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import io
import base64
from matplotlib.backends.backend_pdf import PdfPages

class EDAReport:
    def __init__(self, df: pd.DataFrame, target: str = None):
        """
        Initialize the EDAReport class.
        
        Args:
            df (pd.DataFrame): The dataframe to analyze.
            target (str, optional): The name of the target column. Defaults to None.
        """
        self.df = df
        self.target = target
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Store report artifacts
        self.report_content = [] # List of dicts: {'type': 'text'|'header'|'table'|'plot', 'content': ...}
        
        # Set professional plot style
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
        plt.rcParams['figure.figsize'] = (12, 6)

    def _add_header(self, text, level=2):
        self.report_content.append({'type': 'header', 'content': text, 'level': level})

    def _add_text(self, text):
        self.report_content.append({'type': 'text', 'content': text})

    def _add_table(self, df, title=None):
        self.report_content.append({'type': 'table', 'content': df, 'title': title})

    def _add_plot(self, fig, title=None):
        self.report_content.append({'type': 'plot', 'content': fig, 'title': title})

    def _get_outliers_iqr(self, series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()

    def basic_info(self):
        """Analyzes basic information and metadata."""
        self._add_header("1. Dataset Overview")
        
        # Shape and Duplicates
        info_df = pd.DataFrame({
            'Metric': ['Rows', 'Columns', 'Duplicates', 'Total Missing Values'],
            'Value': [self.df.shape[0], self.df.shape[1], self.df.duplicated().sum(), self.df.isnull().sum().sum()]
        })
        self._add_table(info_df, "Dataset Statistics")

        self._add_header("2. Column Metadata")
        metadata = []
        for col in self.df.columns:
            col_data = self.df[col]
            dtype = col_data.dtype
            missing = col_data.isnull().sum()
            missing_pct = (missing / len(self.df)) * 100
            unique = col_data.nunique()
            
            meta = {
                'Column': col,
                'Type': str(dtype),
                'Missing': missing,
                'Missing (%)': f"{missing_pct:.2f}%",
                'Unique': unique
            }
            
            if col in self.numerical_cols:
                try:
                    meta['Skewness'] = f"{col_data.skew():.2f}"
                    meta['Kurtosis'] = f"{col_data.kurtosis():.2f}"
                    meta['Outliers (IQR)'] = self._get_outliers_iqr(col_data.dropna())
                except:
                    meta['Skewness'] = '-'
                    meta['Kurtosis'] = '-'
                    meta['Outliers (IQR)'] = '-'
            else:
                meta['Skewness'] = '-'
                meta['Kurtosis'] = '-'
                meta['Outliers (IQR)'] = '-'
                
            metadata.append(meta)
            
        self._add_table(pd.DataFrame(metadata), "Column Metadata")
        self._add_text("Head of Data:")
        self._add_table(self.df.head(), "First 5 Rows")

    def numerical_analysis(self):
        """Analyzes numerical columns."""
        if not self.numerical_cols:
            return

        self._add_header("3. Numerical Feature Analysis")
        desc = self.df[self.numerical_cols].describe().T
        self._add_table(desc, "Descriptive Statistics")
        
        for col in self.numerical_cols:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Analysis of {col}', fontsize=16)
            
            # Histogram with KDE
            try:
                sns.histplot(self.df[col], kde=True, ax=axes[0], color='skyblue')
            except Exception as e:
                sns.histplot(self.df[col], kde=False, ax=axes[0], color='skyblue')
                print(f"Warning: KDE failed for {col}, showing histogram only.")
            axes[0].set_title('Distribution')
            
            # Box Plot
            sns.boxplot(x=self.df[col], ax=axes[1], color='lightgreen')
            axes[1].set_title('Box Plot')
            
            # QQ Plot
            try:
                stats.probplot(self.df[col].dropna(), dist="norm", plot=axes[2])
                axes[2].set_title('Q-Q Plot')
            except:
                axes[2].text(0.5, 0.5, "QQ Plot Failed", ha='center')
            
            plt.tight_layout()
            self._add_plot(fig, f"Numerical Analysis: {col}")
            plt.close(fig)

    def categorical_analysis(self):
        """Analyzes categorical columns."""
        if not self.categorical_cols:
            return

        self._add_header("4. Categorical Feature Analysis")
        for col in self.categorical_cols:
            unique_count = self.df[col].nunique()
            self._add_text(f"**Column: {col}** | Unique Values: {unique_count}")
            
            if unique_count < 20:
                fig = plt.figure(figsize=(10, 5))
                ax = sns.countplot(y=self.df[col], hue=self.df[col], order=self.df[col].value_counts().index, palette='viridis', legend=False)
                plt.title(f'Distribution of {col}')
                for container in ax.containers:
                    ax.bar_label(container)
                plt.tight_layout()
                self._add_plot(fig, f"Categorical Dist: {col}")
                plt.close(fig)
            else:
                top_10 = self.df[col].value_counts().head(10).to_frame(name='Count')
                self._add_table(top_10, f"Top 10 Values for {col}")

    def correlation_analysis(self):
        """Analyzes correlations."""
        if len(self.numerical_cols) < 2:
            return

        self._add_header("5. Correlation Analysis")
        corr = self.df[self.numerical_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        fig = plt.figure(figsize=(12, 10))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Correlation Matrix (Numerical Features)')
        plt.tight_layout()
        self._add_plot(fig, "Correlation Matrix")
        plt.close(fig)

    def target_analysis(self):
        """Analyzes relationship with target."""
        if not self.target:
            self._add_text("No target specified for analysis.")
            return
            
        if self.target not in self.df.columns:
            self._add_text(f"Target column '{self.target}' not found in DataFrame.")
            return

        self._add_header(f"6. Target Analysis: {self.target}")
        
        is_target_numeric = self.target in self.numerical_cols
        
        if is_target_numeric:
            self._add_text(f"Target '{self.target}' is Numerical.")
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(self.df[self.target], kde=True, color='purple')
            plt.title(f'Target Distribution: {self.target}')
            self._add_plot(fig, "Target Distribution")
            plt.close(fig)
            
            for col in self.numerical_cols:
                if col == self.target: continue
                fig = plt.figure(figsize=(10, 6))
                # Sample if data is too large for scatter
                plot_df = self.df.sample(min(10000, len(self.df))) if len(self.df) > 10000 else self.df
                sns.regplot(x=plot_df[col], y=plot_df[self.target], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                plt.title(f'{col} vs {self.target}')
                self._add_plot(fig, f"{col} vs Target")
                plt.close(fig)
                
        else:
            self._add_text(f"Target '{self.target}' is Categorical.")
            fig = plt.figure(figsize=(8, 5))
            sns.countplot(x=self.df[self.target], hue=self.df[self.target], palette='pastel', legend=False)
            plt.title(f'Target Distribution: {self.target}')
            self._add_plot(fig, "Target Distribution")
            plt.close(fig)
            
            for col in self.numerical_cols:
                fig = plt.figure(figsize=(10, 6))
                sns.kdeplot(data=self.df, x=col, hue=self.target, fill=True, common_norm=False, palette='crest')
                plt.title(f'{col} Distribution by {self.target}')
                self._add_plot(fig, f"{col} by Target")
                plt.close(fig)

    def generate_full_report(self, save_html=None, save_pdf=None):
        """Generates the full EDA report.
        
        Args:
            save_html (str, optional): Path to save HTML report.
            save_pdf (str, optional): Path to save PDF report.
            
        If neither save_html nor save_pdf is provided, the report is displayed in the notebook.
        """
        self.report_content = [] # Reset
        self.basic_info()
        self.numerical_analysis()
        self.categorical_analysis()
        self.correlation_analysis()
        self.target_analysis()
        
        if save_html:
            self.save_html(save_html)
        if save_pdf:
            self.save_pdf(save_pdf)
            
        if not save_html and not save_pdf:
            self.show_notebook()

    def show_notebook(self):
        """Displays the report in a Jupyter Notebook."""
        for item in self.report_content:
            if item['type'] == 'header':
                display(Markdown(f"{'#' * item['level']} {item['content']}"))
            elif item['type'] == 'text':
                display(Markdown(item['content']))
            elif item['type'] == 'table':
                if item.get('title'):
                    display(Markdown(f"**{item['title']}**"))
                display(item['content'])
            elif item['type'] == 'plot':
                if item.get('title'):
                    display(Markdown(f"**{item['title']}**"))
                display(item['content'])


    def _generate_html_content(self, as_pdf=False):
        """Generates the HTML content for the report."""
        
        # CSS adjustments based on output format
        if as_pdf:
            table_css = "table { width: 100%; border: 1px solid #ddd; font-size: 10px; } th, td { padding: 5px; border: 1px solid #ddd; }"
            container_css = ""
            # PDF specific page CSS
            extra_css = """
                @page { size: A4; margin: 1cm; }
                .plot-container img { width: 100%; height: auto; }
            """
        else:
            table_css = """
                table { border-collapse: collapse; width: 100%; margin: 0; font-size: 0.9em; }
                th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #3498db; color: white; position: sticky; top: 0; z-index: 1; }
                tr:hover { background-color: #f1f1f1; }
            """
            container_css = ".table-container { overflow-x: auto; max-height: 600px; overflow-y: auto; border: 1px solid #eee; margin: 20px 0; }"
            extra_css = ""

        html_content = [f"""
        <html>
        <head>
            <title>EDA Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f5f5; color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 0 20px rgba(0,0,0,0.1); border-radius: 8px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                h1 {{ text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 20px; }}
                h2 {{ border-left: 5px solid #3498db; padding-left: 15px; margin-top: 40px; }}
                {table_css}
                img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #eee; border-radius: 4px; }}
                .plot-container {{ text-align: center; margin: 30px 0; }}
                .text-content {{ line-height: 1.6; }}
                {container_css}
                {extra_css}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Exploratory Data Analysis Report</h1>
        """]
        
        for item in self.report_content:
            if item['type'] == 'header':
                html_content.append(f"<h{item['level']}>{item['content']}</h{item['level']}>")
            elif item['type'] == 'text':
                # Simple markdown-like bold parsing
                text = item['content'].replace('**', '<b>').replace('**', '</b>')
                html_content.append(f"<div class='text-content'><p>{text}</p></div>")
            elif item['type'] == 'table':
                if item.get('title'):
                    html_content.append(f"<h3>{item['title']}</h3>")
                
                table_html = item['content'].to_html(classes='table', border=0)
                if not as_pdf:
                    html_content.append(f"<div class='table-container'>{table_html}</div>")
                else:
                    html_content.append(table_html)
                    
            elif item['type'] == 'plot':
                if item.get('title'):
                    html_content.append(f"<h3>{item['title']}</h3>")
                
                # Convert plot to base64
                buf = io.BytesIO()
                item['content'].savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                html_content.append(f"<div class='plot-container'><img src='data:image/png;base64,{img_str}'/></div>")
                buf.close()
                
        html_content.append("</div></body></html>")
        return '\n'.join(html_content)

    def save_html(self, filename):
        """Saves the report as a standalone HTML file."""
        html_content = self._generate_html_content(as_pdf=False)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML Report saved to {filename}")

    def save_pdf(self, filename):
        """Saves the report as a PDF file using xhtml2pdf."""
        from xhtml2pdf import pisa
        
        print(f"Generating PDF report to {filename}...")
        html_content = self._generate_html_content(as_pdf=True)
        
        with open(filename, "wb") as f:
            pisa_status = pisa.CreatePDF(html_content, dest=f)
            
        if pisa_status.err:
            print(f"Error generating PDF: {pisa_status.err}")
        else:
            print(f"PDF Report saved to {filename}")


