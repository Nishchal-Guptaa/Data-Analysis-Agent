import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
# from dotenv import load_dotenv

# env_path = r"C:\Users\nishc\Desktop\Projects\Python\.env"
# load_dotenv(env_path)

# Alternative imports to avoid conflicts
try:
    import pdfplumber  # Alternative to PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx2txt  # Simpler alternative to python-docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
    # You may need to set the tesseract path on Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    OCR_AVAILABLE = False

import openpyxl
from together import Together
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import base64
import json
import re

class DocumentAnalysisAgent:
    def __init__(self, together_api_key: str):
        """Initialize the agent with Together.ai API key"""
        self.client = Together(api_key=together_api_key)
        self.model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        self.conversation_history = []
        self.current_data = None
        self.file_metadata = {}
        self.last_processed_content = ""
        
    def upload_and_process_file(self, uploaded_file) -> Dict[str, Any]:
        """Process uploaded file based on its type"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'csv':
                return self._process_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                return self._process_excel(uploaded_file)
            elif file_extension == 'txt':
                result = self._process_text(uploaded_file)
                if result.get("success"):
                    self.last_processed_content = result.get("content", "")
                return result
            elif file_extension in ['doc', 'docx']:
                result = self._process_word(uploaded_file)
                if result.get("success"):
                    self.last_processed_content = result.get("content", "")
                return result
            elif file_extension == 'pdf':
                result = self._process_pdf(uploaded_file)
                if result.get("success"):
                    self.last_processed_content = result.get("content", "")
                return result
            elif file_extension in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp']:
                result = self._process_image(uploaded_file)
                # Store extracted text for context
                if result.get("success") and result.get("extracted_text"):
                    self.last_processed_content = result["extracted_text"]
                    # If we extracted text that looks like structured data, try to parse it
                    structured_data = self._process_image_as_data(result["extracted_text"])
                    if structured_data is not None:
                        self.current_data = structured_data
                        result["data"] = structured_data
                        result["analysis"]["structured_data_detected"] = True
                        result["analysis"]["data_shape"] = structured_data.shape
                        result["analysis"]["data_columns"] = structured_data.columns.tolist()
                return result
            else:
                return {"error": f"Unsupported file type: {file_extension}"}
                
        except Exception as e:
            return {"error": f"Error processing file: {str(e)}"}
    
    def _process_csv(self, file) -> Dict[str, Any]:
        """Process CSV files"""
        df = pd.read_csv(file)
        self.current_data = df
        
        analysis = {
            "file_type": "CSV",
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "summary_stats": df.describe().to_dict(),
            "sample_data": df.to_dict() if len(df) <= 50 else df.head(10).to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        return {"success": True, "analysis": analysis, "data": df}
    
    def _process_excel(self, file) -> Dict[str, Any]:
        """Process Excel files"""
        df = pd.read_excel(file)
        self.current_data = df
        
        analysis = {
            "file_type": "Excel",
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "summary_stats": df.describe().to_dict(),
            "sample_data": df.head().to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        return {"success": True, "analysis": analysis, "data": df}
    
    def _process_text(self, file) -> Dict[str, Any]:
        """Process text files"""
        content = file.read().decode('utf-8')
        
        analysis = {
            "file_type": "Text",
            "character_count": len(content),
            "word_count": len(content.split()),
            "line_count": len(content.split('\n')),
            "content_preview": content[:500] + "..." if len(content) > 500 else content
        }
        
        return {"success": True, "analysis": analysis, "content": content}
    
    def _process_word(self, file) -> Dict[str, Any]:
        """Process Word documents using docx2txt"""
        if not DOCX_AVAILABLE:
            return {"error": "docx2txt package not available. Please install it with: pip install docx2txt"}
        
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            
            # Extract text
            content = docx2txt.process(temp_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
            analysis = {
                "file_type": "Word Document",
                "character_count": len(content),
                "word_count": len(content.split()),
                "line_count": len(content.split('\n')),
                "content_preview": content[:500] + "..." if len(content) > 500 else content
            }
            
            return {"success": True, "analysis": analysis, "content": content}
        except Exception as e:
            return {"error": f"Error processing Word document: {str(e)}"}
    
    def _process_pdf(self, file) -> Dict[str, Any]:
        """Process PDF files using pdfplumber"""
        if not PDF_AVAILABLE:
            return {"error": "pdfplumber package not available. Please install it with: pip install pdfplumber"}
        
        try:
            content = ""
            page_count = 0
            
            with pdfplumber.open(file) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
            
            analysis = {
                "file_type": "PDF",
                "page_count": page_count,
                "character_count": len(content),
                "word_count": len(content.split()),
                "content_preview": content[:500] + "..." if len(content) > 500 else content
            }
            
            return {"success": True, "analysis": analysis, "content": content}
        except Exception as e:
            return {"error": f"Error processing PDF: {str(e)}"}
    
    def _process_image(self, file) -> Dict[str, Any]:
        """Process image files with OCR text extraction"""
        image = Image.open(file)
        
        # Basic image analysis
        analysis = {
            "file_type": "Image",
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height
        }
        
        # Extract text using OCR if available
        extracted_text = ""
        if OCR_AVAILABLE:
            try:
                # Extract text from image
                extracted_text = pytesseract.image_to_string(image)
                
                # Add text analysis to results
                if extracted_text.strip():
                    analysis.update({
                        "extracted_text_length": len(extracted_text),
                        "extracted_word_count": len(extracted_text.split()),
                        "extracted_line_count": len(extracted_text.split('\n')),
                        "text_preview": extracted_text[:300] + "..." if len(extracted_text) > 300 else extracted_text
                    })
                    
                    # Try to detect if the text contains structured data (like tables)
                    lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
                    potential_table_data = self._detect_table_structure(lines)
                    
                    if potential_table_data:
                        analysis["potential_table_detected"] = True
                        analysis["table_preview"] = potential_table_data
                else:
                    analysis["extracted_text_note"] = "No readable text detected in image"
            except Exception as e:
                analysis["ocr_error"] = f"OCR processing failed: {str(e)}"
        else:
            analysis["ocr_note"] = "OCR not available. Install pytesseract for text extraction."
        
        # Convert image to base64 for display/analysis
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True, 
            "analysis": analysis, 
            "image": image, 
            "base64": img_base64,
            "extracted_text": extracted_text,
            "content": extracted_text  # Treat extracted text as content for analysis
        }
    
    def _detect_table_structure(self, lines: List[str]) -> Optional[Dict]:
        """Detect if extracted text contains tabular data"""
        if len(lines) < 2:
            return None
            
        # Look for common table patterns
        separators = ['\t', '|', ',', ';', '  ']  # Common column separators
        
        for sep in separators:
            # Check if multiple lines have the same number of separators
            split_counts = []
            valid_lines = []
            
            for line in lines[:10]:  # Check first 10 lines
                if sep in line:
                    parts = [part.strip() for part in line.split(sep) if part.strip()]
                    if len(parts) > 1:  # Must have at least 2 columns
                        split_counts.append(len(parts))
                        valid_lines.append(parts)
            
            # If we have consistent column counts, it might be a table
            if len(split_counts) >= 2 and len(set(split_counts)) == 1:  # Same column count
                try:
                    # Try to create a simple DataFrame preview
                    df_preview = pd.DataFrame(valid_lines[1:], columns=valid_lines[0])
                    return {
                        "separator": sep,
                        "columns": len(valid_lines[0]),
                        "rows": len(valid_lines) - 1,
                        "headers": valid_lines[0],
                        "sample_data": df_preview.head(3).to_dict()
                    }
                except:
                    continue
        
        return None
    
    def _process_image_as_data(self, extracted_text: str) -> Optional[pd.DataFrame]:
        """Try to convert extracted text to structured data"""
        if not extracted_text.strip():
            return None
            
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        
        # Try different parsing approaches
        separators = ['\t', '|', ',', ';', '  ']
        
        for sep in separators:
            try:
                # Check if it looks like CSV-style data
                data_lines = []
                for line in lines:
                    if sep in line:
                        parts = [part.strip() for part in line.split(sep) if part.strip()]
                        if len(parts) > 1:
                            data_lines.append(parts)
                
                if len(data_lines) >= 2:  # At least header + 1 data row
                    # Try to create DataFrame
                    headers = data_lines[0]
                    data_rows = data_lines[1:]
                    
                    # Ensure all rows have same length as headers
                    filtered_rows = []
                    for row in data_rows:
                        if len(row) == len(headers):
                            filtered_rows.append(row)
                    
                    if filtered_rows:
                        df = pd.DataFrame(filtered_rows, columns=headers)
                        # Try to convert numeric columns
                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        return df
                        
            except Exception:
                continue
        
        return None
    
    def analyze_with_llama(self, query: str, context: str = None) -> str:
        """Send query to Llama model for analysis"""
        
        # Prepare context from current data
        if context is None:
            if self.current_data is not None and isinstance(self.current_data, pd.DataFrame):
    # For small datasets, include all data
                if len(self.current_data) <= 20:
                    sample_data = self.current_data.to_string()
                else:
                    sample_data = self.current_data.head(10).to_string()
                
                context = f"""
                Data Summary:
                - Shape: {self.current_data.shape}
                - Columns: {', '.join(self.current_data.columns)}
                - Data:\n{sample_data}
                - Data types:\n{self.current_data.dtypes.to_string()}
                """
            elif hasattr(self, 'last_processed_content') and self.last_processed_content:
                # For text-based content (including OCR extracted text)
                context = f"Content preview:\n{self.last_processed_content[:1000]}"
        
        # Build conversation prompt
        messages = [
            {
                "role": "system",
                "content": """You are an expert data analyst. Analyze the provided data and answer questions accurately. 
                When suggesting visualizations, be specific about chart types, variables to plot, and insights to highlight.
                Provide clear, actionable insights."""
            }
        ]
        
        # Add conversation history
        for msg in self.conversation_history[-5:]:  # Keep last 5 exchanges
            messages.append(msg)
        
        # Add current query with context
        user_message = f"Context:\n{context}\n\nQuery: {query}" if context else query
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return answer
            
        except Exception as e:
            return f"Error querying Llama model: {str(e)}"
    
    def create_visualization(self, viz_type: str, **kwargs) -> Optional[go.Figure]:
        """Create visualizations based on current data"""
        if self.current_data is None or not isinstance(self.current_data, pd.DataFrame):
            return None
        
        df = self.current_data
        
        try:
            if viz_type == "histogram":
                column = kwargs.get('column')
                if column and column in df.columns:
                    fig = px.histogram(df, x=column, title=f"Distribution of {column}")
                    return fig
                    
            elif viz_type == "scatter":
                x_col = kwargs.get('x_column')
                y_col = kwargs.get('y_column')
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                    return fig
                    
            elif viz_type == "line":
                x_col = kwargs.get('x_column')
                y_col = kwargs.get('y_column')
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                    return fig
                    
            elif viz_type == "bar":
                x_col = kwargs.get('x_column')
                y_col = kwargs.get('y_column')
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    return fig
                    
            elif viz_type == "correlation_heatmap":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, 
                                  title="Correlation Heatmap",
                                  color_continuous_scale="RdBu_r")
                    return fig
                    
            elif viz_type == "box":
                column = kwargs.get('column')
                if column and column in df.columns:
                    fig = px.box(df, y=column, title=f"Box Plot of {column}")
                    return fig
                    
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            return None
        
        return None


def main():
    """Streamlit app interface"""
    st.set_page_config(page_title="Document Analysis Agent", layout="wide", page_icon="🤖")
    
    st.title("🤖 Document Analysis Agent")
    st.subheader("Upload documents, ask questions, and create visualizations")
    
    # Check for missing packages
    missing_packages = []
    if not PDF_AVAILABLE:
        missing_packages.append("pdfplumber")
    if not DOCX_AVAILABLE:
        missing_packages.append("docx2txt")
    if not OCR_AVAILABLE:
        missing_packages.append("pytesseract")
    
    if missing_packages:
        st.warning(f"Missing packages: {', '.join(missing_packages)}")
        install_cmd = "pip install " + " ".join(missing_packages)
        st.code(install_cmd)
        
        if not OCR_AVAILABLE:
            st.info("""
            📝 **For OCR functionality (text extraction from images):**
            1. Install pytesseract: `pip install pytesseract`
            2. Install Tesseract OCR engine:
               - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
               - **macOS**: `brew install tesseract`
               - **Linux**: `sudo apt-get install tesseract-ocr`
            """)
    
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    
    # Sidebar for API key and file upload
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.secrets["TOGETHER_API_KEY"]
        
        if api_key:
            if st.session_state.agent is None:
                st.session_state.agent = DocumentAnalysisAgent(api_key)
                st.success("Agent initialized!")
        
        st.header("File Upload")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'txt', 'doc', 'docx', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'],
            help="Supports data files, documents, and images (with OCR text extraction)"
        )
        
        if uploaded_file and st.session_state.agent:
            if st.button("Process File"):
                with st.spinner("Processing file..."):
                    result = st.session_state.agent.upload_and_process_file(uploaded_file)
                    st.session_state.processed_file = result
                    
                if result.get("success"):
                    st.success("File processed successfully!")
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    # Main content area
    if st.session_state.processed_file and st.session_state.processed_file.get("success"):
        
        # Display file analysis
        st.header("📊 File Analysis")
        analysis = st.session_state.processed_file["analysis"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json(analysis)
        
        with col2:
            # Show data preview for structured data
            if "data" in st.session_state.processed_file:
                st.subheader("Data Preview")
                st.dataframe(st.session_state.processed_file["data"])
            elif "image" in st.session_state.processed_file:
                st.subheader("Image Preview")
                st.image(st.session_state.processed_file["image"], caption="Uploaded Image", use_container_width=True)
                
                # Show extracted text if available
                if st.session_state.processed_file.get("extracted_text"):
                    st.subheader("Extracted Text (OCR)")
                    with st.expander("View extracted text"):
                        st.text(st.session_state.processed_file["extracted_text"])
        
        # Query interface
        st.header("💬 Ask Questions")
        
        # Predefined questions based on file type
        file_type = analysis.get("file_type", "")
        
        if file_type in ["CSV", "Excel"] or analysis.get("structured_data_detected"):
            st.subheader("Quick Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Summarize Data"):
                    if analysis.get("structured_data_detected"):
                        response = st.session_state.agent.analyze_with_llama("This data was extracted from an image using OCR. Provide a comprehensive summary of this dataset including key statistics, patterns, and insights.")
                    else:
                        response = st.session_state.agent.analyze_with_llama("Provide a comprehensive summary of this dataset including key statistics, patterns, and insights.")
                    st.write(response)
            
            with col2:
                if st.button("Find Correlations"):
                    response = st.session_state.agent.analyze_with_llama("Identify and explain the strongest correlations in this dataset.")
                    st.write(response)
            
            with col3:
                if st.button("Suggest Visualizations"):
                    response = st.session_state.agent.analyze_with_llama("Suggest the most appropriate visualizations for this dataset and explain why.")
                    st.write(response)
        
        elif file_type == "Image":
            st.subheader("Image Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Analyze Image Content"):
                    if st.session_state.processed_file.get("extracted_text"):
                        response = st.session_state.agent.analyze_with_llama("Analyze the text content extracted from this image. What type of document or information does it contain?")
                    else:
                        response = st.session_state.agent.analyze_with_llama("Describe what you can determine about this image based on its technical properties.")
                    st.write(response)
            
            with col2:
                if st.button("Extract Key Information") and st.session_state.processed_file.get("extracted_text"):
                    response = st.session_state.agent.analyze_with_llama("Extract and summarize the key information, data points, or insights from the text extracted from this image.")
                    st.write(response)
            
            with col3:
                if st.button("Identify Structure") and st.session_state.processed_file.get("extracted_text"):
                    response = st.session_state.agent.analyze_with_llama("Analyze the structure of the text extracted from this image. Does it contain tables, lists, or other structured information?")
                    st.write(response)
        
        # Custom query input
        user_query = st.text_area("Ask a question about your data:")
        
        if st.button("Analyze") and user_query:
            with st.spinner("Analyzing..."):
                response = st.session_state.agent.analyze_with_llama(user_query)
                st.write("**Response:**")
                st.write(response)
        
        # Visualization section
        if "data" in st.session_state.processed_file:
            st.header("📈 Create Visualizations")
            
            df = st.session_state.processed_file["data"]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()
            
            viz_type = st.selectbox(
                "Select visualization type:",
                ["histogram", "scatter", "line", "bar", "correlation_heatmap", "box"]
            )
            
            col1, col2 = st.columns(2)
            
            if viz_type in ["scatter", "line", "bar"]:
                with col1:
                    x_col = st.selectbox("X-axis column:", all_cols)
                with col2:
                    y_col = st.selectbox("Y-axis column:", numeric_cols)
                
                if st.button("Create Visualization"):
                    fig = st.session_state.agent.create_visualization(
                        viz_type, x_column=x_col, y_column=y_col
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
            elif viz_type in ["histogram", "box"]:
                column = st.selectbox("Select column:", numeric_cols)
                
                if st.button("Create Visualization"):
                    fig = st.session_state.agent.create_visualization(viz_type, column=column)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
            elif viz_type == "correlation_heatmap":
                if st.button("Create Correlation Heatmap"):
                    fig = st.session_state.agent.create_visualization(viz_type)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        
        # Conversation history
        if st.session_state.agent.conversation_history:
            st.header("💭 Conversation History")
            for i, msg in enumerate(st.session_state.agent.conversation_history):
                if msg["role"] == "user":
                    st.write(f"**You:** {msg['content']}")
                else:
                    st.write(f"**Agent:** {msg['content']}")
                st.write("---")
    
    else:
        st.info("👆 Please configure your API key and upload a file to get started!")


if __name__ == "__main__":
    main()