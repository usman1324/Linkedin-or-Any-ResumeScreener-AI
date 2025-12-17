"""
ResumeScreener AI - Fixed & Enhanced Version
NLP + ML Resume Ranking System with Beautiful UI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from wordcloud import WordCloud
import PyPDF2
import re
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Check and download NLTK data
try:
    import nltk

    nltk.data.find('tokenizers/punkt_tab')
except:
    try:
        nltk.download('punkt_tab')
    except:
        pass


class ResumeScreenerAI:
    def __init__(self, root):
        self.root = root
        self.root.title("✨ ResumeScreener AI - Intelligent Resume Ranking")
        self.root.geometry("1400x850")  # Slightly smaller for better fit
        self.root.configure(bg='#1a1a2e')

        # Make window resizable
        self.root.minsize(1200, 700)

        # Initialize data
        self.resumes = pd.DataFrame(
            columns=['filename', 'candidate_name', 'content', 'processed_text', 'score', 'filepath', 'status'])
        self.job_description = ""

        # Setup styles first
        self.setup_styles()

        # Initialize NLP after UI setup
        self.nlp_loaded = False
        self.init_nlp_async()

        # Create GUI
        self.create_widgets()

    def setup_styles(self):
        """Setup modern gradient styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Colors - Modern gradient palette
        self.bg_color = '#1a1a2e'
        self.card_bg = '#16213e'
        self.accent_color = '#00d4ff'
        self.accent_gradient = ['#00d4ff', '#0099ff']
        self.primary_color = '#0f3460'
        self.success_color = '#00ff88'
        self.warning_color = '#ffaa00'
        self.danger_color = '#ff4d4d'
        self.text_color = '#ffffff'
        self.text_secondary = '#b0b0b0'

        # Configure custom styles
        self.style.configure('Title.TLabel',
                             font=('Segoe UI', 24, 'bold'),
                             background=self.bg_color,
                             foreground=self.accent_color)

        self.style.configure('Card.TFrame',
                             background=self.card_bg,
                             relief='flat',
                             borderwidth=0)

        self.style.configure('Modern.TButton',
                             font=('Segoe UI', 10, 'bold'),
                             padding=10,
                             background=self.accent_color,
                             foreground='white',
                             borderwidth=0)

        self.style.map('Modern.TButton',
                       background=[('active', '#0099ff')])

    def init_nlp_async(self):
        """Initialize NLP in background to avoid blocking UI"""
        try:
            # Try to import nltk
            import nltk
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize

            # Download required NLTK data
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)

            self.stop_words = set(stopwords.words('english'))
            self.word_tokenize = word_tokenize

            # Custom stopwords
            self.custom_stopwords = {
                'experience', 'work', 'job', 'position', 'role', 'company',
                'university', 'college', 'school', 'degree', 'skills', 'skill',
                'projects', 'project', 'certification', 'certificate', 'looking',
                'seeking', 'available', 'email', 'phone', 'contact', 'linkedin',
                'github', 'portfolio', 'resume', 'cv', 'curriculum', 'vitae'
            }

            self.nlp_loaded = True
            print("NLP initialized successfully")

        except Exception as e:
            print(f"NLTK initialization failed: {e}")
            # Create fallback tokenizer
            self.stop_words = set()
            self.custom_stopwords = set()
            self.word_tokenize = lambda x: x.split()
            self.nlp_loaded = False

    def create_widgets(self):
        """Create the main GUI layout with modern design"""
        # Main container with gradient effect
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title with gradient effect
        title_frame = tk.Frame(main_container, bg=self.bg_color)
        title_frame.pack(fill=tk.X, pady=(0, 20))

        title_label = tk.Label(title_frame,
                               text="✨ ResumeScreener AI",
                               font=('Segoe UI', 32, 'bold'),
                               bg=self.bg_color,
                               fg=self.accent_color)
        title_label.pack(side=tk.LEFT)

        subtitle_label = tk.Label(title_frame,
                                  text="AI-Powered Resume Screening • NLP + ML Engine",
                                  font=('Segoe UI', 12),
                                  bg=self.bg_color,
                                  fg=self.text_secondary)
        subtitle_label.pack(side=tk.LEFT, padx=(20, 0), pady=(10, 0))

        # Status indicator
        self.status_indicator = tk.Label(title_frame,
                                         text="●",
                                         font=('Segoe UI', 12),
                                         bg=self.bg_color,
                                         fg=self.success_color if self.nlp_loaded else self.warning_color)
        self.status_indicator.pack(side=tk.RIGHT, padx=(0, 10))

        self.status_label = tk.Label(title_frame,
                                     text="Ready" if self.nlp_loaded else "NLP Initializing...",
                                     font=('Segoe UI', 10),
                                     bg=self.bg_color,
                                     fg=self.text_secondary)
        self.status_label.pack(side=tk.RIGHT)

        # Main content area
        content_frame = tk.Frame(main_container, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left Panel - Input Section
        left_panel = tk.Frame(content_frame, bg=self.card_bg, relief='flat', bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Job Description Card
        jd_card = tk.Frame(left_panel, bg=self.card_bg, relief='flat', bd=0)
        jd_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Card header
        jd_header = tk.Frame(jd_card, bg=self.primary_color, height=40)
        jd_header.pack(fill=tk.X)
        jd_header.pack_propagate(False)

        tk.Label(jd_header,
                 text="📋 Job Description",
                 font=('Segoe UI', 12, 'bold'),
                 bg=self.primary_color,
                 fg='white').pack(side=tk.LEFT, padx=15, pady=10)

        # JD Content
        jd_content = tk.Frame(jd_card, bg='#1e2a47', padx=15, pady=15)
        jd_content.pack(fill=tk.BOTH, expand=True)

        self.jd_text = scrolledtext.ScrolledText(jd_content,
                                                 height=12,
                                                 font=('Segoe UI', 10),
                                                 bg='#2d3748',
                                                 fg='white',
                                                 wrap=tk.WORD,
                                                 insertbackground='white',
                                                 relief='flat')
        self.jd_text.pack(fill=tk.BOTH, expand=True)

        # Load sample JD
        sample_jd = """Software Engineer - Machine Learning

Responsibilities:
• Design and implement machine learning models for real-world applications
• Develop and deploy scalable ML pipelines using Python, TensorFlow, PyTorch
• Collaborate with data scientists and engineers to productionize models
• Optimize algorithms for performance and scalability
• Implement MLOps practices and CI/CD pipelines

Requirements:
• Bachelor's/Master's in Computer Science or related field
• 3+ years experience in software development with Python
• Strong knowledge of machine learning algorithms and frameworks
• Experience with cloud platforms (AWS, GCP, Azure)
• Proficiency in SQL and NoSQL databases
• Experience with Docker, Kubernetes, and microservices architecture
• Excellent problem-solving and communication skills"""

        self.jd_text.insert('1.0', sample_jd)

        # JD Controls
        jd_controls = tk.Frame(jd_content, bg='#1e2a47')
        jd_controls.pack(fill=tk.X, pady=(10, 0))

        tk.Button(jd_controls,
                  text="📁 Load JD",
                  command=self.load_job_description,
                  font=('Segoe UI', 9),
                  bg=self.accent_color,
                  fg='white',
                  relief='flat',
                  padx=15,
                  pady=5).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(jd_controls,
                  text="💾 Save JD",
                  command=self.save_job_description,
                  font=('Segoe UI', 9),
                  bg=self.primary_color,
                  fg='white',
                  relief='flat',
                  padx=15,
                  pady=5).pack(side=tk.LEFT)

        # Resume Upload Card
        upload_card = tk.Frame(left_panel, bg=self.card_bg, relief='flat', bd=0)
        upload_card.pack(fill=tk.BOTH, expand=True)

        # Card header
        upload_header = tk.Frame(upload_card, bg=self.primary_color, height=40)
        upload_header.pack(fill=tk.X)
        upload_header.pack_propagate(False)

        tk.Label(upload_header,
                 text="📤 Upload Resumes",
                 font=('Segoe UI', 12, 'bold'),
                 bg=self.primary_color,
                 fg='white').pack(side=tk.LEFT, padx=15, pady=10)

        # Upload Content
        upload_content = tk.Frame(upload_card, bg='#1e2a47', padx=15, pady=15)
        upload_content.pack(fill=tk.BOTH, expand=True)

        # Upload buttons with icons
        btn_frame = tk.Frame(upload_content, bg='#1e2a47')
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Button(btn_frame,
                  text="📄 Add Single Resume",
                  command=self.add_single_resume,
                  font=('Segoe UI', 9, 'bold'),
                  bg=self.accent_color,
                  fg='white',
                  relief='flat',
                  padx=20,
                  pady=8).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(btn_frame,
                  text="📁 Add Folder",
                  command=self.add_resume_folder,
                  font=('Segoe UI', 9),
                  bg=self.primary_color,
                  fg='white',
                  relief='flat',
                  padx=20,
                  pady=8).pack(side=tk.LEFT)

        # Resume list with count
        list_header = tk.Frame(upload_content, bg='#1e2a47')
        list_header.pack(fill=tk.X, pady=(0, 5))

        tk.Label(list_header,
                 text="Selected Resumes:",
                 font=('Segoe UI', 10),
                 bg='#1e2a47',
                 fg=self.text_secondary).pack(side=tk.LEFT)

        self.resume_count_label = tk.Label(list_header,
                                           text="0",
                                           font=('Segoe UI', 10, 'bold'),
                                           bg='#1e2a47',
                                           fg=self.accent_color)
        self.resume_count_label.pack(side=tk.LEFT, padx=(5, 0))

        # Listbox with modern scrollbar
        list_frame = tk.Frame(upload_content, bg='#2d3748')
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Custom scrollbar
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.resume_listbox = tk.Listbox(list_frame,
                                         yscrollcommand=scrollbar.set,
                                         font=('Segoe UI', 10),
                                         bg='#2d3748',
                                         fg='white',
                                         selectbackground=self.accent_color,
                                         selectforeground='white',
                                         borderwidth=0,
                                         highlightthickness=0)
        self.resume_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.resume_listbox.yview)

        # Process Controls
        process_frame = tk.Frame(upload_content, bg='#1e2a47')
        process_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Button(process_frame,
                  text="🚀 Start Screening",
                  command=self.screen_resumes,
                  font=('Segoe UI', 10, 'bold'),
                  bg=self.success_color,
                  fg='white',
                  relief='flat',
                  padx=30,
                  pady=10).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(process_frame,
                  text="🗑️ Clear All",
                  command=self.clear_all,
                  font=('Segoe UI', 10),
                  bg=self.danger_color,
                  fg='white',
                  relief='flat',
                  padx=20,
                  pady=10).pack(side=tk.LEFT)

        # Right Panel - Results
        right_panel = tk.Frame(content_frame, bg=self.card_bg, relief='flat', bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Results Card
        results_card = tk.Frame(right_panel, bg=self.card_bg, relief='flat', bd=0)
        results_card.pack(fill=tk.BOTH, expand=True)

        # Card header
        results_header = tk.Frame(results_card, bg=self.primary_color, height=40)
        results_header.pack(fill=tk.X)
        results_header.pack_propagate(False)

        tk.Label(results_header,
                 text="🏆 Screening Results",
                 font=('Segoe UI', 12, 'bold'),
                 bg=self.primary_color,
                 fg='white').pack(side=tk.LEFT, padx=15, pady=10)

        # Results content
        results_content = tk.Frame(results_card, bg='#1e2a47', padx=15, pady=15)
        results_content.pack(fill=tk.BOTH, expand=True)

        # Create treeview with custom style
        style = ttk.Style()
        style.configure("Custom.Treeview",
                        background="#2d3748",
                        foreground="white",
                        fieldbackground="#2d3748",
                        borderwidth=0)
        style.configure("Custom.Treeview.Heading",
                        background=self.primary_color,
                        foreground="white",
                        relief="flat")
        style.map("Custom.Treeview.Heading",
                  background=[('active', self.primary_color)])

        columns = ('Rank', 'Candidate', 'Score', 'Status')

        self.results_tree = ttk.Treeview(results_content,
                                         columns=columns,
                                         show='headings',
                                         style="Custom.Treeview",
                                         height=20)

        # Define headings
        self.results_tree.heading('Rank', text='Rank')
        self.results_tree.heading('Candidate', text='Candidate')
        self.results_tree.heading('Score', text='Score')
        self.results_tree.heading('Status', text='Status')

        # Define columns
        self.results_tree.column('Rank', width=60, anchor=tk.CENTER)
        self.results_tree.column('Candidate', width=250)
        self.results_tree.column('Score', width=100, anchor=tk.CENTER)
        self.results_tree.column('Status', width=120, anchor=tk.CENTER)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_content,
                                  orient=tk.VERTICAL,
                                  command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        # Pack treeview and scrollbar
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection
        self.results_tree.bind('<<TreeviewSelect>>', self.on_resume_select)

        # Bottom controls
        bottom_frame = tk.Frame(main_container, bg=self.bg_color)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        # Status bar with progress
        self.status_bar = tk.Label(bottom_frame,
                                   text="✨ Ready to screen resumes. Load JD and add resumes.",
                                   bg='#0f3460',
                                   fg='white',
                                   font=('Segoe UI', 10),
                                   anchor=tk.W,
                                   padx=15,
                                   pady=10,
                                   relief='flat')
        self.status_bar.pack(fill=tk.X)

        # Analytics button
        tk.Button(bottom_frame,
                  text="📊 View Analytics",
                  command=self.show_analytics,
                  font=('Segoe UI', 9),
                  bg=self.warning_color,
                  fg='white',
                  relief='flat',
                  padx=20,
                  pady=5).pack(side=tk.RIGHT, padx=(10, 0))

        # Export button
        tk.Button(bottom_frame,
                  text="💾 Export Results",
                  command=self.export_to_excel,
                  font=('Segoe UI', 9),
                  bg=self.accent_color,
                  fg='white',
                  relief='flat',
                  padx=20,
                  pady=5).pack(side=tk.RIGHT)

    def preprocess_text(self, text):
        """Preprocess text with fallback if NLTK not available"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs, emails, phone numbers
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', '', text)

        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text)

        if self.nlp_loaded:
            # Use NLTK tokenization
            tokens = self.word_tokenize(text)

            # Filter stopwords
            tokens = [word for word in tokens
                      if word not in self.stop_words
                      and word not in self.custom_stopwords
                      and len(word) > 2]
        else:
            # Simple fallback tokenization
            tokens = text.split()
            tokens = [word for word in tokens
                      if word not in self.custom_stopwords
                      and len(word) > 2]

        return ' '.join(tokens)

    def extract_text_from_file(self, filepath):
        """Extract text from different file formats"""
        try:
            if filepath.endswith('.pdf'):
                return self.extract_text_from_pdf(filepath)
            elif filepath.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                # Try docx if available
                try:
                    import docx2txt
                    return docx2txt.process(filepath)
                except:
                    return ""
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return ""

    def extract_text_from_pdf(self, filepath):
        """Extract text from PDF file"""
        text = ""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e:
            print(f"PDF extraction error: {e}")
        return text

    def extract_keywords(self, text, top_n=20):
        """Extract keywords from text"""
        if not text:
            return []

        processed = self.preprocess_text(text)
        tokens = processed.split() if processed else []

        # Simple frequency counting
        from collections import Counter
        freq_dist = Counter(tokens)

        return [word for word, freq in freq_dist.most_common(top_n)]

    def calculate_similarity(self, jd_text, resume_text):
        """Calculate similarity between job description and resume"""
        # Preprocess texts
        jd_processed = self.preprocess_text(jd_text)
        resume_processed = self.preprocess_text(resume_text)

        if not jd_processed or not resume_processed:
            return 0.0

        # Simple cosine similarity with TF-IDF
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([jd_processed, resume_processed])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            # Normalize to 0-100
            score = min(max(similarity * 100, 0), 100)
            return round(score, 2)

        except Exception as e:
            print(f"Similarity calculation error: {e}")
            # Fallback: keyword matching
            jd_keywords = set(self.extract_keywords(jd_text, 30))
            resume_keywords = set(self.extract_keywords(resume_text, 50))

            if jd_keywords:
                match_ratio = len(jd_keywords.intersection(resume_keywords)) / len(jd_keywords)
                return round(match_ratio * 100, 2)
            return 0.0

    def load_job_description(self):
        """Load job description from file"""
        filetypes = [
            ('Text files', '*.txt'),
            ('PDF files', '*.pdf'),
            ('All files', '*.*')
        ]

        filepath = filedialog.askopenfilename(
            title="Select Job Description File",
            filetypes=filetypes
        )

        if filepath:
            text = self.extract_text_from_file(filepath)
            if text:
                self.jd_text.delete('1.0', tk.END)
                self.jd_text.insert('1.0', text)
                self.update_status(f"✅ Loaded JD: {os.path.basename(filepath)}")
            else:
                messagebox.showerror("Error", "Could not read the file.")

    def save_job_description(self):
        """Save job description to file"""
        text = self.jd_text.get('1.0', tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Job description is empty.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ('Text files', '*.txt'),
                ('All files', '*.*')
            ]
        )

        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)

                self.update_status(f"💾 Saved JD: {os.path.basename(filepath)}")
                messagebox.showinfo("Success", "Job description saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")

    def add_single_resume(self):
        """Add a single resume file"""
        filetypes = [
            ('PDF files', '*.pdf'),
            ('Text files', '*.txt'),
            ('All files', '*.*')
        ]

        filepaths = filedialog.askopenfilenames(
            title="Select Resume Files",
            filetypes=filetypes
        )

        for filepath in filepaths:
            self.add_resume_to_list(filepath)

    def add_resume_folder(self):
        """Add all resumes from a folder"""
        folder_path = filedialog.askdirectory(title="Select Folder with Resumes")

        if folder_path:
            # Get all supported files
            supported_extensions = ('.pdf', '.txt')

            for filename in os.listdir(folder_path):
                if filename.lower().endswith(supported_extensions):
                    filepath = os.path.join(folder_path, filename)
                    self.add_resume_to_list(filepath)

    def add_resume_to_list(self, filepath):
        """Add resume to the list with proper error handling"""
        try:
            filename = os.path.basename(filepath)

            # Check if already exists
            items = self.resume_listbox.get(0, tk.END)
            if filename in items:
                messagebox.showwarning("Duplicate", f"{filename} is already in the list.")
                return

            # Extract candidate name from filename
            candidate_name = filename.rsplit('.', 1)[0]
            candidate_name = ' '.join([word.capitalize() for word in re.split(r'[_\-\s]+', candidate_name)])

            # Extract text
            text = self.extract_text_from_file(filepath)
            if not text.strip():
                messagebox.showwarning("Warning", f"Could not extract text from {filename}")
                return

            # Add to listbox
            self.resume_listbox.insert(tk.END, filename)

            # Add to DataFrame
            new_row = {
                'filename': filename,
                'candidate_name': candidate_name,
                'content': text,
                'processed_text': self.preprocess_text(text),
                'score': 0.0,
                'filepath': filepath,
                'status': 'Pending'
            }

            if self.resumes.empty:
                self.resumes = pd.DataFrame([new_row])
            else:
                self.resumes = pd.concat([self.resumes, pd.DataFrame([new_row])], ignore_index=True)

            # Update count
            self.resume_count_label.config(text=str(len(self.resumes)))
            self.update_status(f"📄 Added: {filename}")

        except Exception as e:
            print(f"Error adding resume: {e}")
            messagebox.showerror("Error", f"Error processing {filepath}")

    def screen_resumes(self):
        """Screen all resumes against job description"""
        # Check if we have JD and resumes
        jd_text = self.jd_text.get('1.0', tk.END).strip()
        if not jd_text:
            messagebox.showwarning("Warning", "Please enter a job description.")
            return

        if self.resumes.empty:
            messagebox.showwarning("Warning", "Please add some resumes first.")
            return

        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        self.update_status("🔍 Screening resumes...")

        # Calculate scores for each resume
        scores = []
        total_resumes = len(self.resumes)

        for idx, row in self.resumes.iterrows():
            # Update status
            self.update_status(f"Processing {idx + 1}/{total_resumes}: {row['candidate_name']}")
            self.root.update()

            score = self.calculate_similarity(jd_text, row['content'])
            self.resumes.at[idx, 'score'] = score
            scores.append((row['candidate_name'], row['filename'], score))

        # Sort by score
        scores.sort(key=lambda x: x[2], reverse=True)

        # Add to treeview with ranking
        for rank, (candidate, filename, score) in enumerate(scores, 1):
            # Determine status with emojis
            if score >= 80:
                status = "⭐ Strong Match"
                status_color = self.success_color
            elif score >= 60:
                status = "✅ Good Match"
                status_color = self.success_color
            elif score >= 40:
                status = "⚠️ Average"
                status_color = self.warning_color
            else:
                status = "❌ Weak Match"
                status_color = self.danger_color

            # Insert into treeview
            self.results_tree.insert('', 'end',
                                     values=(rank, candidate, f"{score:.1f}%", status))

        # Update status
        top_candidate = scores[0][0] if scores else "None"
        top_score = scores[0][2] if scores else 0
        self.update_status(f"✅ Screening complete! Top match: {top_candidate} ({top_score:.1f}%)")

    def on_resume_select(self, event):
        """Show resume details when selected"""
        selection = self.results_tree.selection()
        if not selection:
            return

        item = self.results_tree.item(selection[0])
        candidate_name = item['values'][1]

        # Find the resume in DataFrame
        resume_row = self.resumes[self.resumes['candidate_name'] == candidate_name]
        if not resume_row.empty:
            self.show_resume_details(resume_row.iloc[0])

    def show_resume_details(self, resume):
        """Show detailed resume analysis"""
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Resume Analysis: {resume['candidate_name']}")
        detail_window.geometry("800x600")
        detail_window.configure(bg=self.bg_color)

        # Title
        title_frame = tk.Frame(detail_window, bg=self.bg_color, pady=20)
        title_frame.pack(fill=tk.X)

        tk.Label(title_frame,
                 text=f"📋 {resume['candidate_name']}",
                 font=('Segoe UI', 20, 'bold'),
                 bg=self.bg_color,
                 fg=self.accent_color).pack()

        # Score display
        score = resume['score']
        score_color = self.success_color if score >= 60 else self.warning_color if score >= 40 else self.danger_color

        score_frame = tk.Frame(detail_window, bg=self.bg_color, pady=10)
        score_frame.pack(fill=tk.X)

        tk.Label(score_frame,
                 text="Match Score: ",
                 font=('Segoe UI', 14),
                 bg=self.bg_color,
                 fg='white').pack(side=tk.LEFT)

        tk.Label(score_frame,
                 text=f"{score:.1f}%",
                 font=('Segoe UI', 16, 'bold'),
                 bg=self.bg_color,
                 fg=score_color).pack(side=tk.LEFT, padx=(5, 0))

        # Resume preview
        preview_frame = tk.Frame(detail_window, bg=self.card_bg)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        preview_text = scrolledtext.ScrolledText(preview_frame,
                                                 font=('Courier', 10),
                                                 bg='#2d3748',
                                                 fg='white',
                                                 wrap=tk.WORD)
        preview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Show first 2000 characters
        content = resume['content']
        preview_text.insert('1.0', content[:2000])
        if len(content) > 2000:
            preview_text.insert('end', "\n\n[...Content truncated...]")

        preview_text.config(state=tk.DISABLED)

    def show_analytics(self):
        """Show analytics dashboard"""
        if self.resumes.empty or 'score' not in self.resumes.columns:
            messagebox.showwarning("Warning", "Please screen resumes first.")
            return

        analytics_window = tk.Toplevel(self.root)
        analytics_window.title("📊 Screening Analytics")
        analytics_window.geometry("1000x700")
        analytics_window.configure(bg=self.bg_color)

        # Title
        tk.Label(analytics_window,
                 text="📊 Screening Analytics",
                 font=('Segoe UI', 24, 'bold'),
                 bg=self.bg_color,
                 fg=self.accent_color).pack(pady=20)

        # Create simple analytics
        scores = self.resumes['score']

        # Statistics frame
        stats_frame = tk.Frame(analytics_window, bg=self.card_bg, padx=20, pady=20)
        stats_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        stats = [
            ("Total Resumes", f"{len(scores)}"),
            ("Average Score", f"{scores.mean():.1f}%"),
            ("Highest Score", f"{scores.max():.1f}%"),
            ("Lowest Score", f"{scores.min():.1f}%"),
            ("Strong Matches", f"{(scores >= 80).sum()}"),
            ("Good Matches", f"{(scores >= 60).sum() - (scores >= 80).sum()}")
        ]

        for i, (label, value) in enumerate(stats):
            frame = tk.Frame(stats_frame, bg=self.card_bg)
            frame.grid(row=i // 3, column=i % 3, padx=10, pady=5, sticky='w')

            tk.Label(frame,
                     text=label,
                     font=('Segoe UI', 10),
                     bg=self.card_bg,
                     fg=self.text_secondary).pack(anchor='w')

            tk.Label(frame,
                     text=value,
                     font=('Segoe UI', 14, 'bold'),
                     bg=self.card_bg,
                     fg='white').pack(anchor='w')

        # Top candidates
        top_frame = tk.Frame(analytics_window, bg=self.card_bg, padx=20, pady=20)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        tk.Label(top_frame,
                 text="🏆 Top 5 Candidates",
                 font=('Segoe UI', 16, 'bold'),
                 bg=self.card_bg,
                 fg='white').pack(anchor='w', pady=(0, 10))

        # Get top 5
        top_candidates = self.resumes.nlargest(5, 'score')

        for idx, row in top_candidates.iterrows():
            candidate_frame = tk.Frame(top_frame, bg='#2d3748', padx=15, pady=10)
            candidate_frame.pack(fill=tk.X, pady=5)

            tk.Label(candidate_frame,
                     text=f"{row['candidate_name']}",
                     font=('Segoe UI', 12),
                     bg='#2d3748',
                     fg='white').pack(side=tk.LEFT)

            tk.Label(candidate_frame,
                     text=f"{row['score']:.1f}%",
                     font=('Segoe UI', 12, 'bold'),
                     bg='#2d3748',
                     fg=self.accent_color).pack(side=tk.RIGHT)

    def export_to_excel(self):
        """Export results to Excel"""
        if self.resumes.empty or 'score' not in self.resumes.columns:
            messagebox.showwarning("Warning", "No data to export. Please screen resumes first.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[('Excel files', '*.xlsx'), ('All files', '*.*')]
        )

        if filepath:
            try:
                export_data = self.resumes[['candidate_name', 'filename', 'score', 'filepath']].copy()
                export_data = export_data.sort_values('score', ascending=False)
                export_data['Rank'] = range(1, len(export_data) + 1)
                export_data = export_data[['Rank', 'candidate_name', 'score', 'filename', 'filepath']]

                export_data.to_excel(filepath, index=False)
                self.update_status(f"💾 Exported to: {os.path.basename(filepath)}")
                messagebox.showinfo("Success", "Results exported to Excel successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export: {str(e)}")

    def clear_all(self):
        """Clear all data"""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all data?"):
            # Clear listbox
            self.resume_listbox.delete(0, tk.END)

            # Clear results tree
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            # Clear DataFrame
            self.resumes = pd.DataFrame(
                columns=['filename', 'candidate_name', 'content', 'processed_text', 'score', 'filepath', 'status'])

            # Update count
            self.resume_count_label.config(text="0")

            self.update_status("✨ All data cleared. Ready for new screening session.")

    def update_status(self, message):
        """Update status bar"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_bar.config(text=f"[{timestamp}] {message}")
        self.root.update()


def main():
    root = tk.Tk()

    # Set icon if available
    try:
        root.iconbitmap(default='icon.ico')
    except:
        pass

    # Center window
    window_width = 1400
    window_height = 850
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Create application
    app = ResumeScreenerAI(root)

    # Make window responsive
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Check for required packages
    try:
        import pandas
        import matplotlib
    except ImportError:
        messagebox.showwarning(
            "Missing Dependencies",
            "Please install required packages:\n\n"
            "pip install pandas matplotlib wordcloud PyPDF2 scikit-learn"
        )

    root.mainloop()


if __name__ == "__main__":
    main()