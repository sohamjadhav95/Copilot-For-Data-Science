import tkinter as tk
from tkinter import ttk, scrolledtext, font as tkfont
from tkinter import messagebox
import sys
import traceback
import os
import threading
import io
from datetime import datetime
import textwrap

sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 3 Human in the Loop (Interactive AI)\v3.3 UI Proposals\Machine_Learning")
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 3 Human in the Loop (Interactive AI)\v3.3 UI Proposals\Core_Automation_Engine")
sys.path.append(r"E:\Projects\Copilot-For-Data-Science\Copilote for data science\Phase 3 Human in the Loop (Interactive AI)\v3.3 UI Proposals\Retrival_Agumented_Generation")


# Import modules - Display friendly error if imports fail
try:
    from NL_processor import NL_processor, split_multi_commands, genral_response_chatbot
    from Engine_Display_data import Groq_Input as Display_Groq_Input
    from Engine_Visualize_data import Groq_Input as Visualize_Groq_Input
    from Engine_Modify_data import Groq_Input as Modify_Groq_Input, undo_last_change
    from Engine_Data_analysis import DataAnalysisReport, Dashboard
    from rag_command_parser import basic_rag
    from ML_Models_Engine_autogluon import SupervisedUniversalMachineLearning
    from groq import Groq
except ImportError as e:
    print(f"Import Error: {e}. Please check that all required modules are installed and paths are correct.")
    import tkinter.messagebox as msgbox
    msgbox.showerror("Import Error", 
                     f"Error importing required modules: {e}\n\nPlease check that all required modules are installed and paths are correct.")
    sys.exit(1)

# Initialize Groq client with API key
api_key = "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"
client = Groq(api_key=api_key)

# Initialize ML model and other objects
ML = SupervisedUniversalMachineLearning()
dashboard = Dashboard()
data_analysis_report = DataAnalysisReport()

# Output capture class
class OutputCapture:
    def __init__(self):
        self.buffer = io.StringIO()
        self.stdout = sys.stdout
        
    def __enter__(self):
        sys.stdout = self.buffer
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        
    def get_output(self):
        return self.buffer.getvalue()

def explain_error(error_message):
    """Use Groq API to dynamically explain errors"""
    try:
        prompt = f"""
        You are an AI assistant that helps developers debug Python errors.
        Given the following error message, provide an explanation and a solution.

        Error Message:
        {error_message}

        Explanation and Suggested Fix:
        """

        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1024,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        return completion.choices[0].message.content.strip()
    except Exception:
        return "Error explanation service is unavailable. Please check your API connection."

def safe_execute(operation_function, *args):
    """Safely executes any function with error handling and captures output."""
    with OutputCapture() as output:
        try:
            result = operation_function(*args)
            return output.get_output(), None
        except Exception as e:
            error_message = traceback.format_exc()
            explanation = explain_error(error_message)
            error_output = f"\n[ERROR] An error occurred:\n{explanation}"
            return output.get_output(), error_output

def route_command(operation, user_input):
    """Routes command execution based on operation type."""
    output = ""
    error = None
    
    refined_user_input_by_rag = basic_rag(user_input)
    operation_output = f"\nExecuting command: {refined_user_input_by_rag}\n"
    
    if operation == "visualize":
        operation_output += "Visualizing data...\n"
        func_output, func_error = safe_execute(Visualize_Groq_Input, refined_user_input_by_rag)
    elif operation == "display":
        operation_output += "Displaying data...\n"
        func_output, func_error = safe_execute(Display_Groq_Input, refined_user_input_by_rag)
    elif operation == "modify":
        operation_output += "Modifying data...\n"
        func_output, func_error = safe_execute(Modify_Groq_Input, refined_user_input_by_rag)
    elif operation == "undo":
        operation_output += "Undoing last modification...\n"
        func_output, func_error = safe_execute(undo_last_change)
    elif operation == "meaningful_response":
        operation_output += "Generating a response...\n"
        func_output, func_error = safe_execute(genral_response_chatbot, refined_user_input_by_rag)
    elif operation == "analyze_data":
        operation_output += "Analyzing data...\n"
        func_output, func_error = safe_execute(data_analysis_report.generate_insights, refined_user_input_by_rag)
    elif operation == "generate_report":
        operation_output += "Generating PDF report...\n"
        func_output, func_error = safe_execute(data_analysis_report.generate_pdf_report, refined_user_input_by_rag)
    elif operation == "create_dashboard":
        operation_output += "Creating dashboard...\n"
        func_output, func_error = safe_execute(dashboard.create_dashboard, refined_user_input_by_rag)
    elif operation == "build_model":
        operation_output += "Building ML model...\n"
        func_output, func_error = safe_execute(ML.build_model_and_test, refined_user_input_by_rag)
    elif operation == "test_model":
        operation_output += "Testing ML model...\n"
        func_output, func_error = safe_execute(ML.build_model_and_test, refined_user_input_by_rag)
    elif operation == "deploy_model":
        operation_output += "Deploying ML model...\n"
        func_output, func_error = safe_execute(ML.deploy_model, refined_user_input_by_rag)
    elif operation == "predict_custom_input":
        operation_output += "Predicting custom input...\n"
        func_output, func_error = safe_execute(ML.predict_custom_input, refined_user_input_by_rag)
    elif operation == "os_operations":
        operation_output += "Performing OS operations...\n"
        try:
            from OS_operations import OS_Operation
            func_output, func_error = safe_execute(OS_Operation, refined_user_input_by_rag)
        except ImportError:
            func_output = ""
            func_error = "OS_operations module not found."
    else:
        operation_output += "Unable to determine the operation. Please try again.\n"
        func_output = ""
        func_error = None
    
    output = operation_output + (func_output if func_output else "")
    error = func_error
    
    return output, error

def execute_multiple_commands(user_input):
    """Execute multiple commands sequentially with error handling."""
    all_output = ""
    all_errors = []
    
    commands = split_multi_commands(user_input)  # Split the input into commands
    for command in commands:
        if command:
            refined_user_input_by_rag = basic_rag(command)
            operation = NL_processor(refined_user_input_by_rag)
            output, error = route_command(operation, refined_user_input_by_rag)
            all_output += output + "\n"
            if error:
                all_errors.append(error)
    
    return all_output, all_errors

class DataScienceCopilotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Science Copilot")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set app icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # Configure the grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=4)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Setup theme and styling
        self.setup_style()
        
        # Create the main container frames
        self.create_sidebar()
        self.create_main_panel()
        
        # Chat history list
        self.chat_history = []
        
        # Status bar at the bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        
    def setup_style(self):
        """Set up the visual style for the application"""
        self.style = ttk.Style()
        
        # Try to use a modern theme if available
        try:
            self.style.theme_use("clam")  # 'clam', 'alt', 'default', 'classic'
        except:
            pass
            
        # Configure colors
        bg_color = "#f0f0f0"
        accent_color = "#4a6fa5"
        
        # Configure fonts
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=10)
        
        text_font = tkfont.Font(family="Consolas", size=10)
        
        # Configure styles
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TButton", padding=6, relief="flat", background=accent_color)
        self.style.configure("TNotebook", background=bg_color)
        self.style.configure("TNotebook.Tab", padding=[12, 4], font=('Segoe UI', 10))
        self.style.configure("Sidebar.TFrame", background="#e0e0e0")
        self.style.configure("Main.TFrame", background=bg_color)
        self.style.configure("Chat.TFrame", background="#ffffff")
        
        return text_font
        
    def create_sidebar(self):
        """Create the sidebar with available commands"""
        self.sidebar_frame = ttk.Frame(self.root, style="Sidebar.TFrame", padding="10")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        # Add logo or title at the top
        logo_label = ttk.Label(self.sidebar_frame, text="Data Science Copilot", font=('Segoe UI', 14, 'bold'))
        logo_label.pack(pady=10, fill="x")
        
        # Separator
        ttk.Separator(self.sidebar_frame, orient="horizontal").pack(fill="x", pady=5)
        
        # Command categories
        categories = {
            "Data Operations": ["Display data", "Visualize data", "Modify data"],
            "Analysis": ["Analyze data", "Generate report", "Create dashboard"],
            "Machine Learning": ["Build model", "Test model", "Deploy model", "Predict"],
        }
        
        # Create command buttons grouped by category
        for category, commands in categories.items():
            # Category label
            cat_frame = ttk.LabelFrame(self.sidebar_frame, text=category)
            cat_frame.pack(fill="x", pady=5, padx=2)
            
            # Command buttons
            for cmd in commands:
                cmd_btn = ttk.Button(cat_frame, text=cmd, 
                                     command=lambda c=cmd: self.insert_command_template(c))
                cmd_btn.pack(fill="x", padx=2, pady=2)
        
        # Quick actions section at the bottom
        quick_frame = ttk.LabelFrame(self.sidebar_frame, text="Quick Actions")
        quick_frame.pack(fill="x", pady=5, padx=2, side="bottom")
        
        clear_btn = ttk.Button(quick_frame, text="Clear History", command=self.clear_history)
        clear_btn.pack(fill="x", padx=2, pady=2)
        
        undo_btn = ttk.Button(quick_frame, text="Undo Last Change", 
                             command=lambda: self.process_command("undo"))
        undo_btn.pack(fill="x", padx=2, pady=2)
    
    def create_main_panel(self):
        """Create the main panel with chat interface"""
        self.main_frame = ttk.Frame(self.root, style="Main.TFrame")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        
        self.main_frame.grid_rowconfigure(0, weight=8)
        self.main_frame.grid_rowconfigure(1, weight=0)  # Separator
        self.main_frame.grid_rowconfigure(2, weight=1)  # Input area
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Chat display area (where conversation appears)
        self.chat_frame = ttk.Frame(self.main_frame, style="Chat.TFrame")
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure the chat frame for scrolling
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1)
        
        # Scrollable chat area
        self.chat_canvas = tk.Canvas(self.chat_frame, bg="#ffffff", highlightthickness=0)
        self.chat_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbar for chat
        self.chat_scrollbar = ttk.Scrollbar(self.chat_frame, orient="vertical", command=self.chat_canvas.yview)
        self.chat_scrollbar.grid(row=0, column=1, sticky="ns")
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        
        # Frame inside canvas for chat messages
        self.chat_messages_frame = ttk.Frame(self.chat_canvas, style="Chat.TFrame")
        self.chat_canvas_window = self.chat_canvas.create_window((0, 0), window=self.chat_messages_frame, anchor="nw")
        
        # Configure the chat messages frame
        self.chat_messages_frame.bind("<Configure>", lambda e: self.chat_canvas.configure(
            scrollregion=self.chat_canvas.bbox("all"),
            width=e.width))
        
        # Make sure the canvas expands with the window
        self.chat_canvas.bind("<Configure>", lambda e: self.chat_canvas.itemconfig(
            self.chat_canvas_window, width=e.width))
        
        # Separator between chat and input
        ttk.Separator(self.main_frame, orient="horizontal").grid(row=1, column=0, sticky="ew", padx=10)
        
        # Input area at the bottom
        input_frame = ttk.Frame(self.main_frame)
        input_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(1, weight=0)
        
        # Text input field
        self.input_text = scrolledtext.ScrolledText(input_frame, height=3, wrap=tk.WORD, 
                                                   font=('Consolas', 10))
        self.input_text.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.input_text.bind("<Control-Return>", self.on_enter_key)
        
        # Send button
        send_button = ttk.Button(input_frame, text="Send", command=self.on_send)
        send_button.grid(row=0, column=1, sticky="ns")
        
        # Add a small hint below the input
        hint_label = ttk.Label(input_frame, text="Press Ctrl+Enter to send", font=('Segoe UI', 8))
        hint_label.grid(row=1, column=0, sticky="w", pady=(2, 0))
    
    def insert_command_template(self, command):
        """Insert command template into the input field"""
        command_templates = {
            "Display data": "Show me the first 5 rows of the dataset",
            "Visualize data": "Create a bar chart of ",
            "Modify data": "Clean missing values in the ",
            "Analyze data": "Analyze trends in the ",
            "Generate report": "Generate a PDF report of the analysis results",
            "Create dashboard": "Create a dashboard showing ",
            "Build model": "Build a classification model to predict ",
            "Test model": "Test the model on ",
            "Deploy model": "Deploy the model as ",
            "Predict": "Predict using the model with inputs "
        }
        
        template = command_templates.get(command, command)
        self.input_text.delete(1.0, tk.END)
        self.input_text.insert(tk.END, template)
        self.input_text.focus()
    
    def on_enter_key(self, event):
        """Handle Ctrl+Enter key press"""
        self.on_send()
        return "break"  # Prevents the default behavior
    
    def on_send(self):
        """Process the command when send button is clicked"""
        user_input = self.input_text.get(1.0, tk.END).strip()
        if not user_input:
            return
            
        # Clear the input field
        self.input_text.delete(1.0, tk.END)
        
        # Add user message to chat
        self.add_message("You", user_input)
        
        # Process the command in a separate thread to keep UI responsive
        self.set_status("Processing...")
        threading.Thread(target=self.process_command, args=(user_input,), daemon=True).start()
    
    def process_command(self, user_input):
        """Process the user command"""
        try:
            if user_input.lower() == "undo":
                # Special case for undo command
                output, error = safe_execute(undo_last_change)
                response = "Undoing last change...\n" + (output if output else "")
            elif "then" in user_input.lower() or "and" in user_input.lower() or "after" in user_input.lower():
                # Multiple commands
                output, errors = execute_multiple_commands(user_input)
                response = output
                if errors:
                    response += "\n\n" + "\n".join([e for e in errors if e])
            else:
                # Single command
                operation = NL_processor(user_input)
                output, error = route_command(operation, user_input)
                response = output
                if error:
                    response += "\n\n" + error
            
            # Add response to chat
            self.root.after(0, lambda: self.add_message("Copilot", response))
            
        except Exception as e:
            error_msg = f"Error processing command: {str(e)}\n{traceback.format_exc()}"
            self.root.after(0, lambda: self.add_message("Copilot", error_msg))
        
        # Reset status
        self.root.after(0, lambda: self.set_status("Ready"))
    
    def add_message(self, sender, message):
        """Add a message to the chat"""
        # Create a frame for this message
        msg_frame = ttk.Frame(self.chat_messages_frame)
        msg_frame.pack(fill="x", padx=5, pady=5)
        
        # Style based on sender
        if sender == "You":
            header_style = {'font': ('Segoe UI', 10, 'bold'), 'foreground': '#303f9f'}
            bg_color = "#e3f2fd"
            align = "e"
        else:
            header_style = {'font': ('Segoe UI', 10, 'bold'), 'foreground': '#00695c'}
            bg_color = "#f1f8e9"
            align = "w"
        
        # Message header with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        header = ttk.Label(msg_frame, text=f"{sender} - {timestamp}", **header_style)
        header.pack(fill="x", anchor=align)
        
        # Message content
        content_frame = tk.Frame(msg_frame, bg=bg_color, bd=1, relief=tk.GROOVE)
        content_frame.pack(fill="x", padx=5, pady=2)
        
        # Format and wrap the message content
        wrapped_lines = []
        for line in message.split('\n'):
            if len(line) > 80:
                wrapped_lines.extend(textwrap.wrap(line, width=80))
            else:
                wrapped_lines.append(line)
        
        formatted_message = '\n'.join(wrapped_lines)
        
        content = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, height=min(15, len(wrapped_lines) + 2),
                                          font=('Consolas', 9), bg=bg_color)
        content.pack(fill="both", expand=True, padx=5, pady=5)
        content.insert(tk.END, formatted_message)
        content.configure(state="disabled")  # Make it read-only
        
        # Store in history
        self.chat_history.append((sender, message))
        
        # Scroll to see the latest message
        self.chat_canvas.update_idletasks()
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        self.chat_canvas.yview_moveto(1.0)
    
    def clear_history(self):
        """Clear the chat history"""
        # Destroy all messages
        for widget in self.chat_messages_frame.winfo_children():
            widget.destroy()
        
        # Clear history list
        self.chat_history = []
        
        # Add a system message
        self.add_message("System", "Chat history cleared.")
    
    def set_status(self, message):
        """Update the status bar message"""
        self.status_var.set(message)

# Entry point
def main():
    root = tk.Tk()
    app = DataScienceCopilotApp(root)
    
    # Add a welcome message
    app.add_message("System", "Welcome to Data Science Copilot! Type your command below or use the templates from the sidebar.")
    
    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()